#!/usr/bin/env python3
"""
Multi-Agent Orchestrator
========================

The main engine that runs multi-agent experiments with:
- Flexible agent configurations
- Custom interaction topologies
- Per-agent injection control
- Comprehensive metrics tracking

Usage:
    orchestrator = MultiAgentOrchestrator(model, tokenizer, device)
    orchestrator.load_registry(registry)
    orchestrator.set_interaction_graph(graph)
    results = orchestrator.run_experiment(question, num_rounds=5)
"""

import os
import sys
import json
import time
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from .agent_registry import AgentRegistry, AgentConfig, AgentBehavior
from .interaction_graph import InteractionGraph, InteractionEdge, ConversationState, EdgeType
from .model_compatibility import ModelCompatibility, load_model_and_tokenizer, get_device


@dataclass
class ProbeConfig:
    """Configuration for a linear probe."""
    category: str
    direction: np.ndarray
    layer_idx: int
    hidden_size: int
    
    def project(self, activation: np.ndarray) -> float:
        """Project activation onto probe direction."""
        norm_dir = self.direction / (np.linalg.norm(self.direction) + 1e-8)
        return float(np.dot(activation, norm_dir))


@dataclass
class InjectionConfig:
    """Configuration for activation injection."""
    injection_type: str = "gated"  # "gated", "amplify", "ablate", "steer"
    strength: float = 2.0
    gate_temp: float = 0.1
    gate_bias: float = -0.3
    target_dims: Optional[List[int]] = None  # For "amplify" mode
    direction: str = "add" # "add" or "subtract"


@dataclass
class AgentMetrics:
    """Metrics collected for a single agent across the experiment."""
    agent_id: str
    probe_scores: List[float] = field(default_factory=list)  # Per-turn scores
    token_scores: List[List[float]] = field(default_factory=list)  # Per-token scores per turn
    token_strings: List[List[str]] = field(default_factory=list)  # Actual tokens per turn
    injection_calls: int = 0
    total_tokens_generated: int = 0
    response_lengths: List[int] = field(default_factory=list)
    
    def mean_score(self) -> float:
        return float(np.mean(self.probe_scores)) if self.probe_scores else 0.0
    
    def score_trajectory(self) -> List[float]:
        return self.probe_scores
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "probe_scores": self.probe_scores,
            "token_scores": self.token_scores,
            "token_strings": self.token_strings,
            "mean_score": self.mean_score(),
            "injection_calls": self.injection_calls,
            "total_tokens": self.total_tokens_generated,
            "response_lengths": self.response_lengths
        }


@dataclass
class ExperimentResult:
    """Complete results from a multi-agent experiment."""
    experiment_id: str
    question: str
    num_rounds: int
    
    # Agent metrics
    agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    
    # Conversation
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    
    # Aggregate metrics
    disagreement_scores: List[float] = field(default_factory=list)  # Std of scores per round
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Metadata
    config: Dict[str, Any] = field(default_factory=dict)
    
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "question": self.question,
            "num_rounds": self.num_rounds,
            "agent_metrics": {k: v.to_dict() for k, v in self.agent_metrics.items()},
            "transcript": self.transcript,
            "disagreement_scores": self.disagreement_scores,
            "duration_seconds": self.duration(),
            "config": self.config
        }
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class InjectableAgent:
    """
    A single agent that can generate responses with optional injection.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        model,
        tokenizer,
        probe: Optional[ProbeConfig],
        device: str,
        model_compat: Optional[ModelCompatibility] = None
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe
        self.device = device
        self.model_compat = model_compat or ModelCompatibility(model, tokenizer)
        
        if probe and probe.layer_idx is not None:
            self.target_layer = probe.layer_idx
        else:
            self.target_layer = self.model_compat.num_layers // 2
        
        self._current_activation: Optional[np.ndarray] = None
        self._handles: List = []
        self._injection_stats = {"calls": 0, "total_delta": 0.0, "gated_skips": 0}

    def _create_injection_hook(
        self,
        strength: float,
        direction: str = "add",
        gate_threshold: float = 0.0,
        gate_direction: str = "below",
    ):
        """Create a steering injection hook with dynamic scaling and gating.

        Dynamic scaling: injection magnitude is proportional to the current
        residual-stream norm rather than a one-time dummy-input estimate.

        Gated injection: the hook reads the probe score from the monitor
        layer and only injects when the model is drifting from desired
        behavior.
        """
        if self.probe is None:
            return None

        vector_tensor = torch.tensor(
            self.probe.direction, dtype=torch.float32
        ).to(self.device)
        vector_tensor = F.normalize(vector_tensor, p=2, dim=0)

        sign = -1.0 if direction == "subtract" else 1.0
        probe_direction = self.probe.direction

        def hook(module, input, output):
            act = output[0] if isinstance(output, tuple) else output

            # Dimension compatibility check
            if act.shape[-1] != vector_tensor.shape[0]:
                return output

            # --- Gated injection: read before write ---
            if self._current_activation is not None:
                norm_dir = probe_direction / (
                    np.linalg.norm(probe_direction) + 1e-8
                )
                current_score = float(
                    np.dot(self._current_activation.flatten(), norm_dir.flatten())
                )
                if gate_direction == "below" and current_score >= gate_threshold:
                    self._injection_stats["gated_skips"] = (
                        self._injection_stats.get("gated_skips", 0) + 1
                    )
                    return output
                elif gate_direction == "above" and current_score <= gate_threshold:
                    self._injection_stats["gated_skips"] = (
                        self._injection_stats.get("gated_skips", 0) + 1
                    )
                    return output

            # --- Dynamic scaling ---
            last_token_act = act[:, -1, :]
            stream_norm = last_token_act.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dynamic_scale = strength * sign * stream_norm * 0.1

            vector_t = vector_tensor.to(dtype=act.dtype)

            modified = act.clone()
            modified[:, -1, :] = modified[:, -1, :] + dynamic_scale * vector_t

            self._injection_stats["calls"] += 1

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook

    def _register_hooks(self, injection_config: Optional[InjectionConfig] = None):
        """Register forward hooks for monitoring and multi-layer injection."""
        for h in self._handles:
            h.remove()
        self._handles = []

        num_layers = self.model_compat.num_layers

        # Injection hooks across multiple layers
        if injection_config and self.probe:
            strength = injection_config.strength
            direction = getattr(injection_config, 'direction', 'add')
            gate_bias = getattr(injection_config, 'gate_bias', 0.0)
            gate_dir = "below" if direction == "add" else "above"

            inject_start = num_layers // 4
            inject_end = num_layers - 1
            n_inject_layers = inject_end - inject_start
            per_layer_strength = strength / max(n_inject_layers ** 0.65, 1.0)

            injected_count = 0
            for layer_idx in range(inject_start, inject_end):
                if layer_idx == self.target_layer:
                    continue  # skip monitor layer for clean probe reading
                layer_mod = self.model_compat.get_layer(layer_idx)
                if layer_mod is None:
                    continue
                hook_fn = self._create_injection_hook(
                    per_layer_strength,
                    direction,
                    gate_threshold=gate_bias,
                    gate_direction=gate_dir,
                )
                if hook_fn:
                    self._handles.append(layer_mod.register_forward_hook(hook_fn))
                    injected_count += 1

            if injected_count > 0:
                print(f"[Injection] Registered on {injected_count} layers ({inject_start}-{inject_end-1}), gating={gate_dir}@{gate_bias}")

        # Monitor hook on target layer (always)
        monitor_module = self.model_compat.get_layer(self.target_layer)
        if monitor_module is None:
            print(f"Warning: Could not find monitor layer {self.target_layer}")
            return

        def monitor_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hasattr(hidden, "shape") and len(hidden.shape) >= 3:
                self._current_activation = hidden[0, -1, :].detach().float().cpu().numpy()

        self._handles.append(monitor_module.register_forward_hook(monitor_hook))
    
    def get_current_score(self) -> float:
        """Get probe score from current activation."""
        if self._current_activation is None or self.probe is None:
            return 0.0
        return self.probe.project(self._current_activation)
    
    def generate_response(
        self,
        prompt: str,
        context: str = "",
        injection_config: Optional[InjectionConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate response with optional injection."""
        self._register_hooks(injection_config)

        full_prompt = f"{self.config.system_prompt}\n\n{context}{prompt}\n\nYour response:"

        # === Main Generation ===
        model_limit = self.model_compat.max_position_embeddings
        safe_input_length = max(50, model_limit - self.config.max_tokens - 1)
        
        inputs = self.tokenizer(
            full_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=safe_input_length
        )
        input_ids = inputs["input_ids"].to(self.device)
        
        scores = []
        tokens = []
        generated_ids = input_ids.clone()
        
        for _ in range(self.config.max_tokens):
            with torch.no_grad():
                attention_mask = torch.ones_like(generated_ids)
                outputs = self.model(generated_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :] / self.config.temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            score = self.get_current_score()
            scores.append(score)
            tokens.append(self.tokenizer.decode(next_token[0]))
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            if generated_ids.shape[1] >= self.model_compat.max_position_embeddings:
                break
        
        for h in self._handles:
            h.remove()
        self._handles = []
        
        response_text = self.tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=True
        )[len(full_prompt):].strip()
        
        return {
            "agent_id": self.config.agent_id,
            "response_text": response_text,
            "scores": scores,
            "tokens": tokens,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "shadow_log": {},
            "injection_stats": self._injection_stats.copy()
        }


class MultiAgentOrchestrator:
    """
    Main orchestrator for multi-agent experiments.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        probe: Optional[ProbeConfig] = None,
        model_compat: Optional[ModelCompatibility] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.probe = probe
        self.model_compat = model_compat or ModelCompatibility(model, tokenizer)
        
        self.registry: Optional[AgentRegistry] = None
        self.graph: Optional[InteractionGraph] = None
        self.agents: Dict[str, InjectableAgent] = {}
        self.injection_targets: Dict[str, InjectionConfig] = {}
        
        self._on_turn_complete: List[Callable] = []
        self._on_round_complete: List[Callable] = []
        
        print(f"Orchestrator initialized: {self.model_compat}")
    
    def load_registry(self, registry: AgentRegistry):
        """Load an agent registry and create agent instances."""
        self.registry = registry
        self.agents = {}
        for agent_id, config in registry.get_all().items():
            self.agents[agent_id] = InjectableAgent(
                config=config,
                model=self.model,
                tokenizer=self.tokenizer,
                probe=self.probe,
                device=self.device,
                model_compat=self.model_compat
            )
        print(f"Loaded {len(self.agents)} agents: {list(self.agents.keys())}")
    
    def set_interaction_graph(self, graph: InteractionGraph):
        """Set the interaction topology."""
        self.graph = graph
        for node in graph._nodes:
            if node not in self.agents:
                raise ValueError(f"Graph contains unknown agent: {node}")
        print(f"Interaction graph set: {len(graph._nodes)} nodes, {len(graph._edges)} edges")
    
    def set_injection(self, agent_id: str, config: InjectionConfig):
        """Configure injection for a specific agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        self.injection_targets[agent_id] = config
        print(f"Injection configured for {agent_id}: {config.injection_type} x{config.strength}")
    
    def clear_injections(self):
        """Remove all injection configurations."""
        self.injection_targets.clear()
    
    def run_experiment(
        self,
        question: str,
        num_rounds: int = 4,
        start_agent: Optional[str] = None,
        verbose: bool = True,
        **kwargs,
    ) -> ExperimentResult:
        """Run a multi-agent experiment."""
        if not self.graph:
            raise ValueError("No interaction graph set. Call set_interaction_graph() first.")

        result = ExperimentResult(
            experiment_id=f"exp_{int(time.time())}",
            question=question,
            num_rounds=num_rounds,
            start_time=time.time(),
            config={
                "agents": list(self.agents.keys()),
                "injection_targets": list(self.injection_targets.keys()),
            }
        )
        
        for agent_id in self.agents:
            result.agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
        
        execution_order = self.graph.compute_execution_order(start_agent)
        if verbose:
            print(f"\nExecution order: {' → '.join(execution_order)}")
        
        state = ConversationState()
        
        for round_num in range(1, num_rounds + 1):
            state.round_number = round_num
            if verbose:
                print(f"\n{'='*60}\nROUND {round_num}\n{'='*60}")
            
            round_scores = []
            
            for agent_id in execution_order:
                agent = self.agents[agent_id]
                
                # Context Management
                incoming = self.graph.get_incoming(agent_id)
                context_window = 3
                edge_prompt = ""
                if incoming:
                    edge = self.graph.get_edge(incoming[0], agent_id)
                    if edge:
                        context_window = edge.context_window
                        edge_prompt = edge.edge_prompt or ""
                
                context = state.get_context_for_agent(agent_id, context_window)
                
                # Prompting
                if round_num == 1 and agent_id == execution_order[0]:
                    prompt = question
                else:
                    prompt = edge_prompt or "Continue the discussion."
                
                # Injection
                injection_config = self.injection_targets.get(agent_id)
                
                if verbose:
                    inj_str = f" [INJECTED: {injection_config.injection_type}]" if injection_config else ""
                    print(f"\n[{agent.config.display_name}]{inj_str}")
                
                # Generate
                response = agent.generate_response(
                    prompt=prompt,
                    context=context,
                    injection_config=injection_config,
                )
                
                if verbose:
                    print(f"  \"{response['response_text'][:100]}...\"")
                    print(f"  Score: {response['mean_score']:.3f}")
                
                state.add_message(
                    agent_id=agent_id,
                    content=response["response_text"],
                    metadata={"score": response["mean_score"]}
                )
                
                # Update Metrics
                metrics = result.agent_metrics[agent_id]
                metrics.probe_scores.append(response["mean_score"])
                metrics.token_scores.append(response["scores"])
                metrics.token_strings.append(response["tokens"])
                metrics.total_tokens_generated += len(response["tokens"])
                metrics.response_lengths.append(len(response["response_text"]))
                
                if injection_config:
                    metrics.injection_calls += response["injection_stats"].get("calls", 0)
                
                result.transcript.append({
                    "round": round_num,
                    "agent": agent_id,
                    "text": response["response_text"],
                    "score": response["mean_score"],
                })
                round_scores.append(response["mean_score"])
                
                for callback in self._on_turn_complete:
                    callback(agent_id, response)
            
            disagreement = float(np.std(round_scores)) if len(round_scores) > 1 else 0.0
            result.disagreement_scores.append(disagreement)
            
            for callback in self._on_round_complete:
                callback(round_num, state)
        
        result.end_time = time.time()
        return result
    
    def run_comparison(
        self,
        question: str,
        target_agents: List[str],
        injection_config: InjectionConfig,
        num_rounds: int = 4,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, ExperimentResult]:
        """Run comparison experiment."""
        results = {}

        if verbose:
            print(f"\n{'='*60}\nBASELINE (No Injection)\n{'='*60}")

        self.clear_injections()
        results["baseline"] = self.run_experiment(
            question=question,
            num_rounds=num_rounds,
            verbose=verbose
        )

        if verbose:
            print(f"\n{'='*60}\nINJECTED ({injection_config.injection_type} x{injection_config.strength})\nTargets: {target_agents}\n{'='*60}")

        for agent_id in target_agents:
            self.set_injection(agent_id, injection_config)

        results["injected"] = self.run_experiment(
            question=question,
            num_rounds=num_rounds,
            verbose=verbose
        )

        return results


def quick_setup(
    model,
    tokenizer,
    device: str,
    topology: str = "debate",
    probe_path: Optional[str] = None,
    probe_category: str = "overconfidence"
) -> MultiAgentOrchestrator:
    """Quick setup factory."""
    probe = None
    if probe_path:
        import pickle
        with open(probe_path, 'rb') as f:
            probe_data = pickle.load(f)
        if probe_category in probe_data:
            p = probe_data[probe_category]
            # Handle different pickle formats
            direction = p.get("direction") or p.get("weights")
            metadata = p.get("metadata", {})
            probe = ProbeConfig(
                category=probe_category,
                direction=np.array(direction),
                layer_idx=metadata.get("layer_idx", 12),
                hidden_size=metadata.get("hidden_size", 4096)
            )
    
    orchestrator = MultiAgentOrchestrator(model, tokenizer, device, probe)
    registry = AgentRegistry()
    
    if topology == "debate":
        registry.register_from_template("proposer")
        registry.register_from_template("critic")
        registry.register_from_template("judge")
        graph = InteractionGraph.create_debate_topology(["proposer", "critic", "judge"])
    elif topology == "panel":
        registry.register_from_template("mediator", "moderator")
        registry.register_from_template("researcher", "expert_1")
        registry.register_from_template("devil_advocate", "expert_2")
        graph = InteractionGraph.create_panel_discussion("moderator", ["expert_1", "expert_2"])
    else:
        raise ValueError(f"Unknown topology: {topology}")
        
    orchestrator.load_registry(registry)
    orchestrator.set_interaction_graph(graph)
    return orchestrator

# =============================================================================
# FACTORY FUNCTIONS (Append this to the end of core/orchestrator.py)
# =============================================================================

def create_orchestrator_from_config(
    model,
    tokenizer,
    device: str,
    config_path: str,
    probe_path: Optional[str] = None,
    probe_category: str = "overconfidence"
) -> MultiAgentOrchestrator:
    """
    Create an orchestrator from config files.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        device: Device string
        config_path: Path to JSON config with registry and graph
        probe_path: Optional path to probe pickle file
        probe_category: Which probe category to use
    """
    # Load probe
    probe = None
    if probe_path:
        with open(probe_path, 'rb') as f:
            probe_data = pickle.load(f)
        if probe_category in probe_data:
            p = probe_data[probe_category]
            # Handle different pickle formats (PAS vs Legacy)
            direction = p.get("direction") or p.get("weights")
            metadata = p.get("metadata", {})
            
            probe = ProbeConfig(
                category=probe_category,
                direction=np.array(direction),
                layer_idx=metadata.get("layer_idx", 12), # Default fallback
                hidden_size=metadata.get("hidden_size", 4096)
            )
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(model, tokenizer, device, probe)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load registry
    registry = AgentRegistry()
    for agent_data in config.get("agents", []):
        registry.register(**agent_data)
    orchestrator.load_registry(registry)
    
    # Load graph
    if "graph" in config:
        graph = InteractionGraph()
        for node in config["graph"].get("nodes", []):
            graph.add_node(node)
        for edge in config["graph"].get("edges", []):
            graph.add_edge(**edge)
        orchestrator.set_interaction_graph(graph)
    
    return orchestrator


def quick_setup(
    model,
    tokenizer,
    device: str,
    topology: str = "debate",
    probe_path: Optional[str] = None,
    probe_category: str = "overconfidence"
) -> MultiAgentOrchestrator:
    """
    Quick setup with predefined topologies.
    
    Args:
        topology: "debate", "panel", "adversarial"
    """
    # Load probe
    probe = None
    if probe_path and os.path.exists(probe_path):
        import pickle
        with open(probe_path, 'rb') as f:
            probe_data = pickle.load(f)
        if probe_category in probe_data:
            p = probe_data[probe_category]
            direction = p.get("direction") or p.get("weights")
            metadata = p.get("metadata", {})
            probe = ProbeConfig(
                category=probe_category,
                direction=np.array(direction),
                layer_idx=metadata.get("layer_idx", 12),
                hidden_size=metadata.get("hidden_size", 4096)
            )
    
    orchestrator = MultiAgentOrchestrator(model, tokenizer, device, probe)
    
    # Create registry with predefined agents
    from .agent_registry import AgentRegistry
    registry = AgentRegistry()
    
    if topology == "debate":
        registry.register_from_template("proposer")
        registry.register_from_template("critic")
        registry.register_from_template("judge")
        graph = InteractionGraph.create_debate_topology(["proposer", "critic", "judge"])
        
    elif topology == "panel":
        registry.register_from_template("mediator", "moderator")
        registry.register_from_template("researcher", "expert_1")
        registry.register_from_template("devil_advocate", "expert_2")
        graph = InteractionGraph.create_panel_discussion(
            "moderator", 
            ["expert_1", "expert_2"]
        )
        
    elif topology == "adversarial":
        registry.register_from_template("proposer", "defender")
        registry.register_from_template("devil_advocate", "attacker_1")
        registry.register_from_template("judge", "evaluator")
        graph = InteractionGraph.create_adversarial(
            "defender",
            ["attacker_1"],
            "evaluator"
        )

    elif topology == "court":
        from .agent_registry import create_court_agents
        create_court_agents(registry)
        graph = InteractionGraph.create_court_topology(
            "plaintiff_attorney",
            "defense_attorney",
            "court_judge",
        )

    else:
        # Fallback for simple tests
        registry.register_from_template("proposer")
        graph = InteractionGraph()
        graph.add_node("proposer")
    
    orchestrator.load_registry(registry)
    orchestrator.set_interaction_graph(graph)
    
    return orchestrator