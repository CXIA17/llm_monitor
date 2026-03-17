#!/usr/bin/env python3
"""
Steered Agent
=============

An agent that can generate responses with probe-based activation steering.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from .orchestrator import ProbeConfig, InjectionConfig
from .model_compatibility import ModelCompatibility


class SteeredAgent:
    """
    An agent that generates responses with optional probe-based steering.
    
    Features:
    - Activation monitoring via probes
    - Injection/steering of activations
    - Per-token score tracking
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        probe: Optional[ProbeConfig],
        config: Any,  # AgentConfig from agent_registry
        device: str,
        model_compat: Optional[ModelCompatibility] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe
        self.config = config
        self.device = device
        self.model_compat = model_compat or ModelCompatibility(model, tokenizer)
        
        # Set target layer from probe or default to middle
        if probe and hasattr(probe, 'layer_idx') and probe.layer_idx is not None:
            self.target_layer = probe.layer_idx
        elif isinstance(probe, dict) and probe.get('layer_idx'):
            self.target_layer = probe['layer_idx']
        else:
            self.target_layer = self.model_compat.num_layers // 2
        
        self._current_activation: Optional[np.ndarray] = None
        self._handles: List = []
        self._injection_stats = {"calls": 0, "total_delta": 0.0, "gated_skips": 0}
        self._layer_std_cache = {}
        self._collect_activations: bool = False
        self._activation_history: List[np.ndarray] = []
    
    def set_probe(self, probe) -> None:
        """Set or update the probe for this agent."""
        print(f"[SetProbe] Setting probe: type={type(probe)}, is_none={probe is None}")
        self.probe = probe
        
        # Update target layer based on new probe
        if probe and hasattr(probe, 'layer_idx') and probe.layer_idx is not None:
            self.target_layer = probe.layer_idx
            print(f"[SetProbe] Updated target_layer from probe.layer_idx: {self.target_layer}")
        elif isinstance(probe, dict) and probe.get('layer_idx'):
            self.target_layer = probe['layer_idx']
            print(f"[SetProbe] Updated target_layer from dict: {self.target_layer}")
        else:
            print(f"[SetProbe] Keeping existing target_layer: {self.target_layer}")
    
    def _get_layer_std(self, layer_idx: int) -> float:
        """Estimate activation standard deviation for adaptive scaling."""
        if layer_idx in self._layer_std_cache:
            return self._layer_std_cache[layer_idx]
        
        dummy_input = self.tokenizer("The quick brown fox", return_tensors="pt").to(self.device)
        stats = []
        
        def stat_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            stats.append(hidden.detach().float().std().item())
        
        layer_module = self.model_compat.get_layer(layer_idx)
        if layer_module is None:
            return 1.0
            
        handle = layer_module.register_forward_hook(stat_hook)
        
        with torch.no_grad():
            self.model(**dummy_input)
        
        handle.remove()
        
        std = stats[0] if stats else 1.0
        self._layer_std_cache[layer_idx] = std
        return std
    
    def _get_probe_direction(self) -> Optional[np.ndarray]:
        """Get probe direction from various formats."""
        if self.probe is None:
            print(f"[Probe] self.probe is None")
            return None
        
        print(f"[Probe] probe type: {type(self.probe)}")
        
        direction = None
        
        # Handle standardized format from dashboard (dict with 'direction' key)
        if isinstance(self.probe, dict) and 'direction' in self.probe:
            d = self.probe['direction']
            if d is not None:
                direction = np.array(d).flatten()
                print(f"[Probe] Found standardized direction, shape={direction.shape}")
        
        # Direct sklearn model
        elif hasattr(self.probe, 'coef_'):
            direction = np.array(self.probe.coef_).flatten()
            print(f"[Probe] Found coef_, shape={direction.shape}")
        
        # Object with direction attribute
        elif hasattr(self.probe, 'direction') and self.probe.direction is not None:
            direction = np.array(self.probe.direction).flatten()
            print(f"[Probe] Found direction attribute, shape={direction.shape}")
        
        # Handle nested dict (legacy support)
        elif isinstance(self.probe, dict):
            print(f"[Probe] dict keys: {list(self.probe.keys())}")
            
            def try_extract(obj, depth=0):
                if depth > 5:
                    return None
                if hasattr(obj, 'coef_'):
                    return np.array(obj.coef_).flatten()
                if hasattr(obj, 'direction') and obj.direction is not None:
                    return np.array(obj.direction).flatten()
                if isinstance(obj, dict):
                    for key in ['direction', 'coef', 'weights', 'classifier', 'model']:
                        if key in obj and obj[key] is not None:
                            result = try_extract(obj[key], depth + 1)
                            if result is not None:
                                return result
                    if len(obj) == 1:
                        return try_extract(list(obj.values())[0], depth + 1)
                if isinstance(obj, np.ndarray) and obj.size > 10:
                    return obj.flatten()
                return None
            
            direction = try_extract(self.probe)
            if direction is not None:
                print(f"[Probe] Extracted from nested structure, shape={direction.shape}")
        
        # Array-like
        elif isinstance(self.probe, np.ndarray) and self.probe.size > 10:
            direction = self.probe.flatten()
            print(f"[Probe] probe is array, shape={direction.shape}")
        
        if direction is None:
            print(f"[Probe] Could not extract direction")
            return None
        
        # CRITICAL: Validate dimension matches model's hidden size
        expected_dim = self._get_model_hidden_dim()
        if expected_dim and direction.shape[0] != expected_dim:
            print(f"[Probe] ⚠️ DIMENSION MISMATCH: probe has {direction.shape[0]} dims, model expects {expected_dim}")
            print(f"[Probe] The probe was trained on a different model architecture.")
            print(f"[Probe] Shadow/steering will be DISABLED for this session.")
            return None
        
        return direction
    
    def _get_model_hidden_dim(self) -> Optional[int]:
        """Get the model's hidden dimension."""
        try:
            # Try common config attributes
            if hasattr(self.model, 'config'):
                config = self.model.config
                for attr in ['hidden_size', 'd_model', 'n_embd', 'dim']:
                    if hasattr(config, attr):
                        return getattr(config, attr)
            
            # Try to get from model_compat
            if self.model_compat and hasattr(self.model_compat, 'hidden_size'):
                return self.model_compat.hidden_size
            
            return None
        except Exception:
            return None
    
    def _create_injection_hook(
        self,
        strength: float,
        direction: str = "add",
        layer_idx: int = None,
        gate_threshold: float = 0.0,
        gate_direction: str = "below",
    ):
        """Create a steering injection hook with dynamic scaling and gating.

        Dynamic scaling: injection magnitude is proportional to the current
        residual-stream norm rather than a one-time dummy-input estimate.
        This prevents over-injection when activations are small and
        under-injection when they are large.

        Gated injection: the hook reads the probe score from the monitor
        layer (``self._current_activation``) and only injects when the
        model is drifting from the desired behavior.  ``gate_direction``
        controls the polarity:
          - "below": inject only when score < threshold (push score up)
          - "above": inject only when score > threshold (push score down)
        """
        probe_direction = self._get_probe_direction()
        if probe_direction is None:
            return None

        vector_tensor = torch.tensor(probe_direction, dtype=torch.float32).to(self.device)
        vector_tensor = F.normalize(vector_tensor, p=2, dim=0)

        sign = -1.0 if direction == "subtract" else 1.0

        def hook(module, input, output):
            act = output[0] if isinstance(output, tuple) else output

            # Dimension compatibility check
            if act.shape[-1] != vector_tensor.shape[0]:
                return output

            # --- Gated injection: read before write ---
            # Use the latest monitor-layer activation to decide whether
            # steering is needed for this token.
            if self._current_activation is not None:
                probe_dir = self._get_probe_direction()
                if probe_dir is not None:
                    norm_dir = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
                    current_score = float(np.dot(
                        self._current_activation.flatten(), norm_dir.flatten()
                    ))
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
            # Scale injection as a fraction of the residual-stream norm.
            # This keeps the perturbation proportional to the activation
            # magnitude without the sqrt(d) dampening that made the
            # injection too weak on large models.
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
    
    def _create_monitor_hook(self):
        """Create a hook to capture activations for scoring and SAE collection."""
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hasattr(hidden, "shape") and len(hidden.shape) >= 3:
                self._current_activation = hidden[0, -1, :].detach().float().cpu().numpy()
                # Accumulate for SAE: store last-token activation at each generation step
                if self._collect_activations:
                    self._activation_history.append(self._current_activation.copy())
        return hook
    
    def _register_hooks(self, injection_config: Optional[InjectionConfig] = None):
        """Register forward hooks for monitoring and injection.

        Injection is applied across a range of layers (middle third of the
        network) for stronger behavioral steering, following Representation
        Engineering best practices.  Monitoring remains on the single target
        layer so probe scores reflect post-injection activations.
        """
        # Clear old hooks
        for h in self._handles:
            h.remove()
        self._handles = []

        # Injection hooks across multiple layers
        if injection_config is not None:
            strength = injection_config.strength if hasattr(injection_config, 'strength') else injection_config.get('strength', 0.0)
            direction = injection_config.direction if hasattr(injection_config, 'direction') else injection_config.get('direction', 'add')

            # Gating parameters from InjectionConfig
            gate_bias = injection_config.gate_bias if hasattr(injection_config, 'gate_bias') else injection_config.get('gate_bias', 0.0)
            # Determine gate direction from injection direction:
            # "add" means we want to push score up   -> inject when score is below threshold
            # "subtract" means we want to push down  -> inject when score is above threshold
            gate_dir = "below" if direction == "add" else "above"

            num_layers = self.model_compat.num_layers
            # Inject from early-middle through second-to-last layer
            # Must cover late layers to affect output token distribution
            inject_start = num_layers // 4
            inject_end = num_layers - 1  # leave only the final layer clean
            n_inject_layers = inject_end - inject_start
            # Scale per-layer strength: divide by n^0.65 to balance
            # total effect across layers without overwhelming the model
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
                    layer_idx=layer_idx,
                    gate_threshold=gate_bias,
                    gate_direction=gate_dir,
                )
                if hook_fn:
                    self._handles.append(layer_mod.register_forward_hook(hook_fn))
                    injected_count += 1

            if injected_count > 0:
                print(f"[Injection] Registered on {injected_count} layers ({inject_start}-{inject_end-1}), gating={gate_dir}@{gate_bias}")

        # Monitor hook on target layer (always)
        # Note: if injection is active on this layer, the score reflects
        # the injected state. To get a clean reading, we exclude the
        # target layer from injection and monitor there instead.
        monitor_module = self.model_compat.get_layer(self.target_layer)
        if monitor_module is None:
            print(f"Warning: Could not find monitor layer {self.target_layer}")
            return
        self._handles.append(monitor_module.register_forward_hook(self._create_monitor_hook()))
    
    def get_current_score(self) -> float:
        """Get probe score from current activation."""
        if self._current_activation is None:
            return 0.0
        
        probe_direction = self._get_probe_direction()
        if probe_direction is None:
            return 0.0
        
        # Project activation onto probe direction
        norm_dir = probe_direction / (np.linalg.norm(probe_direction) + 1e-8)
        return float(np.dot(self._current_activation.flatten(), norm_dir.flatten()))
    
    def generate_response(
        self,
        prompt: str,
        injection_config: Optional[InjectionConfig] = None,
        context: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a response with optional steering.

        Args:
            prompt: The input prompt
            injection_config: Optional steering configuration
            context: Additional context to prepend

        Returns:
            Dictionary with response_text, scores, mean_score, token_strings,
            and optionally live_activations (n_tokens, d_model) numpy array.
        """
        # Enable activation collection for SAE
        self._collect_activations = True
        self._activation_history = []

        self._register_hooks(injection_config)

        # Build full prompt using chat template if available, otherwise raw
        system = self.config.system_prompt if hasattr(self.config, 'system_prompt') else "You are a helpful assistant."

        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{context}{prompt}"},
            ]
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            full_prompt = f"{system}\n\n{context}{prompt}\n\nYour response:"

        # Tokenize with safety limits
        max_ctx = getattr(self.model_compat, 'max_position_embeddings', 2048)
        max_new = self.config.max_tokens if hasattr(self.config, 'max_tokens') else 150
        safe_input_len = max(50, max_ctx - max_new - 10)

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=safe_input_len
        )
        input_ids = inputs["input_ids"].to(self.device)
        input_len = input_ids.shape[1]

        scores = []
        token_strings = []
        generated_ids = input_ids.clone()
        in_think = False  # Track whether we're inside <think> tags

        # Detect think token IDs for this tokenizer
        think_start_id = None
        think_end_id = None
        try:
            think_start_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
            think_end_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
            if len(think_start_ids) == 1:
                think_start_id = think_start_ids[0]
            if len(think_end_ids) == 1:
                think_end_id = think_end_ids[0]
        except Exception:
            pass

        temperature = self.config.temperature if hasattr(self.config, 'temperature') else 0.7

        # Token-by-token generation with scoring
        for _ in range(max_new):
            with torch.no_grad():
                attention_mask = torch.ones_like(generated_ids)
                outputs = self.model(generated_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()

            # Track think state
            if token_id == think_start_id:
                in_think = True
            elif token_id == think_end_id:
                in_think = False
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                continue  # Don't score the </think> token itself

            # Only score non-thinking tokens
            if not in_think:
                score = self.get_current_score()
                scores.append(score)
                token_strings.append(self.tokenizer.decode(token_id))

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop conditions
            if token_id == self.tokenizer.eos_token_id:
                break
            if generated_ids.shape[1] >= max_ctx:
                break

        # Clean up hooks
        for h in self._handles:
            h.remove()
        self._handles = []

        # Decode only the generated portion
        generated_token_ids = generated_ids[0][input_len:]
        full_output = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        # Strip <think>...</think> content from the response text
        response_text = full_output
        response_text = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)
        response_text = response_text.strip()

        # If generation hit the token limit (no EOS), trim to last complete sentence
        hit_eos = (generated_ids[0][-1].item() == self.tokenizer.eos_token_id)
        if not hit_eos and response_text:
            # Find last sentence-ending punctuation
            last_period = max(
                response_text.rfind('. '),
                response_text.rfind('.\n'),
                response_text.rfind('."'),
            )
            last_end = max(
                last_period,
                response_text.rfind('? '),
                response_text.rfind('?\n'),
                response_text.rfind('! '),
                response_text.rfind('!\n'),
            )
            if last_end > len(response_text) // 3:
                response_text = response_text[:last_end + 1].strip()

        # Build live activation matrix from generation-time captures
        # Shape: (n_generated_tokens, d_model) — includes injection effects
        live_activations = None
        if self._activation_history:
            live_activations = np.stack(self._activation_history, axis=0)
        self._collect_activations = False
        self._activation_history = []

        return {
            "response_text": response_text,
            "scores": scores,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "token_strings": token_strings,
            "shadow_log": {},
            "injection_stats": self._injection_stats.copy(),
            "live_activations": live_activations,
        }
    
