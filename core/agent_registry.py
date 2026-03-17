#!/usr/bin/env python3
"""
Agent Registry & Configuration
==============================

Allows users to define arbitrary agents with custom roles, system prompts,
and behaviors. No hardcoded Proposer/Critic/Judge.

Usage:
    registry = AgentRegistry()
    registry.register("researcher", role_config={
        "system_prompt": "You are a researcher presenting findings.",
        "temperature": 0.7,
        "max_tokens": 200
    })
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json


class AgentBehavior(Enum):
    """Predefined behavior templates (can be customized)."""
    ASSERTIVE = "assertive"      # Makes strong claims
    ANALYTICAL = "analytical"    # Questions and analyzes
    NEUTRAL = "neutral"          # Balanced evaluation
    ADVERSARIAL = "adversarial"  # Challenges everything
    COOPERATIVE = "cooperative"  # Builds on others' ideas
    CUSTOM = "custom"            # Fully user-defined


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    agent_id: str
    display_name: str
    system_prompt: str
    behavior: AgentBehavior = AgentBehavior.CUSTOM
    temperature: float = 0.7
    max_tokens: int = 150

    # Injection settings (can be enabled/disabled per agent)
    injectable: bool = True
    injection_config: Optional[Dict[str, Any]] = None

    # Response processing
    response_prefix: str = ""  # e.g., "[Researcher]: "
    stop_sequences: List[str] = field(default_factory=list)

    # Tools available to this agent (list of tool names from ToolRegistry)
    tools: List[str] = field(default_factory=list)

    # Structured output format (e.g. for Judge rulings)
    response_format: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "system_prompt": self.system_prompt,
            "behavior": self.behavior.value,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "injectable": self.injectable,
            "injection_config": self.injection_config,
            "tools": self.tools,
            "response_format": self.response_format,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        data["behavior"] = AgentBehavior(data.get("behavior", "custom"))
        return cls(**data)


class AgentRegistry:
    """
    Central registry for all agents in the system.
    
    Supports:
    - Registering custom agents
    - Loading from config files
    - Predefined templates
    - Dynamic agent creation
    """
    
    # Predefined templates
    TEMPLATES = {
        "proposer": AgentConfig(
            agent_id="proposer",
            display_name="Proposer",
            system_prompt="You are presenting an argument. Be clear, provide reasoning, and defend your position.",
            behavior=AgentBehavior.ASSERTIVE,
            temperature=0.7
        ),
        "critic": AgentConfig(
            agent_id="critic", 
            display_name="Critic",
            system_prompt="You are critiquing arguments. Identify flaws, logical fallacies, and counterarguments.",
            behavior=AgentBehavior.ADVERSARIAL,
            temperature=0.6
        ),
        "judge": AgentConfig(
            agent_id="judge",
            display_name="Judge", 
            system_prompt="You are judging a debate. Evaluate arguments fairly and declare which side is more convincing.",
            behavior=AgentBehavior.NEUTRAL,
            temperature=0.5
        ),
        "researcher": AgentConfig(
            agent_id="researcher",
            display_name="Researcher",
            system_prompt="You are a researcher presenting factual findings. Cite evidence and maintain objectivity.",
            behavior=AgentBehavior.ANALYTICAL,
            temperature=0.5
        ),
        "devil_advocate": AgentConfig(
            agent_id="devil_advocate",
            display_name="Devil's Advocate",
            system_prompt="You challenge every position, regardless of your personal views. Find weaknesses in any argument.",
            behavior=AgentBehavior.ADVERSARIAL,
            temperature=0.8
        ),
        "mediator": AgentConfig(
            agent_id="mediator",
            display_name="Mediator",
            system_prompt="You find common ground between opposing views. Synthesize perspectives into balanced conclusions.",
            behavior=AgentBehavior.COOPERATIVE,
            temperature=0.6
        ),
        "fact_checker": AgentConfig(
            agent_id="fact_checker",
            display_name="Fact Checker",
            system_prompt="You verify claims for accuracy. Point out unsupported assertions and request evidence.",
            behavior=AgentBehavior.ANALYTICAL,
            temperature=0.4
        ),
        "strategist": AgentConfig(
            agent_id="strategist",
            display_name="Strategist",
            system_prompt="You analyze situations strategically. Consider long-term implications and propose action plans.",
            behavior=AgentBehavior.ANALYTICAL,
            temperature=0.6
        ),

        # ── Court Simulation Templates ──────────────────────────────

        "plaintiff_attorney": AgentConfig(
            agent_id="plaintiff_attorney",
            display_name="Plaintiff Attorney",
            system_prompt=(
                "You are Plaintiff's Counsel in a court proceeding. Your duties:\n"
                "1. Advocate zealously for the plaintiff within the bounds of the law.\n"
                "2. Present arguments supported by evidence and legal precedent.\n"
                "3. Before making a legal claim, SEARCH your available tools:\n"
                "   - Use the 'case_law_rag' tool to find relevant precedents and statutes.\n"
                "   - Use the 'evidence_rag' tool to cite specific evidence from the case file.\n"
                "   - If the internal library is insufficient, use the 'legal_fact_check' tool\n"
                "     to verify claims against external legal databases.\n"
                "4. Cite specific cases, statutes, or evidence when making arguments.\n"
                "5. Respond directly to Defense's rebuttals with counter-arguments.\n"
                "6. Structure every argument as: CLAIM → EVIDENCE → LEGAL BASIS → CONCLUSION.\n"
                "\n"
                "You have access to the following tools:\n"
                "  [RAG] case_law_rag — search the case law database\n"
                "  [RAG] evidence_rag — search case-specific evidence files\n"
                "  [FACT-CHECK] legal_fact_check — external legal web search (use when internal library is insufficient)\n"
            ),
            behavior=AgentBehavior.ASSERTIVE,
            temperature=0.7,
            max_tokens=400,
            tools=["case_law_rag", "evidence_rag", "legal_fact_check"],
            metadata={"role": "plaintiff", "side": "plaintiff"},
        ),

        "defense_attorney": AgentConfig(
            agent_id="defense_attorney",
            display_name="Defense Attorney",
            system_prompt=(
                "You are Defense Counsel in a court proceeding. Your duties:\n"
                "1. Defend your client vigorously within ethical and legal bounds.\n"
                "2. Respond DIRECTLY to the Plaintiff's most recent argument — address\n"
                "   their specific claims, evidence, and reasoning.\n"
                "3. Before making a legal claim, SEARCH your available tools:\n"
                "   - Use the 'case_law_rag' tool to find counter-precedents and statutes.\n"
                "   - Use the 'evidence_rag' tool to find exculpatory or mitigating evidence.\n"
                "   - If the internal library is insufficient, use the 'legal_fact_check' tool\n"
                "     to verify claims against external legal databases.\n"
                "4. Cite specific cases, statutes, or evidence to rebut Plaintiff's arguments.\n"
                "5. Challenge the Plaintiff's evidence, legal reasoning, and conclusions.\n"
                "6. Structure every rebuttal as: PLAINTIFF'S CLAIM → YOUR CHALLENGE → COUNTER-EVIDENCE → CONCLUSION.\n"
                "\n"
                "You have access to the following tools:\n"
                "  [RAG] case_law_rag — search the case law database\n"
                "  [RAG] evidence_rag — search case-specific evidence files\n"
                "  [FACT-CHECK] legal_fact_check — external legal web search (use when internal library is insufficient)\n"
            ),
            behavior=AgentBehavior.ADVERSARIAL,
            temperature=0.6,
            max_tokens=400,
            tools=["case_law_rag", "evidence_rag", "legal_fact_check"],
            metadata={"role": "defense", "side": "defense"},
        ),

        "court_judge": AgentConfig(
            agent_id="court_judge",
            display_name="Presiding Judge",
            system_prompt=(
                "You are the Presiding Judge in this proceeding. Your duties:\n"
                "1. Listen carefully to ALL arguments from both Plaintiff and Defense.\n"
                "2. You may ask clarifying questions to either party when arguments are\n"
                "   ambiguous, unsupported, or require elaboration. Prefix questions with\n"
                "   'CLARIFICATION REQUEST:' so the Orchestrator routes them properly.\n"
                "3. Evaluate arguments ONLY on the evidence and legal reasoning presented.\n"
                "   Do NOT introduce outside knowledge or personal opinion.\n"
                "4. When issuing your final ruling, you MUST use this exact format:\n"
                "\n"
                "RULING:\n"
                "winning_party: [Plaintiff or Defense]\n"
                "key_evidence_cited:\n"
                "  - [fact 1 relied upon]\n"
                "  - [fact 2 relied upon]\n"
                "  - [fact 3 relied upon]\n"
                "legal_reasoning: [Your justification for the ruling, citing applicable\n"
                "  law, standards of proof, and how the evidence meets or fails to meet\n"
                "  the legal standard.]\n"
                "awarded_damages: [Dollar amount as integer, or 0 if not applicable]\n"
                "\n"
                "5. Before issuing the ruling, summarise each side's strongest and\n"
                "   weakest arguments.\n"
                "6. Apply the correct standard of proof (preponderance of evidence for\n"
                "   civil cases; beyond reasonable doubt for criminal cases).\n"
            ),
            behavior=AgentBehavior.NEUTRAL,
            temperature=0.4,
            max_tokens=600,
            tools=[],
            response_format={
                "type": "structured",
                "fields": {
                    "winning_party": {"type": "string", "enum": ["Plaintiff", "Defense"]},
                    "key_evidence_cited": {"type": "list", "items": "string"},
                    "legal_reasoning": {"type": "string"},
                    "awarded_damages": {"type": "integer"},
                },
            },
            metadata={"role": "judge", "side": "neutral"},
        ),

        "court_orchestrator": AgentConfig(
            agent_id="court_orchestrator",
            display_name="Court Orchestrator",
            system_prompt=(
                "You manage the flow of a court proceeding. Your responsibilities:\n"
                "1. Pass the full transcript (State) to the Plaintiff to generate their argument.\n"
                "2. Pass the updated State (including the Plaintiff's new argument) to the Defense.\n"
                "3. After a set number of argument rounds, pass the complete State to the Judge.\n"
                "4. If the Judge issues a CLARIFICATION REQUEST, route the question to the\n"
                "   addressed party and feed their answer back into the State.\n"
                "5. Enforce turn order: Plaintiff → Defense → (repeat) → Judge.\n"
                "6. Track and display the round number and current phase.\n"
            ),
            behavior=AgentBehavior.NEUTRAL,
            temperature=0.3,
            max_tokens=100,
            injectable=False,
            metadata={"role": "orchestrator", "side": "neutral"},
        ),
    }
    
    def __init__(self):
        self._agents: Dict[str, AgentConfig] = {}
        self._creation_hooks: List[Callable] = []
    
    def register(
        self, 
        agent_id: str,
        display_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        **kwargs
    ) -> AgentConfig:
        """
        Register a new agent.
        
        Args:
            agent_id: Unique identifier for the agent
            display_name: Human-readable name
            system_prompt: The agent's system prompt
            template: Use a predefined template as base
            **kwargs: Additional AgentConfig parameters
            
        Returns:
            The created AgentConfig
        """
        if template and template in self.TEMPLATES:
            # Start from template
            base = self.TEMPLATES[template]
            config = AgentConfig(
                agent_id=agent_id,
                display_name=display_name or base.display_name,
                system_prompt=system_prompt or base.system_prompt,
                behavior=base.behavior,
                temperature=kwargs.get("temperature", base.temperature),
                max_tokens=kwargs.get("max_tokens", base.max_tokens),
                injectable=kwargs.get("injectable", base.injectable),
                injection_config=kwargs.get("injection_config"),
                metadata=kwargs.get("metadata", {})
            )
        else:
            # Create from scratch
            config = AgentConfig(
                agent_id=agent_id,
                display_name=display_name or agent_id.replace("_", " ").title(),
                system_prompt=system_prompt or "You are a helpful assistant.",
                **kwargs
            )
        
        self._agents[agent_id] = config
        
        # Call creation hooks
        for hook in self._creation_hooks:
            hook(config)
            
        return config
    
    def register_from_template(self, template_name: str, agent_id: Optional[str] = None) -> AgentConfig:
        """Quick registration from a template."""
        if template_name not in self.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(self.TEMPLATES.keys())}")
        
        aid = agent_id or template_name
        return self.register(aid, template=template_name)
    
    def get(self, agent_id: str) -> Optional[AgentConfig]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_all(self) -> Dict[str, AgentConfig]:
        """Get all registered agents."""
        return self._agents.copy()
    
    def list_agents(self) -> List[str]:
        """List all agent IDs."""
        return list(self._agents.keys())
    
    def remove(self, agent_id: str) -> bool:
        """Remove an agent."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False
    
    def clear(self):
        """Remove all agents."""
        self._agents.clear()
    
    def add_creation_hook(self, hook: Callable[[AgentConfig], None]):
        """Add a hook called when agents are created."""
        self._creation_hooks.append(hook)
    
    # === Serialization ===
    
    def save(self, filepath: str):
        """Save registry to JSON file."""
        data = {aid: cfg.to_dict() for aid, cfg in self._agents.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load registry from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        for aid, cfg_dict in data.items():
            self._agents[aid] = AgentConfig.from_dict(cfg_dict)
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List available templates."""
        return list(cls.TEMPLATES.keys())
    
    def __len__(self) -> int:
        return len(self._agents)
    
    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents
    
    def __repr__(self) -> str:
        return f"AgentRegistry({list(self._agents.keys())})"


# === Convenience Functions ===

def create_debate_agents(registry: AgentRegistry) -> List[str]:
    """Quick setup for standard debate (Proposer, Critic, Judge)."""
    registry.register_from_template("proposer")
    registry.register_from_template("critic")
    registry.register_from_template("judge")
    return ["proposer", "critic", "judge"]


def create_research_team(registry: AgentRegistry) -> List[str]:
    """Quick setup for research team."""
    registry.register_from_template("researcher", "lead_researcher")
    registry.register_from_template("fact_checker")
    registry.register_from_template("critic", "peer_reviewer")
    registry.register_from_template("mediator", "editor")
    return ["lead_researcher", "fact_checker", "peer_reviewer", "editor"]


def create_adversarial_team(registry: AgentRegistry) -> List[str]:
    """Quick setup for red team / adversarial testing."""
    registry.register_from_template("proposer", "defender")
    registry.register_from_template("devil_advocate", "attacker")
    registry.register_from_template("judge", "evaluator")
    return ["defender", "attacker", "evaluator"]


def create_court_agents(registry: AgentRegistry) -> List[str]:
    """
    Quick setup for a court simulation with:
      - plaintiff_attorney  (with RAG + fact-check tools)
      - defense_attorney    (with RAG + fact-check tools)
      - court_judge         (structured ruling output)
    """
    registry.register_from_template("plaintiff_attorney")
    registry.register_from_template("defense_attorney")
    registry.register_from_template("court_judge")
    return ["plaintiff_attorney", "defense_attorney", "court_judge"]