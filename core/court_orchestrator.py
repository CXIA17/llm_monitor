#!/usr/bin/env python3
"""
Court Orchestrator
==================

A specialised orchestrator for courtroom simulations that enforces the
proper adversarial loop:

    ┌─────────────────────────────────────────────────┐
    │  For each argument round:                       │
    │    1. Plaintiff receives the State (transcript)  │
    │       → generates argument → appends to State    │
    │    2. Defense receives the updated State          │
    │       → reads Plaintiff's latest argument         │
    │       → generates rebuttal → appends to State    │
    │  After all rounds:                               │
    │    3. Judge receives full State                   │
    │       → may ask clarifying questions              │
    │       → issues structured ruling                  │
    └─────────────────────────────────────────────────┘

The orchestrator also invokes agent tools (RAG, fact-checking) on behalf
of attorneys when their prompts contain search directives.

Usage:
    from core.court_orchestrator import CourtOrchestrator

    court = CourtOrchestrator(model, tokenizer, device)
    court.load_tools(tool_registry)
    result = court.run_trial(
        case_description="Plaintiff alleges ...",
        num_rounds=3,
    )
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

import numpy as np

from .agent_registry import AgentRegistry, AgentConfig, create_court_agents
from .interaction_graph import ConversationState
from .tools import ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class JudgeRuling:
    """Structured ruling issued by the Judge."""
    winning_party: str              # "Plaintiff" or "Defense"
    key_evidence_cited: List[str]   # facts relied upon
    legal_reasoning: str            # justification
    awarded_damages: int            # dollar amount (0 if N/A)
    raw_text: str = ""              # the full unstructured response

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winning_party": self.winning_party,
            "key_evidence_cited": self.key_evidence_cited,
            "legal_reasoning": self.legal_reasoning,
            "awarded_damages": self.awarded_damages,
            "raw_text": self.raw_text,
        }


@dataclass
class CourtTranscript:
    """Full trial transcript with metadata."""
    case_description: str
    num_rounds: int
    turns: List[Dict[str, Any]] = field(default_factory=list)
    ruling: Optional[JudgeRuling] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    clarifications: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def add_turn(self, agent_id: str, role: str, content: str, round_num: int, metadata: Optional[Dict] = None):
        self.turns.append({
            "round": round_num,
            "agent": agent_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
        })

    def format_state(self, for_agent: Optional[str] = None) -> str:
        """
        Format the entire transcript as a string (the 'State') that is
        passed to the next agent.  Every agent sees the *same* transcript.
        """
        lines = [f"CASE: {self.case_description}\n"]
        for t in self.turns:
            label = t["role"].upper()
            lines.append(f"[Round {t['round']} — {label}]:\n{t['content']}\n")
        return "\n".join(lines)

    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_description": self.case_description,
            "num_rounds": self.num_rounds,
            "turns": self.turns,
            "ruling": self.ruling.to_dict() if self.ruling else None,
            "tool_calls": self.tool_calls,
            "clarifications": self.clarifications,
            "duration_seconds": self.duration(),
        }


# ---------------------------------------------------------------------------
# Tool-calling helpers
# ---------------------------------------------------------------------------

# Regex to detect when an agent wants to invoke a tool in its output.
# Agents are prompted to use the pattern: SEARCH[tool_name]: query
_TOOL_CALL_RE = re.compile(
    r"SEARCH\[(\w+)\]\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE
)


def _extract_tool_calls(text: str) -> List[Dict[str, str]]:
    """Parse tool-call directives embedded in agent output."""
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        calls.append({"tool": match.group(1).strip(), "query": match.group(2).strip()})
    return calls


def _run_tool_calls(
    calls: List[Dict[str, str]],
    tool_registry: Optional[ToolRegistry],
    allowed_tools: List[str],
) -> str:
    """Execute tool calls and return formatted results to inject back."""
    if not tool_registry or not calls:
        return ""

    parts = []
    for call in calls:
        tool_name = call["tool"]
        if tool_name not in allowed_tools:
            parts.append(f"[TOOL ERROR]: '{tool_name}' is not available to you.")
            continue
        result = tool_registry.invoke(tool_name, call["query"])
        parts.append(result.format_for_prompt(max_results=3))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Judge ruling parser
# ---------------------------------------------------------------------------

def _parse_ruling(text: str) -> JudgeRuling:
    """Best-effort parse of the Judge's structured ruling from free text."""
    ruling = JudgeRuling(
        winning_party="Unknown",
        key_evidence_cited=[],
        legal_reasoning="",
        awarded_damages=0,
        raw_text=text,
    )

    # winning_party
    m = re.search(r"winning_party\s*:\s*(Plaintiff|Defense)", text, re.IGNORECASE)
    if m:
        ruling.winning_party = m.group(1).capitalize()
        # Normalise to exact enum
        if ruling.winning_party.lower().startswith("p"):
            ruling.winning_party = "Plaintiff"
        else:
            ruling.winning_party = "Defense"

    # key_evidence_cited (bulleted list)
    evidence_block = re.search(
        r"key_evidence_cited\s*:(.*?)(?=legal_reasoning|awarded_damages|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if evidence_block:
        items = re.findall(r"[-•*]\s*(.+)", evidence_block.group(1))
        ruling.key_evidence_cited = [i.strip() for i in items if i.strip()]

    # legal_reasoning
    reasoning_block = re.search(
        r"legal_reasoning\s*:\s*(.*?)(?=awarded_damages|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if reasoning_block:
        ruling.legal_reasoning = reasoning_block.group(1).strip()

    # awarded_damages
    dmg = re.search(r"awarded_damages\s*:\s*\$?([\d,]+)", text, re.IGNORECASE)
    if dmg:
        ruling.awarded_damages = int(dmg.group(1).replace(",", ""))

    return ruling


# ---------------------------------------------------------------------------
# Court Orchestrator
# ---------------------------------------------------------------------------

class CourtOrchestrator:
    """
    Manages the adversarial courtroom loop.

    The State (transcript) is the single source of truth.  Every agent
    receives the *full* State when it is their turn.  The Orchestrator
    never summarises or filters — it passes the raw transcript so that
    each agent can read exactly what every other agent has said.

    Loop:
        for round in 1..num_rounds:
            state → Plaintiff → updated state → Defense → updated state
        full state → Judge → ruling

    If the Judge emits a CLARIFICATION REQUEST, the Orchestrator re-routes
    the question to the addressed party, appends their answer, and feeds
    the updated state back to the Judge.
    """

    def __init__(
        self,
        generate_fn: Callable[..., str],
        agent_configs: Optional[Dict[str, AgentConfig]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_clarifications: int = 2,
        verbose: bool = True,
    ):
        """
        Args:
            generate_fn:  A callable with signature
                            (system_prompt: str, user_prompt: str,
                             temperature: float, max_tokens: int) -> str
                          This abstracts the underlying model so the orchestrator
                          works with any backend (local HF, API, etc.).
            agent_configs: dict of agent_id -> AgentConfig.  If None, defaults
                           are loaded from the registry templates.
            tool_registry: optional ToolRegistry for RAG / fact-check tools.
            max_clarifications: max Judge clarification rounds before forcing
                                a ruling.
            verbose: print progress to stdout.
        """
        self.generate_fn = generate_fn
        self.tool_registry = tool_registry
        self.max_clarifications = max_clarifications
        self.verbose = verbose

        # Load default court agents if none provided
        if agent_configs is None:
            reg = AgentRegistry()
            create_court_agents(reg)
            self.agents = reg.get_all()
        else:
            self.agents = dict(agent_configs)

        # Callbacks
        self.on_turn_complete: List[Callable] = []
        self.on_round_complete: List[Callable] = []
        self.on_tool_call: List[Callable] = []

    # ---- public API -------------------------------------------------------

    def load_tools(self, tool_registry: ToolRegistry):
        """Attach a tool registry (RAG + fact-check tools)."""
        self.tool_registry = tool_registry

    def run_trial(
        self,
        case_description: str,
        num_rounds: int = 3,
        plaintiff_id: str = "plaintiff_attorney",
        defense_id: str = "defense_attorney",
        judge_id: str = "court_judge",
    ) -> CourtTranscript:
        """
        Execute a full trial.

        Args:
            case_description: the case facts / complaint.
            num_rounds: how many Plaintiff→Defense argument rounds.
            plaintiff_id / defense_id / judge_id: agent IDs in self.agents.

        Returns:
            CourtTranscript with all turns, tool calls, and parsed ruling.
        """
        transcript = CourtTranscript(
            case_description=case_description,
            num_rounds=num_rounds,
            start_time=time.time(),
        )

        plaintiff_cfg = self.agents[plaintiff_id]
        defense_cfg = self.agents[defense_id]
        judge_cfg = self.agents[judge_id]

        # ── Argument Rounds ──────────────────────────────────────────
        for round_num in range(1, num_rounds + 1):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"  ROUND {round_num} / {num_rounds}")
                print(f"{'='*60}")

            # 1. Plaintiff's turn ─────────────────────────────────────
            state_for_plaintiff = transcript.format_state(for_agent=plaintiff_id)
            if round_num == 1:
                user_prompt = (
                    f"The case before the court:\n{case_description}\n\n"
                    "Present your opening argument for the Plaintiff."
                )
            else:
                user_prompt = (
                    "The transcript so far is shown above. "
                    "Present your next argument or rebuttal for the Plaintiff."
                )

            plaintiff_response = self._agent_turn(
                agent_cfg=plaintiff_cfg,
                state=state_for_plaintiff,
                user_prompt=user_prompt,
                transcript=transcript,
                round_num=round_num,
                role_label="Plaintiff",
            )
            transcript.add_turn(plaintiff_id, "Plaintiff", plaintiff_response, round_num)

            # 2. Defense's turn ───────────────────────────────────────
            state_for_defense = transcript.format_state(for_agent=defense_id)
            user_prompt = (
                "The transcript so far is shown above. The Plaintiff has just "
                "presented their argument. Respond directly to the Plaintiff's "
                "latest argument with your rebuttal for the Defense."
            )

            defense_response = self._agent_turn(
                agent_cfg=defense_cfg,
                state=state_for_defense,
                user_prompt=user_prompt,
                transcript=transcript,
                round_num=round_num,
                role_label="Defense",
            )
            transcript.add_turn(defense_id, "Defense", defense_response, round_num)

            # Round callbacks
            for cb in self.on_round_complete:
                cb(round_num, transcript)

        # ── Judge's Ruling ───────────────────────────────────────────
        if self.verbose:
            print(f"\n{'='*60}")
            print("  JUDGE DELIBERATION")
            print(f"{'='*60}")

        ruling = self._judge_deliberation(
            judge_cfg=judge_cfg,
            transcript=transcript,
            judge_id=judge_id,
            plaintiff_id=plaintiff_id,
            defense_id=defense_id,
        )
        transcript.ruling = ruling
        transcript.end_time = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  RULING: {ruling.winning_party}")
            print(f"  DAMAGES: ${ruling.awarded_damages:,}")
            print(f"  DURATION: {transcript.duration():.1f}s")
            print(f"{'='*60}")

        return transcript

    # ---- internal ---------------------------------------------------------

    def _agent_turn(
        self,
        agent_cfg: AgentConfig,
        state: str,
        user_prompt: str,
        transcript: CourtTranscript,
        round_num: int,
        role_label: str,
    ) -> str:
        """
        Run a single agent turn.  If the agent emits tool-call directives,
        execute them and give the results back for a second generation pass.
        """
        full_user = f"{state}\n\n{user_prompt}"

        response = self.generate_fn(
            system_prompt=agent_cfg.system_prompt,
            user_prompt=full_user,
            temperature=agent_cfg.temperature,
            max_tokens=agent_cfg.max_tokens,
        )

        if self.verbose:
            print(f"\n  [{role_label} — {agent_cfg.display_name}]")
            print(f"  {response[:200]}{'...' if len(response) > 200 else ''}")

        # ── Tool calls ───────────────────────────────────────────────
        tool_calls = _extract_tool_calls(response)
        if tool_calls and self.tool_registry:
            tool_output = _run_tool_calls(
                tool_calls, self.tool_registry, agent_cfg.tools,
            )
            if tool_output:
                # Log the tool calls
                for tc in tool_calls:
                    transcript.tool_calls.append({
                        "round": round_num,
                        "agent": agent_cfg.agent_id,
                        "tool": tc["tool"],
                        "query": tc["query"],
                    })
                for cb in self.on_tool_call:
                    cb(agent_cfg.agent_id, tool_calls)

                # Second pass: agent gets tool results injected
                augmented_prompt = (
                    f"{full_user}\n\n"
                    f"Your previous draft referenced external sources. "
                    f"Here are the search results:\n\n{tool_output}\n\n"
                    f"Now rewrite your argument incorporating these results."
                )
                response = self.generate_fn(
                    system_prompt=agent_cfg.system_prompt,
                    user_prompt=augmented_prompt,
                    temperature=agent_cfg.temperature,
                    max_tokens=agent_cfg.max_tokens,
                )
                if self.verbose:
                    print(f"  [Revised after tool use]")
                    print(f"  {response[:200]}{'...' if len(response) > 200 else ''}")

        # Turn callbacks
        for cb in self.on_turn_complete:
            cb(agent_cfg.agent_id, role_label, response, round_num)

        return response

    def _judge_deliberation(
        self,
        judge_cfg: AgentConfig,
        transcript: CourtTranscript,
        judge_id: str,
        plaintiff_id: str,
        defense_id: str,
    ) -> JudgeRuling:
        """
        Judge receives the full state, may ask clarifying questions,
        then issues a structured ruling.
        """
        state = transcript.format_state(for_agent=judge_id)
        user_prompt = (
            "You have heard all arguments from both parties. The full "
            "transcript is shown above.\n\n"
            "If you need clarification on any point, begin your response "
            "with 'CLARIFICATION REQUEST:' followed by your question and "
            "the party you are addressing (Plaintiff or Defense).\n\n"
            "Otherwise, issue your final ruling using the required format:\n"
            "RULING:\n"
            "winning_party: ...\n"
            "key_evidence_cited:\n"
            "  - ...\n"
            "legal_reasoning: ...\n"
            "awarded_damages: ...\n"
        )

        clarification_count = 0

        while clarification_count <= self.max_clarifications:
            full_user = f"{state}\n\n{user_prompt}"
            judge_response = self.generate_fn(
                system_prompt=judge_cfg.system_prompt,
                user_prompt=full_user,
                temperature=judge_cfg.temperature,
                max_tokens=judge_cfg.max_tokens,
            )

            if self.verbose:
                print(f"\n  [Judge — {judge_cfg.display_name}]")
                print(f"  {judge_response[:300]}{'...' if len(judge_response) > 300 else ''}")

            # Check for clarification request
            if "CLARIFICATION REQUEST:" in judge_response.upper() and clarification_count < self.max_clarifications:
                clarification_count += 1
                # Determine which party is addressed
                addressed = "plaintiff" if "plaintiff" in judge_response.lower() else "defense"
                addressed_id = plaintiff_id if addressed == "plaintiff" else defense_id
                addressed_cfg = self.agents[addressed_id]
                addressed_label = "Plaintiff" if addressed == "plaintiff" else "Defense"

                if self.verbose:
                    print(f"\n  [Judge requests clarification from {addressed_label}]")

                # Add judge question to transcript
                transcript.add_turn(judge_id, "Judge (Clarification)", judge_response, 0)
                transcript.clarifications.append({
                    "question": judge_response,
                    "addressed_to": addressed_label,
                })

                # Get clarification from the addressed party
                clar_state = transcript.format_state()
                clar_prompt = (
                    f"The Judge has requested clarification. "
                    f"Please answer the Judge's question directly."
                )
                clar_response = self._agent_turn(
                    agent_cfg=addressed_cfg,
                    state=clar_state,
                    user_prompt=clar_prompt,
                    transcript=transcript,
                    round_num=0,
                    role_label=f"{addressed_label} (Clarification)",
                )
                transcript.add_turn(addressed_id, f"{addressed_label} (Clarification)", clar_response, 0)

                # Update state and loop back to Judge
                state = transcript.format_state(for_agent=judge_id)
                user_prompt = (
                    "The party has responded to your clarification request. "
                    "The updated transcript is above. Now issue your final ruling."
                )
                continue

            # No clarification — we have a ruling
            transcript.add_turn(judge_id, "Judge (Ruling)", judge_response, 0)
            return _parse_ruling(judge_response)

        # Exhausted clarification budget — force-parse whatever we have
        transcript.add_turn(judge_id, "Judge (Ruling)", judge_response, 0)
        return _parse_ruling(judge_response)


# ---------------------------------------------------------------------------
# Factory: create a CourtOrchestrator from a local HF model
# ---------------------------------------------------------------------------

def create_court_from_model(
    model,
    tokenizer,
    device: str,
    tool_registry: Optional[ToolRegistry] = None,
    verbose: bool = True,
) -> CourtOrchestrator:
    """
    Convenience factory that wraps a HuggingFace model into a CourtOrchestrator.

    Args:
        model:         HuggingFace model (AutoModelForCausalLM)
        tokenizer:     corresponding tokenizer
        device:        "cuda", "cpu", etc.
        tool_registry: optional ToolRegistry
        verbose:       print progress

    Returns:
        A ready-to-use CourtOrchestrator.
    """
    import torch
    import torch.nn.functional as F

    def generate_fn(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> str:
        full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nYour response:"
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return response

    return CourtOrchestrator(
        generate_fn=generate_fn,
        tool_registry=tool_registry,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Factory: create a CourtOrchestrator from an API-based LLM
# ---------------------------------------------------------------------------

def create_court_from_api(
    api_call_fn: Callable[..., str],
    tool_registry: Optional[ToolRegistry] = None,
    verbose: bool = True,
) -> CourtOrchestrator:
    """
    Convenience factory for API-backed models (OpenAI, Anthropic, etc.).

    Args:
        api_call_fn:  callable(system_prompt, user_prompt, temperature, max_tokens) -> str
        tool_registry: optional ToolRegistry
        verbose:       print progress

    Returns:
        A ready-to-use CourtOrchestrator.
    """
    return CourtOrchestrator(
        generate_fn=api_call_fn,
        tool_registry=tool_registry,
        verbose=verbose,
    )
