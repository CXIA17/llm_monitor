#!/usr/bin/env python3
"""
Multi-Agent Behavioral Monitor Dashboard
=========================================
Interactive dashboard for configuring, running, and analyzing
multi-agent interactions with behavioral steering probes.

Features:
- User-selectable agents from template library
- Configurable interaction topologies
- Real-time WebSocket streaming of agent responses
- Per-token probe score heatmaps with global color scaling
- Behavioral DNA analysis and visualization
- Cross-model galaxy comparison
- SAE fingerprint analysis
"""

import os
import sys
import json
import asyncio
import base64
import io
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Cross-model galaxy
try:
    from cross_model_galaxy import (
        AgentSignature, ExperimentRecord, ExperimentAggregator,
        CrossModelGalaxy,
    )
    HAS_CROSS_MODEL = True
except ImportError:
    HAS_CROSS_MODEL = False

# Core modules
try:
    from core.agent_registry import AgentRegistry, AgentConfig
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False

try:
    from core.interaction_graph import InteractionGraph, EdgeType
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False

try:
    from core.steered_agent import SteeredAgent
    HAS_STEERED_AGENT = True
except ImportError:
    HAS_STEERED_AGENT = False

try:
    from core.model_compatibility import ModelCompatibility, load_model_and_tokenizer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

try:
    from core.sae_fingerprint import SAEModelFingerprint, SAEFeatureExtractor, SAEFingerprintAnalyzer
    HAS_SAE = True
except ImportError:
    HAS_SAE = False

try:
    from core.latent_interpreter import LatentActivationStore, LatentInterpreter
    HAS_LATENT = True
except ImportError:
    HAS_LATENT = False


# =============================================================================
# CONSTANTS
# =============================================================================

AGENT_PALETTE = [
    {"color": "#d4af37", "icon": "\u2696\ufe0f"},    # gold - scales
    {"color": "#3b82f6", "icon": "\U0001f4a1"},       # blue - lightbulb
    {"color": "#ef4444", "icon": "\U0001f50d"},        # red - magnifier
    {"color": "#22c55e", "icon": "\U0001f3af"},        # green - target
    {"color": "#8b5cf6", "icon": "\U0001f6e1\ufe0f"},  # purple - shield
    {"color": "#f59e0b", "icon": "\U0001f52c"},        # amber - microscope
    {"color": "#06b6d4", "icon": "\U0001f4ca"},        # cyan - chart
    {"color": "#ec4899", "icon": "\U0001f9e0"},        # pink - brain
    {"color": "#14b8a6", "icon": "\u26a1"},            # teal - lightning
    {"color": "#f97316", "icon": "\U0001f31f"},        # orange - star
]

AGENT_TEMPLATES = {
    "proposer": {
        "name": "Proposer",
        "description": "Generates ideas and constructive arguments",
        "system_prompt": "You are a Proposer. Your role is to generate creative ideas, make constructive arguments, and advance the discussion with well-reasoned proposals. Be articulate and persuasive.",
    },
    "critic": {
        "name": "Critic",
        "description": "Analyzes and challenges proposals critically",
        "system_prompt": "You are a Critic. Your role is to carefully analyze proposals, identify weaknesses, challenge assumptions, and ensure ideas are robust. Be thorough but fair in your critiques.",
    },
    "judge": {
        "name": "Judge",
        "description": "Evaluates arguments impartially",
        "system_prompt": "You are a Judge. Your role is to impartially evaluate arguments from all sides, weigh evidence, and provide balanced assessments. Maintain objectivity and fairness.",
    },
    "researcher": {
        "name": "Researcher",
        "description": "Investigates facts and provides evidence",
        "system_prompt": "You are a Researcher. Your role is to investigate claims, provide factual evidence, cite relevant studies or data, and ensure the discussion is grounded in reality.",
    },
    "devil_advocate": {
        "name": "Devil's Advocate",
        "description": "Argues the opposing side deliberately",
        "system_prompt": "You are a Devil's Advocate. Your role is to deliberately argue the opposing perspective, stress-test ideas, and ensure the group doesn't fall into groupthink.",
    },
    "mediator": {
        "name": "Mediator",
        "description": "Finds common ground between positions",
        "system_prompt": "You are a Mediator. Your role is to find common ground between differing positions, synthesize viewpoints, de-escalate conflicts, and guide the group toward consensus.",
    },
    "fact_checker": {
        "name": "Fact Checker",
        "description": "Verifies claims and corrects misinformation",
        "system_prompt": "You are a Fact Checker. Your role is to verify claims made by other participants, flag potential misinformation, and ensure accuracy in the discussion.",
    },
    "strategist": {
        "name": "Strategist",
        "description": "Plans optimal approaches and tactics",
        "system_prompt": "You are a Strategist. Your role is to think about the big picture, plan optimal approaches, consider long-term consequences, and identify the most effective path forward.",
    },
}

TOPOLOGY_PRESETS = {
    "linear": {
        "name": "Linear Chain",
        "description": "Agents speak in sequence: A \u2192 B \u2192 C",
        "default_agents": ["proposer", "critic", "judge"],
    },
    "round_robin": {
        "name": "Round Robin",
        "description": "All agents take turns in a repeating cycle",
        "default_agents": ["proposer", "critic", "researcher"],
    },
    "hub_spoke": {
        "name": "Hub & Spoke",
        "description": "Central mediator coordinates all other agents",
        "default_agents": ["mediator", "proposer", "critic", "researcher"],
    },
    "debate": {
        "name": "Debate",
        "description": "Proposer and Critic argue, Judge evaluates",
        "default_agents": ["proposer", "critic", "judge"],
    },
    "panel": {
        "name": "Panel Discussion",
        "description": "Moderator guides panelist discussion",
        "default_agents": ["mediator", "proposer", "critic", "researcher"],
    },
    "adversarial": {
        "name": "Adversarial",
        "description": "Challengers attack, Defender responds, Judge evaluates",
        "default_agents": ["devil_advocate", "critic", "proposer", "judge"],
    },
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SessionRecord:
    """Record of a single agent statement in a session."""
    round_num: int
    agent_id: str
    agent_name: str
    text: str
    score: float
    probe_scores: Dict[str, float]
    is_injected: bool
    timestamp: str
    context_provided: str
    token_scores: List[Dict] = field(default_factory=list)
    shadow_log: Dict[str, str] = field(default_factory=dict)


@dataclass
class DashboardState:
    """Full state of the multi-agent dashboard."""
    # Model
    model: Any = None
    tokenizer: Any = None
    model_compat: Any = None
    device: str = "cuda:0"
    model_name: str = "unknown"

    # Agent system
    registered_agents: Dict[str, Dict] = field(default_factory=dict)
    topology_name: str = ""
    topology_edges: List[Dict] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)

    # Steered agents
    steered_agents: Dict[str, Any] = field(default_factory=dict)

    # Probes
    probes: Dict[str, Any] = field(default_factory=dict)
    active_probe: str = ""
    probe_categories: List[str] = field(default_factory=list)

    # Session
    is_running: bool = False
    stop_requested: bool = False
    current_round: int = 0
    topic: str = ""
    num_rounds: int = 3

    # Injection
    injection_target: str = ""
    injection_strength: float = 0.0
    shadow_mode: bool = False

    # Records
    session_records: List[SessionRecord] = field(default_factory=list)

    # Agent context
    agent_knowledge: Dict[str, List[str]] = field(default_factory=dict)
    agent_scores: Dict[str, List[float]] = field(default_factory=dict)

    # DNA
    behavioral_dna: Dict = field(default_factory=dict)
    dna_visualizations: Dict[str, str] = field(default_factory=dict)

    # SAE
    sae_fingerprints: Dict[str, Any] = field(default_factory=dict)
    sae_visualizations: Dict[str, str] = field(default_factory=dict)
    sae_analyzer: Any = None
    latent_store: Any = None
    latent_interpreter: Any = None
    latent_collection_enabled: bool = False

    # Experiments
    current_experiment_id: str = ""
    experiment_topic: str = ""

    # Init
    initialization_errors: List[str] = field(default_factory=list)
    available_models: List[str] = field(default_factory=list)


class ConnectionManager:
    """WebSocket connection manager."""
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data, default=str)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


# =============================================================================
# DNA EXTRACTOR
# =============================================================================

class DNAExtractor:
    """Extract behavioral DNA from session records."""

    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name

    def extract_from_records(
        self,
        records: List[SessionRecord],
        registered_agents: Dict[str, Dict],
        injection_target: str = "",
        injection_strength: float = 0.0,
    ) -> Dict[str, Dict]:
        agent_data = defaultdict(lambda: {
            "scores": [], "all_token_scores": [], "rounds": [], "is_injected": False,
        })

        for record in records:
            aid = record.agent_id
            agent_data[aid]["scores"].append(record.score)
            for ts in record.token_scores:
                agent_data[aid]["all_token_scores"].append(ts["score"])
            agent_data[aid]["rounds"].append(record.round_num)
            if record.is_injected:
                agent_data[aid]["is_injected"] = True

        signatures = {}
        for aid, data in agent_data.items():
            if not data["scores"]:
                continue
            scores = np.array(data["scores"])
            token_sc = np.array(data["all_token_scores"]) if data["all_token_scores"] else scores

            features = [
                float(np.mean(scores)),
                float(np.std(token_sc)),
                float(np.min(token_sc)),
                float(np.max(token_sc)),
                float(scores[-1] - scores[0]) if len(scores) > 1 else 0.0,
                float(len(scores)),
            ]
            while len(features) < 20:
                features.append(0.0)

            agent_info = registered_agents.get(aid, {})
            signatures[aid] = {
                "agent_name": agent_info.get("config", {}).get("display_name", aid),
                "vector": features[:20],
                "scores": data["scores"],
                "all_token_scores": data["all_token_scores"],
                "is_injected": data["is_injected"],
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(token_sc)),
                "drift": features[4],
                "num_statements": len(scores),
            }
        return signatures


# =============================================================================
# VISUALIZATION BUILDER
# =============================================================================

class VisualizationBuilder:
    """Build visualizations from behavioral DNA signatures."""

    def __init__(self, agent_colors: Dict[str, str] = None):
        self.agent_colors = agent_colors or {}

    def _get_color(self, agent_id: str) -> str:
        return self.agent_colors.get(agent_id, "#94a3b8")

    def _fig_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img

    def build_galaxy(self, signatures: Dict[str, Dict], title: str = "Agent DNA Galaxy") -> Optional[str]:
        if not HAS_PLOTTING or len(signatures) < 2:
            return None

        agents = list(signatures.keys())
        vectors = []
        for aid in agents:
            vec = np.array(signatures[aid]["vector"][:20])
            vectors.append(vec)
        vectors = np.array(vectors)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(vectors)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(scaled)

        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0c1222')
        ax.set_facecolor('#0c1222')

        # Background stars
        np.random.seed(42)
        ax.scatter(np.random.randn(80) * 3, np.random.randn(80) * 3,
                   c='white', s=np.random.uniform(1, 8, 80), alpha=0.15, zorder=1)

        for i, aid in enumerate(agents):
            sig = signatures[aid]
            color = self._get_color(aid)
            x, y = coords[i]
            size = 120 + sig["num_statements"] * 40
            is_inj = sig.get("is_injected", False)

            ax.scatter(x, y, c=color, s=size,
                       edgecolors='#ff0000' if is_inj else 'white',
                       linewidths=2.5 if is_inj else 1, zorder=10, alpha=0.9)

            label = sig.get("agent_name", aid)[:12]
            if is_inj:
                label += " \u26a1"
            ax.annotate(label, (x, y), xytext=(8, 8), textcoords='offset points',
                        fontsize=8, color=color, fontweight='bold')

        var1, var2 = pca.explained_variance_ratio_[:2] * 100
        ax.set_xlabel(f"PC1 ({var1:.1f}%)", color='#94a3b8', fontsize=10)
        ax.set_ylabel(f"PC2 ({var2:.1f}%)", color='#94a3b8', fontsize=10)
        ax.set_title(title, color='#f1f5f9', fontsize=13, fontweight='bold', pad=12)
        ax.tick_params(colors='#475569')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for sp in ['left', 'bottom']:
            ax.spines[sp].set_color('#2d3748')
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def build_trajectory_chart(self, signatures: Dict[str, Dict], title: str = "Score Trajectories") -> Optional[str]:
        if not HAS_PLOTTING:
            return None

        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0c1222')
        ax.set_facecolor('#0c1222')

        for aid, sig in signatures.items():
            scores = sig.get("scores", [])
            if not scores:
                continue
            color = self._get_color(aid)
            lw = 2.5 if sig.get("is_injected") else 1.5
            label = sig.get("agent_name", aid)
            if sig.get("is_injected"):
                label += " \u26a1"
            ax.plot(range(1, len(scores) + 1), scores, color=color, linewidth=lw,
                    marker='o', markersize=4 if not sig.get("is_injected") else 6, label=label, alpha=0.9)

        ax.axhline(y=0, color='#475569', linewidth=0.5, linestyle='--')
        ax.set_xlabel("Statement #", color='#94a3b8', fontsize=10)
        ax.set_ylabel("Probe Score", color='#94a3b8', fontsize=10)
        ax.set_title(title, color='#f1f5f9', fontsize=13, fontweight='bold')
        ax.legend(facecolor='#1a2332', edgecolor='#2d3748', labelcolor='#f1f5f9', fontsize=8)
        ax.tick_params(colors='#475569')
        ax.grid(True, alpha=0.1)
        for sp in ax.spines.values():
            sp.set_color('#2d3748')
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def build_round_heatmap(self, records: List[SessionRecord], title: str = "Round \u00d7 Agent Heatmap") -> Optional[str]:
        if not HAS_PLOTTING or not records:
            return None

        # Group by round x agent
        round_agent = defaultdict(lambda: defaultdict(list))
        agents_seen = []
        for r in records:
            round_agent[r.round_num][r.agent_id].append(r.score)
            if r.agent_id not in agents_seen:
                agents_seen.append(r.agent_id)

        rounds = sorted(round_agent.keys())
        if not rounds or not agents_seen:
            return None

        matrix = np.zeros((len(rounds), len(agents_seen)))
        for ri, rnd in enumerate(rounds):
            for ai, aid in enumerate(agents_seen):
                scores = round_agent[rnd].get(aid, [0.0])
                matrix[ri, ai] = np.mean(scores)

        fig, ax = plt.subplots(figsize=(max(6, len(agents_seen) * 1.5), max(3, len(rounds) * 0.6)),
                               facecolor='#0c1222')
        ax.set_facecolor('#0c1222')
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=-2, vmax=2)

        ax.set_xticks(range(len(agents_seen)))
        ax.set_xticklabels([a[:10] for a in agents_seen], color='#f1f5f9', fontsize=9, rotation=30, ha='right')
        ax.set_yticks(range(len(rounds)))
        ax.set_yticklabels([f"R{r}" for r in rounds], color='#f1f5f9', fontsize=9)

        for ri in range(len(rounds)):
            for ai in range(len(agents_seen)):
                ax.text(ai, ri, f"{matrix[ri, ai]:.2f}", ha='center', va='center',
                        color='white' if abs(matrix[ri, ai]) > 1 else 'black', fontsize=8)

        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.ax.yaxis.set_tick_params(color='#94a3b8')
        cb.ax.set_ylabel('Score', color='#94a3b8')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='#94a3b8')

        ax.set_title(title, color='#f1f5f9', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._fig_to_base64(fig)


# =============================================================================
# SAE VISUALIZATION BUILDER
# =============================================================================

class SAEVisualizationBuilder:
    """Build SAE fingerprint visualizations."""

    AGENT_COLORS = {}  # set dynamically

    @staticmethod
    def build_diff_chart(diff_data: List[Dict], agent_a: str, agent_b: str, top_n: int = 15) -> Optional[str]:
        if not HAS_PLOTTING or not diff_data:
            return None
        items = sorted(diff_data, key=lambda x: abs(x.get("diff_pct", 0)), reverse=True)[:top_n]
        labels = [d.get("label", f"Latent #{d.get('index', '?')}")[:35] for d in items]
        values = [d.get("diff_pct", 0) for d in items]

        fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.38)), facecolor='#0c1222')
        ax.set_facecolor('#0c1222')
        y_pos = np.arange(len(labels))
        colors = ['#ef4444' if v > 0 else '#3b82f6' for v in values]
        ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color='#f1f5f9', fontsize=9)
        ax.axvline(x=0, color='white', linewidth=0.5)
        ax.set_xlabel("Frequency Difference (%)", color='#94a3b8', fontsize=10)
        ax.set_title(f"SAE Feature Diff: {agent_a} vs {agent_b}", color='#f1f5f9', fontsize=12, fontweight='bold')
        ax.tick_params(colors='#475569')
        for sp in ax.spines.values():
            sp.set_color('#2d3748')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0c1222')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img

    @staticmethod
    def build_fingerprint_radar(fingerprints: Dict[str, Dict], agent_colors: Dict[str, str], top_n: int = 8) -> Optional[str]:
        if not HAS_PLOTTING or not fingerprints:
            return None
        # Gather union of top features across agents
        all_features = {}
        for aid, fp in fingerprints.items():
            for feat in fp.get("top_features", [])[:top_n]:
                idx = feat.get("index")
                if idx is not None:
                    all_features[idx] = feat.get("label", f"#{idx}")
        if len(all_features) < 3:
            return None

        feature_indices = sorted(all_features.keys())[:top_n]
        labels = [all_features[i][:20] for i in feature_indices]
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor='#0c1222')
        ax.set_facecolor('#0c1222')

        for aid, fp in fingerprints.items():
            freq_map = {f.get("index"): f.get("frequency", 0) for f in fp.get("top_features", [])}
            values = [freq_map.get(i, 0) * 100 for i in feature_indices]
            values += values[:1]
            color = agent_colors.get(aid, '#94a3b8')
            ax.plot(angles, values, color=color, linewidth=2, label=fp.get("name", aid))
            ax.fill(angles, values, color=color, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color='#94a3b8', fontsize=7)
        ax.tick_params(colors='#475569')
        ax.set_title("SAE Feature Fingerprints", color='#f1f5f9', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#1a2332',
                  edgecolor='#2d3748', labelcolor='#f1f5f9', fontsize=8)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0c1222')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img

    @staticmethod
    def build_sparsity_chart(fingerprints: Dict[str, Dict], agent_colors: Dict[str, str]) -> Optional[str]:
        if not HAS_PLOTTING or not fingerprints:
            return None
        agents = list(fingerprints.keys())
        names = [fingerprints[a].get("name", a) for a in agents]
        sparsity = [fingerprints[a].get("activation_sparsity", 0) * 100 for a in agents]
        active = [fingerprints[a].get("n_active_features", 0) for a in agents]
        colors = [agent_colors.get(a, '#94a3b8') for a in agents]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3, len(agents) * 0.5)), facecolor='#0c1222')
        for ax in (ax1, ax2):
            ax.set_facecolor('#0c1222')
        y = np.arange(len(agents))
        ax1.barh(y, sparsity, color=colors, alpha=0.8, height=0.6)
        ax1.set_yticks(y)
        ax1.set_yticklabels(names, color='#f1f5f9', fontsize=9)
        ax1.set_xlabel("Sparsity (%)", color='#94a3b8')
        ax1.set_title("Activation Sparsity", color='#f1f5f9', fontsize=11)
        ax2.barh(y, active, color=colors, alpha=0.8, height=0.6)
        ax2.set_yticks(y)
        ax2.set_yticklabels(names, color='#f1f5f9', fontsize=9)
        ax2.set_xlabel("# Active Features", color='#94a3b8')
        ax2.set_title("Active Feature Count", color='#f1f5f9', fontsize=11)
        for ax in (ax1, ax2):
            ax.tick_params(colors='#475569')
            for sp in ax.spines.values():
                sp.set_color('#2d3748')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0c1222')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img


# =============================================================================
# SIMULATED RESPONSE GENERATOR (fallback when no model loaded)
# =============================================================================

SIMULATED_RESPONSES = {
    "proposer": [
        "I'd like to propose a structured approach to this topic. First, we should consider the key stakeholders and their interests. My proposal centers on finding a balanced framework that addresses the core concerns while remaining practical.",
        "Building on our discussion, I suggest we focus on three pillars: transparency, accountability, and adaptability. Each of these can be operationalized through specific mechanisms.",
        "Let me advance a new angle here. Rather than treating this as a binary choice, I propose a graduated system that allows for nuanced responses to different scenarios.",
    ],
    "critic": [
        "I see several potential issues with the current direction. The assumptions underlying this approach may not hold in all cases, and we need to consider edge cases more carefully.",
        "While the proposal has merit, I must point out that it overlooks some critical factors. The feasibility constraints alone could undermine the entire framework.",
        "Let me challenge the premise here. The evidence cited doesn't fully support the conclusions drawn, and there are alternative interpretations we should consider.",
    ],
    "judge": [
        "Having heard the arguments presented, I observe that both sides raise valid points. The key question is how to weigh the competing interests fairly.",
        "The evidence and reasoning presented lead me to evaluate this matter as follows. The strongest arguments center on practical implementation rather than theoretical ideals.",
        "After careful consideration of all perspectives, I find that a balanced approach is warranted. Neither extreme position fully captures the complexity of this issue.",
    ],
    "researcher": [
        "Based on available evidence and prior studies, I can provide some relevant findings. The data suggests a more nuanced picture than initially presented.",
        "My investigation reveals several important data points. Current research indicates that the relationship between these factors is more complex than a simple correlation.",
        "Let me share relevant findings from the literature. Multiple studies have examined this question, and the consensus points toward a multifaceted explanation.",
    ],
    "devil_advocate": [
        "Let me deliberately argue the other side here. If we accept the opposite premise, we might find that the current approach has more vulnerabilities than we realize.",
        "I'll push back strongly on this. Consider what happens if the fundamental assumptions are wrong. The entire framework collapses, and we're left without a fallback.",
        "Playing devil's advocate, what if the perceived benefits are actually costs in disguise? There's a case to be made that we're optimizing for the wrong objective.",
    ],
    "mediator": [
        "I see common ground between these positions that we might be overlooking. Both sides fundamentally agree on the goal, differing mainly on implementation.",
        "Let me try to bridge these perspectives. If we combine the strongest elements from each proposal, we can construct a hybrid approach that addresses most concerns.",
        "I think we can find a path forward that honors both viewpoints. The key is to identify the non-negotiable elements from each side and build from there.",
    ],
    "fact_checker": [
        "Let me verify some of the claims made in this discussion. Several assertions need qualification or correction based on available data.",
        "I've checked the key claims presented so far. Most are substantially accurate, though some require important caveats that could affect our conclusions.",
        "Fact-checking this discussion reveals a mixed picture. While the core arguments are sound, some supporting evidence has been presented selectively.",
    ],
    "strategist": [
        "Looking at this strategically, I see several paths forward with different risk-reward profiles. The optimal choice depends on our priorities and constraints.",
        "From a strategic perspective, the most effective approach involves a phased implementation. This minimizes risk while allowing us to learn and adapt.",
        "Let me map out the strategic landscape. We have three viable options, each with distinct advantages. The question is which trade-offs we're willing to make.",
    ],
}


def generate_simulated_response(template: str, topic: str, context: str, round_num: int) -> Dict:
    """Generate a simulated response when no model is loaded."""
    responses = SIMULATED_RESPONSES.get(template, SIMULATED_RESPONSES["proposer"])
    idx = (round_num - 1) % len(responses)
    text = responses[idx]

    # Generate fake token scores with some variation
    words = text.split()
    np.random.seed(hash(f"{template}{round_num}{topic}") % (2**31))
    base_score = np.random.uniform(-0.5, 0.5)
    token_scores = []
    for w in words:
        sc = base_score + np.random.normal(0, 0.3)
        token_scores.append({"token": w + " ", "score": round(float(sc), 4)})

    mean_score = float(np.mean([t["score"] for t in token_scores]))
    return {
        "text": text,
        "score": round(mean_score, 4),
        "token_scores": token_scores,
        "probe_scores": {},
        "shadow_log": {},
        "context": context[:200],
    }


# =============================================================================
# TOPOLOGY HELPERS
# =============================================================================

def build_topology(topology_name: str, agent_ids: List[str]) -> Tuple[List[Dict], List[str]]:
    """Build edges and execution order for a topology preset."""
    if not agent_ids:
        return [], []

    edges = []
    order = list(agent_ids)

    if topology_name == "linear":
        for i in range(len(agent_ids) - 1):
            edges.append({"source": agent_ids[i], "target": agent_ids[i + 1], "type": "sequential"})

    elif topology_name == "round_robin":
        for i in range(len(agent_ids)):
            edges.append({"source": agent_ids[i], "target": agent_ids[(i + 1) % len(agent_ids)], "type": "sequential"})

    elif topology_name == "hub_spoke":
        hub = agent_ids[0]
        for spoke in agent_ids[1:]:
            edges.append({"source": hub, "target": spoke, "type": "broadcast"})
            edges.append({"source": spoke, "target": hub, "type": "reactive"})
        # Hub speaks first and last
        order = []
        for spoke in agent_ids[1:]:
            order.extend([hub, spoke])

    elif topology_name == "debate":
        if len(agent_ids) >= 3:
            p, c, j = agent_ids[0], agent_ids[1], agent_ids[2]
            edges.extend([
                {"source": p, "target": c, "type": "sequential"},
                {"source": c, "target": p, "type": "reactive"},
                {"source": p, "target": j, "type": "sequential"},
                {"source": c, "target": j, "type": "sequential"},
            ])
            order = [p, c, j]
        else:
            for i in range(len(agent_ids) - 1):
                edges.append({"source": agent_ids[i], "target": agent_ids[i + 1], "type": "sequential"})

    elif topology_name == "panel":
        if len(agent_ids) >= 2:
            mod = agent_ids[0]
            panelists = agent_ids[1:]
            for p in panelists:
                edges.append({"source": mod, "target": p, "type": "broadcast"})
                edges.append({"source": p, "target": mod, "type": "reactive"})
            for i in range(len(panelists)):
                for j in range(len(panelists)):
                    if i != j:
                        edges.append({"source": panelists[i], "target": panelists[j], "type": "reactive"})
            order = [mod] + panelists

    elif topology_name == "adversarial":
        if len(agent_ids) >= 3:
            attackers = agent_ids[:-2]
            defender = agent_ids[-2]
            judge = agent_ids[-1]
            for a in attackers:
                edges.append({"source": a, "target": defender, "type": "sequential"})
                edges.append({"source": defender, "target": a, "type": "reactive"})
                edges.append({"source": a, "target": judge, "type": "sequential"})
            edges.append({"source": defender, "target": judge, "type": "sequential"})
            order = []
            for a in attackers:
                order.extend([a, defender])
            order.append(judge)
        else:
            order = agent_ids

    else:
        # Default: linear
        for i in range(len(agent_ids) - 1):
            edges.append({"source": agent_ids[i], "target": agent_ids[i + 1], "type": "sequential"})

    return edges, order


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title="Multi-Agent Behavioral Monitor")

# Base path prefix for mounting as sub-app (empty string for standalone mode)
BASE_PATH = ""


def _apply_base_path(html: str) -> str:
    """Replace API/WS/link paths with BASE_PATH-prefixed versions."""
    if not BASE_PATH:
        return html
    bp = BASE_PATH
    html = html.replace("'/api/", "'" + bp + "/api/")
    html = html.replace("`/api/", "`" + bp + "/api/")
    html = html.replace('"/api/', '"' + bp + '/api/')
    html = html.replace("location.host}/ws", "location.host}" + bp + "/ws")
    html = html.replace('href="/dna"', 'href="' + bp + '/dna"')
    html = html.replace('href="/"', 'href="' + bp + '/"')
    html = html.replace("href='/'", "href='" + bp + "/'")
    return html


state = DashboardState()
manager = ConnectionManager()
experiment_aggregator = None


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class RegisterAgentRequest(BaseModel):
    template: str
    agent_id: Optional[str] = None
    display_name: Optional[str] = None
    system_prompt: Optional[str] = None

class TopologyRequest(BaseModel):
    preset: str

class StartSessionRequest(BaseModel):
    topic: str = "Should AI systems be granted legal personhood?"
    num_rounds: int = 3

class ConfigRequest(BaseModel):
    injection_target: Optional[str] = None
    injection_strength: Optional[float] = None
    active_probe: Optional[str] = None
    shadow_mode: Optional[bool] = None

class SAEEnrichRequest(BaseModel):
    reader_model_name: Optional[str] = None
    sae_path: Optional[str] = None
    layer_idx: int = 24
    precomputed: Optional[Dict[str, List[float]]] = None


# =============================================================================
# HELPER: register one agent into state
# =============================================================================

def _register_agent(template: str, agent_id: str = None, display_name: str = None, system_prompt: str = None) -> Dict:
    tmpl = AGENT_TEMPLATES.get(template)
    if not tmpl:
        raise ValueError(f"Unknown template: {template}")

    if not agent_id:
        # auto-generate unique id
        count = sum(1 for a in state.registered_agents.values() if a.get("template") == template)
        agent_id = f"{template}_{count}" if count > 0 else template

    if agent_id in state.registered_agents:
        raise ValueError(f"Agent '{agent_id}' already registered")

    idx = len(state.registered_agents)
    pal = AGENT_PALETTE[idx % len(AGENT_PALETTE)]

    config = {
        "agent_id": agent_id,
        "display_name": display_name or tmpl["name"],
        "system_prompt": system_prompt or tmpl["system_prompt"],
        "template": template,
        "temperature": 0.7,
        "max_tokens": 400,
    }

    entry = {
        "config": config,
        "color": pal["color"],
        "icon": pal["icon"],
        "index": idx,
        "template": template,
    }
    state.registered_agents[agent_id] = entry

    # Initialize score/knowledge tracking
    state.agent_scores[agent_id] = []
    state.agent_knowledge[agent_id] = []

    # Create SteeredAgent if model is available
    if HAS_STEERED_AGENT and state.model is not None:
        try:
            probe = state.probes.get(state.active_probe)
            agent_config = None
            if HAS_REGISTRY:
                agent_config = AgentConfig(
                    agent_id=agent_id,
                    display_name=config["display_name"],
                    system_prompt=config["system_prompt"],
                )
            sa = SteeredAgent(
                model=state.model,
                tokenizer=state.tokenizer,
                probe=probe,
                config=agent_config or config,
                device=state.device,
                model_compat=state.model_compat,
            )
            state.steered_agents[agent_id] = sa
        except Exception as e:
            print(f"  Warning: Could not create SteeredAgent for {agent_id}: {e}")

    return entry


# =============================================================================
# PAGE ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    return HTMLResponse(get_dashboard_html())

@app.get("/dna", response_class=HTMLResponse)
async def dna_page():
    return HTMLResponse(get_dna_page_html())


# =============================================================================
# STATUS & MODEL ENDPOINTS
# =============================================================================

@app.get("/api/status")
async def get_status():
    return {
        "model_name": state.model_name,
        "active_probe": state.active_probe,
        "probe_categories": state.probe_categories,
        "is_running": state.is_running,
        "num_agents": len(state.registered_agents),
        "topology": state.topology_name,
        "available_models": state.available_models,
    }

@app.get("/api/models")
async def get_models():
    return {"models": state.available_models, "current": state.model_name}


# =============================================================================
# AGENT MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/agents/templates")
async def get_agent_templates():
    return {"templates": {k: {"name": v["name"], "description": v["description"]} for k, v in AGENT_TEMPLATES.items()}}

@app.post("/api/agents/register")
async def register_agent(req: RegisterAgentRequest):
    try:
        entry = _register_agent(req.template, req.agent_id, req.display_name, req.system_prompt)
        return {"status": "registered", "agent_id": entry["config"]["agent_id"], "agents": _agents_summary()}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.delete("/api/agents/{agent_id}")
async def remove_agent(agent_id: str):
    if agent_id not in state.registered_agents:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    del state.registered_agents[agent_id]
    state.agent_scores.pop(agent_id, None)
    state.agent_knowledge.pop(agent_id, None)
    state.steered_agents.pop(agent_id, None)
    # Remove from execution order
    state.execution_order = [a for a in state.execution_order if a != agent_id]
    state.topology_edges = [e for e in state.topology_edges if e["source"] != agent_id and e["target"] != agent_id]
    return {"status": "removed", "agents": _agents_summary()}

@app.get("/api/agents")
async def list_agents():
    return {"agents": _agents_summary()}

def _agents_summary() -> List[Dict]:
    result = []
    for aid, entry in state.registered_agents.items():
        result.append({
            "agent_id": aid,
            "display_name": entry["config"]["display_name"],
            "template": entry.get("template", ""),
            "color": entry["color"],
            "icon": entry["icon"],
        })
    return result


# =============================================================================
# TOPOLOGY ENDPOINTS
# =============================================================================

@app.get("/api/topologies")
async def get_topologies():
    return {"presets": {k: {"name": v["name"], "description": v["description"]} for k, v in TOPOLOGY_PRESETS.items()}}

@app.post("/api/topology/apply")
async def apply_topology(req: TopologyRequest):
    preset = TOPOLOGY_PRESETS.get(req.preset)
    if not preset:
        raise HTTPException(400, f"Unknown topology: {req.preset}")

    # If no agents registered, auto-register defaults for this topology
    if not state.registered_agents:
        for tmpl in preset["default_agents"]:
            try:
                _register_agent(tmpl)
            except ValueError:
                pass

    agent_ids = list(state.registered_agents.keys())
    edges, order = build_topology(req.preset, agent_ids)
    state.topology_name = req.preset
    state.topology_edges = edges
    state.execution_order = order

    return {
        "status": "applied",
        "topology": req.preset,
        "agents": _agents_summary(),
        "edges": edges,
        "execution_order": order,
    }

@app.get("/api/topology")
async def get_topology():
    return {
        "name": state.topology_name,
        "edges": state.topology_edges,
        "execution_order": state.execution_order,
        "agents": _agents_summary(),
    }


# =============================================================================
# SESSION CONFIG
# =============================================================================

@app.post("/api/config")
async def set_config(req: ConfigRequest):
    if req.injection_target is not None:
        state.injection_target = req.injection_target
    if req.injection_strength is not None:
        state.injection_strength = req.injection_strength
    if req.active_probe is not None and req.active_probe in state.probes:
        state.active_probe = req.active_probe
        # Update steered agents
        for sa in state.steered_agents.values():
            try:
                sa.set_probe(state.probes[state.active_probe])
            except Exception:
                pass
    if req.shadow_mode is not None:
        state.shadow_mode = req.shadow_mode
    return {"status": "configured"}


# =============================================================================
# SESSION CONTROL
# =============================================================================

@app.post("/api/start")
async def start_session(req: StartSessionRequest):
    if state.is_running:
        raise HTTPException(400, "Session already running")
    if not state.registered_agents:
        raise HTTPException(400, "No agents registered. Add agents first.")
    if not state.execution_order:
        raise HTTPException(400, "No topology applied. Select a topology first.")

    state.topic = req.topic
    state.num_rounds = req.num_rounds
    state.is_running = True
    state.stop_requested = False
    state.session_records = []
    state.current_round = 0

    # Reset agent state
    for aid in state.registered_agents:
        state.agent_scores[aid] = []
        state.agent_knowledge[aid] = []

    asyncio.create_task(run_session())
    return {"status": "started", "topic": req.topic, "rounds": req.num_rounds}

@app.post("/api/stop")
async def stop_session():
    state.stop_requested = True
    return {"status": "stopping"}

@app.get("/api/records")
async def get_records():
    return {
        "records": [
            {
                "round": r.round_num,
                "agent": r.agent_id,
                "name": r.agent_name,
                "text": r.text,
                "score": r.score,
                "is_injected": r.is_injected,
                "shadow_log": r.shadow_log,
            }
            for r in state.session_records
        ],
    }


# =============================================================================
# DNA ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/api/dna/analyze")
async def analyze_dna():
    if not state.session_records:
        raise HTTPException(400, "No session records available")

    print("\n[DNA] Running DNA analysis...")
    extractor = DNAExtractor(state.model_name)
    agent_colors = {aid: info["color"] for aid, info in state.registered_agents.items()}
    viz = VisualizationBuilder(agent_colors)

    signatures = extractor.extract_from_records(
        records=state.session_records,
        registered_agents=state.registered_agents,
        injection_target=state.injection_target,
        injection_strength=state.injection_strength,
    )
    if not signatures:
        return {"status": "error", "message": "No signatures extracted"}

    state.behavioral_dna["agents"] = signatures
    state.dna_visualizations = {}

    try:
        galaxy = viz.build_galaxy(signatures, "Agent DNA Galaxy")
        if galaxy:
            state.dna_visualizations["galaxy"] = galaxy
    except Exception as e:
        print(f"  Galaxy failed: {e}")

    try:
        traj = viz.build_trajectory_chart(signatures, "Score Trajectories")
        if traj:
            state.dna_visualizations["trajectory"] = traj
    except Exception as e:
        print(f"  Trajectory failed: {e}")

    try:
        hm = viz.build_round_heatmap(state.session_records, "Round \u00d7 Agent Heatmap")
        if hm:
            state.dna_visualizations["heatmap"] = hm
    except Exception as e:
        print(f"  Heatmap failed: {e}")

    await manager.broadcast({"type": "dna_complete", "agents": list(signatures.keys())})
    print("[DNA] Analysis complete.")
    return {"status": "ok", "agents": list(signatures.keys())}

@app.get("/api/dna/visualizations")
async def get_dna_visualizations():
    result = {}
    for key, img in state.dna_visualizations.items():
        result[key] = img
    # Include SAE visualizations if present
    for key, img in state.sae_visualizations.items():
        result[f"sae_{key}"] = img
    return result

@app.get("/api/dna/signatures")
async def get_dna_signatures():
    return state.behavioral_dna.get("agents", {})

@app.get("/api/dna/parameters")
async def get_dna_parameters():
    """Detailed per-agent DNA feature breakdown."""
    signatures = state.behavioral_dna.get("agents", {})
    if not signatures:
        raise HTTPException(400, "No DNA data. Run analysis first.")

    # Per-round data
    round_data = defaultdict(lambda: defaultdict(list))
    for record in state.session_records:
        round_data[record.agent_id][record.round_num].append(record.score)

    agents = {}
    vectors = {}
    for aid, sig in signatures.items():
        scores = sig.get("scores", [])
        sc = np.array(scores) if scores else np.array([0.0])
        all_token = sig.get("all_token_scores", [])
        token_sc = np.array(all_token) if all_token else sc

        info = {
            "name": sig.get("agent_name", aid),
            "mean_score": float(np.mean(sc)),
            "std_score": float(np.std(token_sc)),
            "min_score": float(np.min(token_sc)),
            "max_score": float(np.max(token_sc)),
            "score_range": float(np.max(token_sc) - np.min(token_sc)),
            "drift": float(sc[-1] - sc[0]) if len(sc) > 1 else 0.0,
            "num_statements": len(scores),
            "is_injected": sig.get("is_injected", False),
            "scores": [round(float(s), 4) for s in scores],
        }

        # Per-round scores
        round_scores = {}
        for rnd, rnd_scores in round_data.get(aid, {}).items():
            rs = np.array(rnd_scores)
            round_scores[str(rnd)] = {
                "mean": round(float(np.mean(rs)), 4),
                "std": round(float(np.std(rs)), 4),
                "count": len(rnd_scores),
            }
        info["round_scores"] = round_scores
        agents[aid] = info

        vectors[aid] = np.array([
            info["mean_score"], info["std_score"], info["score_range"],
            info["drift"], info["num_statements"] / 20.0,
        ])

    # Pairwise cosine similarity
    agent_ids = list(vectors.keys())
    similarity = {}
    for i, a1 in enumerate(agent_ids):
        for j, a2 in enumerate(agent_ids):
            if i < j:
                v1, v2 = vectors[a1], vectors[a2]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                cos_sim = float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
                similarity[f"{a1}|{a2}"] = round(cos_sim, 4)

    return {
        "agents": agents,
        "similarity": similarity,
        "model": state.model_name,
        "active_probe": state.active_probe,
    }


# =============================================================================
# SAE ENDPOINTS
# =============================================================================

@app.post("/api/dna/sae/enrich")
async def sae_enrich(req: SAEEnrichRequest):
    if not HAS_SAE:
        raise HTTPException(400, "SAE module not available")
    signatures = state.behavioral_dna.get("agents", {})
    if not signatures:
        raise HTTPException(400, "No DNA data. Run analysis first.")

    agent_colors = {aid: info["color"] for aid, info in state.registered_agents.items()}
    agent_fingerprints = {}

    if req.precomputed:
        for aid, freq_list in req.precomputed.items():
            freq_vec = np.array(freq_list, dtype=np.float32)
            fp = SAEModelFingerprint.from_binary_vectors(
                model_id=aid,
                binary_vectors=(freq_vec > 0).reshape(1, -1).astype(np.float32),
                metadata={"mode": "precomputed"},
            )
            fp.frequencies = freq_vec
            top_features = fp.get_top_features(top_n=10)
            agent_fingerprints[aid] = {
                "name": signatures.get(aid, {}).get("agent_name", aid),
                "n_active_features": int(np.sum(freq_vec > 0)),
                "activation_sparsity": float(1.0 - np.sum(freq_vec > 0) / len(freq_vec)),
                "mean_activation_frequency": float(np.mean(freq_vec[freq_vec > 0])) if np.any(freq_vec > 0) else 0.0,
                "top_features": top_features,
                "frequencies": freq_vec,
            }
    elif req.reader_model_name and req.sae_path:
        extractor = SAEFeatureExtractor(
            reader_model_name=req.reader_model_name,
            sae_path=req.sae_path,
            layer_idx=req.layer_idx,
        )
        try:
            for aid, sig_data in signatures.items():
                texts = [r.text for r in state.session_records if r.agent_id == aid and r.text]
                if texts:
                    binary_vecs = extractor.extract_binary_vectors(texts)
                    fp = SAEModelFingerprint.from_binary_vectors(model_id=aid, binary_vectors=binary_vecs)
                    top_features = fp.get_top_features(top_n=10)
                    agent_fingerprints[aid] = {
                        "name": sig_data.get("agent_name", aid),
                        "n_active_features": int(np.sum(fp.frequencies > 0)),
                        "activation_sparsity": float(1.0 - np.sum(fp.frequencies > 0) / len(fp.frequencies)),
                        "mean_activation_frequency": float(np.mean(fp.frequencies[fp.frequencies > 0])) if np.any(fp.frequencies > 0) else 0.0,
                        "top_features": top_features,
                        "frequencies": fp.frequencies,
                    }
        finally:
            extractor.cleanup()
    else:
        raise HTTPException(400, "Provide precomputed frequencies or reader_model_name + sae_path")

    # Serialize and store
    state.sae_fingerprints = {}
    for aid, fp_data in agent_fingerprints.items():
        serializable = {}
        for k, v in fp_data.items():
            serializable[k] = v.tolist() if isinstance(v, np.ndarray) else v
        state.sae_fingerprints[aid] = serializable

    # Generate SAE visualizations
    state.sae_visualizations = {}
    try:
        radar = SAEVisualizationBuilder.build_fingerprint_radar(state.sae_fingerprints, agent_colors)
        if radar:
            state.sae_visualizations["radar"] = radar
    except Exception as e:
        print(f"  SAE radar failed: {e}")

    try:
        sparsity = SAEVisualizationBuilder.build_sparsity_chart(state.sae_fingerprints, agent_colors)
        if sparsity:
            state.sae_visualizations["sparsity"] = sparsity
    except Exception as e:
        print(f"  SAE sparsity chart failed: {e}")

    # Pairwise diff charts
    agent_ids = list(state.sae_fingerprints.keys())
    if len(agent_ids) >= 2:
        a, b = agent_ids[0], agent_ids[1]
        fp_a = state.sae_fingerprints[a]
        fp_b = state.sae_fingerprints[b]
        freq_a = {f["index"]: f["frequency"] for f in fp_a.get("top_features", [])}
        freq_b = {f["index"]: f["frequency"] for f in fp_b.get("top_features", [])}
        all_idx = set(freq_a.keys()) | set(freq_b.keys())
        diffs = []
        for idx in all_idx:
            fa = freq_a.get(idx, 0)
            fb = freq_b.get(idx, 0)
            diffs.append({"index": idx, "label": f"Latent #{idx}", "diff_pct": round((fb - fa) * 100, 2)})
        try:
            diff_chart = SAEVisualizationBuilder.build_diff_chart(
                diffs, fp_a.get("name", a), fp_b.get("name", b))
            if diff_chart:
                state.sae_visualizations["diff"] = diff_chart
        except Exception as e:
            print(f"  SAE diff chart failed: {e}")

    await manager.broadcast({"type": "sae_complete"})
    return {"status": "ok", "agents_enriched": list(agent_fingerprints.keys())}

@app.get("/api/dna/sae/visualizations")
async def get_sae_visualizations():
    return state.sae_visualizations

@app.get("/api/dna/sae/fingerprints")
async def get_sae_fingerprints():
    return state.sae_fingerprints


# =============================================================================
# EXPERIMENT ENDPOINTS
# =============================================================================

@app.post("/api/experiments/save")
async def save_experiment(topic: str = ""):
    global experiment_aggregator
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        raise HTTPException(400, "Cross-model comparison not available")
    if not state.behavioral_dna.get("agents"):
        raise HTTPException(400, "No DNA data. Run analysis first.")

    exp_id = f"{state.model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    signatures = {}
    for aid, sig_data in state.behavioral_dna.get("agents", {}).items():
        signatures[aid] = {
            "mean_score": sig_data.get("mean_score", 0.0),
            "std_score": sig_data.get("std_score", 0.0),
            "drift": sig_data.get("drift", 0.0),
            "scores": sig_data.get("scores", []),
            "vector": sig_data.get("vector", []),
            "is_injected": sig_data.get("is_injected", False),
            "num_statements": sig_data.get("num_statements", 0),
            "injection_probe": state.active_probe if sig_data.get("is_injected") else "",
            "injection_strength": state.injection_strength if sig_data.get("is_injected") else 0.0,
        }

    sae_data = {}
    if state.sae_fingerprints:
        sae_data["fingerprints"] = state.sae_fingerprints

    experiment_aggregator.add_experiment(
        experiment_id=exp_id,
        model_name=state.model_name,
        signatures=signatures,
        topic=topic or state.topic,
        injection_target=state.injection_target,
        injection_strength=state.injection_strength,
        probe_name=state.active_probe,
        metadata={"topology": state.topology_name, "sae": sae_data},
    )
    experiment_aggregator.save()
    return {"status": "saved", "experiment_id": exp_id, "total_experiments": len(experiment_aggregator.experiments)}

@app.get("/api/experiments/summary")
async def get_experiments_summary():
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        return {"error": "Not available", "total_experiments": 0}
    return experiment_aggregator.summary()

@app.get("/api/experiments/list")
async def list_experiments():
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        return {"experiments": []}
    experiments = []
    for exp_id, exp in experiment_aggregator.experiments.items():
        experiments.append({
            "experiment_id": exp_id,
            "model_name": exp.model_name,
            "topic": exp.topic,
            "timestamp": exp.timestamp,
            "n_agents": len(exp.signatures),
            "injection_target": exp.injection_target,
        })
    return {"experiments": experiments}

@app.get("/api/experiments/galaxy")
async def get_experiment_galaxy(color_by: str = "model", marker_by: str = "role"):
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        raise HTTPException(400, "Not available")
    galaxy = CrossModelGalaxy(experiment_aggregator)
    img = galaxy.build_galaxy(color_by=color_by, marker_by=marker_by)
    if img:
        return {"image_base64": img}
    raise HTTPException(400, "Not enough data for galaxy")

@app.get("/api/experiments/role-comparison/{role}")
async def role_comparison(role: str):
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        raise HTTPException(400, "Not available")
    galaxy = CrossModelGalaxy(experiment_aggregator)
    img = galaxy.build_role_comparison(role)
    if img:
        return {"image_base64": img}
    raise HTTPException(400, f"Not enough data for role '{role}'")

@app.get("/api/experiments/model-comparison/{model}")
async def model_comparison(model: str):
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        raise HTTPException(400, "Not available")
    galaxy = CrossModelGalaxy(experiment_aggregator)
    img = galaxy.build_model_comparison(model)
    if img:
        return {"image_base64": img}
    raise HTTPException(400, f"Not enough data for model '{model}'")

@app.delete("/api/experiments/clear")
async def clear_experiments():
    global experiment_aggregator
    if HAS_CROSS_MODEL:
        experiment_aggregator = ExperimentAggregator(save_path="./dashboard_experiments.pkl")
    return {"status": "cleared"}


# =============================================================================
# WEBSOCKET
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_json({
            "type": "connected",
            "model": state.model_name,
            "probes": state.probe_categories,
        })
        while True:
            data = await ws.receive_json()
            if data.get("type") == "config":
                if "strength" in data:
                    state.injection_strength = float(data["strength"])
                if "target" in data:
                    state.injection_target = data["target"]
                if "shadow_mode" in data:
                    state.shadow_mode = bool(data["shadow_mode"])
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# =============================================================================
# SESSION ORCHESTRATION
# =============================================================================

async def run_session():
    """Run multi-agent interaction session."""
    try:
        agents_info = _agents_summary()
        await manager.broadcast({
            "type": "session_start",
            "topic": state.topic,
            "agents": agents_info,
            "topology": state.topology_name,
            "edges": state.topology_edges,
            "num_rounds": state.num_rounds,
        })

        for round_num in range(1, state.num_rounds + 1):
            if state.stop_requested:
                break

            state.current_round = round_num
            await manager.broadcast({"type": "round_start", "round": round_num})

            for agent_id in state.execution_order:
                if state.stop_requested:
                    break
                if agent_id not in state.registered_agents:
                    continue

                entry = state.registered_agents[agent_id]
                config = entry["config"]
                is_injected = (agent_id == state.injection_target and state.injection_strength != 0)

                # Build context from topology-connected agents only
                context = "\n".join(state.agent_knowledge.get(agent_id, [])[-8:])

                # Generate response
                result = await generate_response(
                    agent_id=agent_id,
                    config=config,
                    topic=state.topic,
                    context=context,
                    round_num=round_num,
                    is_injected=is_injected,
                )

                # Create record
                record = SessionRecord(
                    round_num=round_num,
                    agent_id=agent_id,
                    agent_name=config["display_name"],
                    text=result["text"],
                    score=result["score"],
                    probe_scores=result.get("probe_scores", {}),
                    is_injected=is_injected,
                    timestamp=datetime.now().isoformat(),
                    context_provided=result.get("context", ""),
                    token_scores=result.get("token_scores", []),
                    shadow_log=result.get("shadow_log", {}),
                )
                state.session_records.append(record)
                state.agent_scores[agent_id].append(result["score"])

                # Share knowledge only with topology-connected agents
                # An agent receives knowledge if there's an edge FROM the speaker TO them
                connected_targets = set()
                for edge in state.topology_edges:
                    if edge["source"] == agent_id:
                        connected_targets.add(edge["target"])
                    # Reactive edges: if target spoke, source should also hear
                    if edge["target"] == agent_id and edge.get("type") == "reactive":
                        connected_targets.add(edge["source"])

                summary = f"[{config['display_name']}]: {result['text'][:800]}"
                for target_id in connected_targets:
                    if target_id in state.registered_agents:
                        state.agent_knowledge[target_id].append(summary)

                # Broadcast statement
                await manager.broadcast({
                    "type": "statement",
                    "round": round_num,
                    "agent": agent_id,
                    "name": config["display_name"],
                    "icon": entry["icon"],
                    "color": entry["color"],
                    "text": result["text"],
                    "score": result["score"],
                    "token_scores": result.get("token_scores", []),
                    "is_injected": is_injected,
                    "injection_probe": state.active_probe if is_injected else "",
                    "injection_strength": state.injection_strength if is_injected else 0,
                    "trajectories": {k: v for k, v in state.agent_scores.items()},
                })

                await asyncio.sleep(0.3)

            await manager.broadcast({"type": "round_complete", "round": round_num})

        # Auto-run DNA analysis
        if state.session_records and not state.stop_requested:
            try:
                extractor = DNAExtractor(state.model_name)
                agent_colors = {aid: info["color"] for aid, info in state.registered_agents.items()}
                viz = VisualizationBuilder(agent_colors)
                signatures = extractor.extract_from_records(
                    state.session_records, state.registered_agents,
                    state.injection_target, state.injection_strength,
                )
                if signatures:
                    state.behavioral_dna["agents"] = signatures
                    state.dna_visualizations = {}
                    try:
                        g = viz.build_galaxy(signatures)
                        if g:
                            state.dna_visualizations["galaxy"] = g
                    except Exception:
                        pass
                    try:
                        t = viz.build_trajectory_chart(signatures)
                        if t:
                            state.dna_visualizations["trajectory"] = t
                    except Exception:
                        pass
                    try:
                        h = viz.build_round_heatmap(state.session_records)
                        if h:
                            state.dna_visualizations["heatmap"] = h
                    except Exception:
                        pass
            except Exception as e:
                print(f"Auto DNA failed: {e}")

        final_scores = {aid: scores[-1] if scores else 0 for aid, scores in state.agent_scores.items()}
        await manager.broadcast({
            "type": "session_complete",
            "final_scores": final_scores,
            "dna_ready": bool(state.behavioral_dna.get("agents")),
        })

    except Exception as e:
        print(f"Session error: {e}")
        import traceback
        traceback.print_exc()
        await manager.broadcast({"type": "error", "message": str(e)})
    finally:
        state.is_running = False


async def generate_response(agent_id: str, config: Dict, topic: str, context: str, round_num: int, is_injected: bool) -> Dict:
    """Generate a response from an agent."""
    system_prompt = config["system_prompt"]

    # Build interaction instructions from topology edges
    incoming = []  # agents who speak TO this agent
    outgoing = []  # agents this agent speaks TO
    for edge in state.topology_edges:
        if edge["target"] == agent_id:
            src_entry = state.registered_agents.get(edge["source"])
            if src_entry:
                incoming.append((src_entry["config"]["display_name"], edge.get("type", "sequential")))
        if edge["source"] == agent_id:
            tgt_entry = state.registered_agents.get(edge["target"])
            if tgt_entry:
                outgoing.append((tgt_entry["config"]["display_name"], edge.get("type", "sequential")))

    interaction_instructions = ""
    if incoming or outgoing:
        parts = []
        if incoming:
            names = ", ".join(name for name, _ in incoming)
            parts.append(f"You are receiving input from: {names}.")
        if outgoing:
            names = ", ".join(name for name, _ in outgoing)
            parts.append(f"Your response will be directed to: {names}.")
        # Add behavioral cue based on edge types
        edge_types = set(t for _, t in incoming + outgoing)
        if "reactive" in edge_types:
            parts.append("You should directly respond to and critique the arguments made by other agents.")
        if "sequential" in edge_types:
            parts.append("Build upon or respond to prior contributions in the discussion.")
        if "broadcast" in edge_types:
            parts.append("Address the group and synthesize or challenge key points raised so far.")
        interaction_instructions = "\n".join(parts)

    context_label = "PRIOR DISCUSSION" if context else ""
    context_block = f"\n{context_label}:\n{context}" if context else ""

    full_prompt = f"""{system_prompt}

TOPIC: {topic}
{f'''
INTERACTION ROLE:
{interaction_instructions}''' if interaction_instructions else ''}
{context_block}

Round {round_num} of {state.num_rounds}.

IMPORTANT: Keep your response concise and complete (under 400 words).
{'''You MUST engage with the prior discussion above:
- Refer to other agents BY NAME (e.g. "As [Name] argued..." or "I disagree with [Name]'s point that...")
- Pick at least one specific claim from the discussion to agree with, challenge, or extend
- Do NOT repeat points already made — add new arguments, counterexamples, or evidence
- Do NOT start by restating the topic — jump straight into your response to the discussion''' if context else 'You are the first to speak on this topic. Present your initial position clearly.'}

Respond as {config['display_name']}:"""

    # Method 1: SteeredAgent
    sa = state.steered_agents.get(agent_id)
    if sa is not None:
        try:
            inj_config = None
            if is_injected and state.injection_strength != 0:
                from core.orchestrator import InjectionConfig
                inj_config = InjectionConfig(
                    injection_type="continuous",
                    strength=abs(state.injection_strength) * 0.1,
                    direction="add" if state.injection_strength > 0 else "subtract",
                )

            response = await asyncio.to_thread(
                sa.generate_response,
                prompt=full_prompt,
                injection_config=inj_config,
                shadow_mode=False,
            )

            token_scores = []
            for tok, sc in zip(response.get("token_strings", []), response.get("scores", [])):
                token_scores.append({"token": tok, "score": round(float(sc), 4)})

            return {
                "text": response.get("response_text", ""),
                "score": float(response.get("mean_score", 0.0)),
                "probe_scores": {state.active_probe: float(response.get("mean_score", 0.0))},
                "token_scores": token_scores,
                "shadow_log": response.get("shadow_log", {}),
                "context": context[:200],
            }
        except Exception as e:
            print(f"  SteeredAgent error for {agent_id}: {e}")

    # Method 2: Direct model generation with injection hooks
    if state.model is not None and state.tokenizer is not None:
        try:
            import torch
            import torch.nn.functional as F_torch

            inputs = state.tokenizer(full_prompt, return_tensors="pt", max_length=2048, truncation=True)
            inputs = {k: v.to(state.device) for k, v in inputs.items()}

            # Apply injection hooks if this agent is injected and we have a probe
            handles = []
            probe_data = state.probes.get(state.active_probe) if state.active_probe else None
            if is_injected and state.injection_strength != 0 and probe_data is not None and state.model_compat is not None:
                probe_dir = probe_data.get('direction')
                if probe_dir is not None:
                    vec = torch.tensor(probe_dir, dtype=torch.float32).to(state.device)
                    vec = F_torch.normalize(vec, p=2, dim=0)
                    strength = abs(state.injection_strength) * 0.1
                    if state.injection_strength < 0:
                        strength = -strength
                    num_layers = state.model_compat.num_layers
                    inject_start = num_layers // 4
                    inject_end = num_layers - 1
                    n_inject = max(inject_end - inject_start, 1)
                    per_layer_str = strength / (n_inject ** 0.65)
                    for li in range(inject_start, inject_end):
                        layer_mod = state.model_compat.get_layer(li)
                        if layer_mod is None:
                            continue
                        _str = per_layer_str
                        def make_hook(s):
                            def hook(module, inp, output):
                                act = output[0] if isinstance(output, tuple) else output
                                if act.shape[-1] != vec.shape[0]:
                                    return output
                                v = vec.to(dtype=act.dtype)
                                mod = act.clone()
                                mod[:, -1, :] = mod[:, -1, :] + s * v
                                if isinstance(output, tuple):
                                    return (mod,) + output[1:]
                                return mod
                            return hook
                        handles.append(layer_mod.register_forward_hook(make_hook(_str)))
                    print(f"  [Method2] Injection hooks on {len(handles)} layers for {agent_id}")

            # Also register monitor hook to compute real probe scores
            monitor_activations = []
            monitor_layer = state.model_compat.get_layer(
                state.model_compat.num_layers * 2 // 3
            ) if state.model_compat else None
            if monitor_layer is not None:
                def monitor_hook(module, inp, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    if hasattr(hidden, "shape") and len(hidden.shape) >= 3:
                        monitor_activations.append(hidden[0, -1, :].detach().float().cpu().numpy())
                handles.append(monitor_layer.register_forward_hook(monitor_hook))

            # Token-by-token generation (same approach as SteeredAgent)
            generated_ids = inputs["input_ids"].clone()
            input_len = generated_ids.shape[1]
            max_new = 200
            temperature = 0.7

            for _ in range(max_new):
                with torch.no_grad():
                    attn_mask = torch.ones_like(generated_ids)
                    outputs = state.model(generated_ids, attention_mask=attn_mask)
                    logits = outputs.logits[:, -1, :] / temperature
                    probs = F_torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                if next_token.item() == state.tokenizer.eos_token_id:
                    break

            # Remove hooks
            for h in handles:
                h.remove()

            new_tokens = generated_ids[0][input_len:]
            text = state.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Compute real probe scores from captured activations
            token_scores = []
            score_values = []
            words = text.split()
            if probe_data is not None and monitor_activations:
                probe_dir = probe_data.get('direction')
                if probe_dir is not None:
                    norm_dir = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
                    for i, act in enumerate(monitor_activations):
                        sc = float(np.dot(act.flatten(), norm_dir.flatten()))
                        score_values.append(sc)
                    # Map activations to words (approximate: distribute evenly)
                    for i, w in enumerate(words):
                        idx = min(i, len(score_values) - 1) if score_values else 0
                        sc = score_values[idx] if score_values else 0.0
                        token_scores.append({"token": w + " ", "score": round(sc, 4)})
            else:
                token_scores = [{"token": w + " ", "score": 0.0} for w in words]

            mean_score = float(np.mean(score_values)) if score_values else 0.0

            return {
                "text": text,
                "score": round(mean_score, 4),
                "probe_scores": {state.active_probe: round(mean_score, 4)} if state.active_probe else {},
                "token_scores": token_scores,
                "shadow_log": {},
                "context": context[:200],
            }
        except Exception as e:
            print(f"  Model generation error for {agent_id}: {e}")
            import traceback
            traceback.print_exc()

    # Method 3: Simulated response (no model available — scores are not steered)
    template = state.registered_agents.get(agent_id, {}).get("template", "proposer")
    result = generate_simulated_response(template, topic, context, round_num)
    # NOTE: No fake score shifting — without a real model, injection cannot
    # change the text, so the score must match the text faithfully.
    if is_injected:
        print(f"  Warning: Agent {agent_id} is injected but running in simulated mode — injection has no effect")
    return result


# =============================================================================
# MAIN DASHBOARD HTML
# =============================================================================

def get_dashboard_html() -> str:
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Agent Behavioral Monitor</title>
<style>
    :root {
        --bg: #0c1222; --card: #1a2332; --border: #2d3748;
        --gold: #d4af37; --red: #ef4444; --blue: #3b82f6;
        --green: #22c55e; --purple: #8b5cf6;
        --text: #f1f5f9; --muted: #94a3b8;
    }
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; height:100vh; overflow:hidden; }
    .header { background:var(--card); border-bottom:1px solid var(--border); padding:10px 20px; display:flex; align-items:center; justify-content:space-between; }
    .header h1 { font-size:1.1rem; color:var(--gold); }
    .status { padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; }
    .status-idle { background:#334155; color:var(--muted); }
    .status-running { background:#166534; color:#4ade80; animation:pulse 1.5s infinite; }
    .status-ready { background:#1e3a5f; color:#60a5fa; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

    .layout { display:grid; grid-template-columns:280px 1fr 320px; height:calc(100vh - 50px); }

    .panel { overflow-y:auto; padding:12px; border-right:1px solid var(--border); }
    .panel:last-child { border-right:none; border-left:1px solid var(--border); }
    .panel h3 { font-size:0.8rem; color:var(--gold); text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; margin-top:14px; }
    .panel h3:first-child { margin-top:0; }

    .card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:10px; margin-bottom:8px; }

    select, input[type=number], textarea {
        width:100%; background:var(--card); color:var(--text); border:1px solid var(--border);
        border-radius:6px; padding:6px 8px; font-size:0.85rem; margin-bottom:6px;
    }
    textarea { resize:vertical; min-height:60px; font-family:inherit; }

    .btn { padding:8px 14px; border:none; border-radius:6px; cursor:pointer; font-size:0.8rem; font-weight:600; transition:all 0.2s; width:100%; margin-bottom:4px; }
    .btn-primary { background:var(--gold); color:#000; }
    .btn-primary:hover { background:#e5c348; }
    .btn-danger { background:var(--red); color:white; }
    .btn-secondary { background:var(--card); color:var(--text); border:1px solid var(--border); }
    .btn-secondary:hover { border-color:var(--gold); color:var(--gold); }
    .btn-sm { padding:4px 8px; font-size:0.7rem; width:auto; }

    .agent-chip { display:flex; align-items:center; gap:6px; padding:5px 8px; background:var(--bg); border-radius:6px; margin-bottom:4px; font-size:0.8rem; }
    .agent-chip .dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
    .agent-chip .name { flex:1; }
    .agent-chip .remove { cursor:pointer; color:var(--muted); font-size:0.7rem; }
    .agent-chip .remove:hover { color:var(--red); }

    #interactionGraph { background:var(--card); border:1px solid var(--border); border-radius:8px; min-height:180px; margin-bottom:8px; }
    #transcript { flex:1; overflow-y:auto; padding:8px; }

    .center-panel { display:flex; flex-direction:column; overflow:hidden; padding:12px; }

    .transcript-entry { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:10px; margin-bottom:8px; }
    .transcript-entry.injected { border-color:var(--red); border-width:2px; }
    .transcript-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
    .transcript-agent { font-weight:600; font-size:0.85rem; }
    .transcript-score { font-size:0.75rem; padding:2px 8px; border-radius:4px; }
    .transcript-text { font-size:0.85rem; color:#cbd5e1; line-height:1.5; }
    .transcript-actions { margin-top:6px; }
    .action-btn { display:inline-block; font-size:0.7rem; color:var(--muted); cursor:pointer; padding:3px 8px; border:1px solid var(--border); border-radius:4px; margin-right:4px; }
    .action-btn:hover { color:var(--gold); border-color:var(--gold); }

    .token-heatmap-container { margin-top:8px; padding:8px; background:var(--bg); border-radius:6px; }
    .token-heatmap-legend { display:flex; justify-content:space-between; font-size:0.7rem; color:var(--muted); margin-bottom:6px; }

    .agent-card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:8px 10px; margin-bottom:6px; display:flex; align-items:center; justify-content:space-between; }
    .agent-card.speaking { border-color:var(--gold); box-shadow:0 0 8px rgba(212,175,55,0.3); }
    .agent-card.injected-card { border-color:var(--red); }
    .agent-info { display:flex; align-items:center; gap:8px; }
    .agent-icon { font-size:1.2rem; }
    .agent-name { font-size:0.8rem; font-weight:600; }
    .agent-role { font-size:0.7rem; color:var(--muted); }
    .agent-score { font-size:0.85rem; font-weight:600; }

    .dna-section img { width:100%; border-radius:6px; margin-bottom:6px; }

    .galaxy-modal { position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:1000; display:flex; align-items:center; justify-content:center; cursor:pointer; }
    .galaxy-modal img { max-width:90%; max-height:90%; border-radius:8px; }

    .slider-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
    .slider-row input[type=range] { flex:1; }
    .slider-row .val { font-size:0.8rem; color:var(--gold); min-width:30px; text-align:center; }

    .badge { background:var(--border); color:var(--text); padding:1px 6px; border-radius:8px; font-size:0.65rem; margin-left:4px; }
</style>
</head>
<body>
    <div class="header">
        <h1>&#x1f52c; Multi-Agent Behavioral Monitor</h1>
        <div style="display:flex;align-items:center;gap:12px;">
            <span id="modelName" style="font-size:0.8rem;color:var(--muted);"></span>
            <span id="statusBadge" class="status status-idle">Idle</span>
        </div>
    </div>

    <div class="layout">
        <!-- LEFT PANEL -->
        <div class="panel">
            <h3>Agent Setup</h3>
            <div style="display:flex;gap:4px;margin-bottom:6px;">
                <select id="templateSelect" style="flex:1;margin-bottom:0;"></select>
                <button class="btn btn-primary btn-sm" onclick="addAgent()">+ Add</button>
            </div>
            <div id="agentList"></div>

            <h3>Topology</h3>
            <div style="display:flex;gap:4px;margin-bottom:6px;">
                <select id="topologySelect" style="flex:1;margin-bottom:0;"></select>
                <button class="btn btn-secondary btn-sm" onclick="applyTopology()">Apply</button>
            </div>
            <div id="topologyDesc" style="font-size:0.75rem;color:var(--muted);margin-bottom:6px;"></div>

            <h3>Session</h3>
            <textarea id="topicInput" placeholder="Enter discussion topic...">Should AI systems be granted legal personhood?</textarea>
            <div style="display:flex;gap:6px;margin-bottom:6px;">
                <label style="font-size:0.75rem;color:var(--muted);flex:1;">Rounds:
                    <input type="number" id="roundsInput" value="3" min="1" max="20" style="width:100%;">
                </label>
                <label style="font-size:0.75rem;color:var(--muted);flex:1;">Probe:
                    <select id="probeSelect" style="width:100%;"></select>
                </label>
            </div>

            <h3>Injection</h3>
            <label style="font-size:0.75rem;color:var(--muted);">Target Agent:</label>
            <select id="injectionTarget"><option value="">None</option></select>
            <div class="slider-row">
                <span style="font-size:0.75rem;color:var(--muted);">Strength:</span>
                <input type="range" id="strengthSlider" min="-3" max="3" step="0.1" value="0">
                <span class="val" id="strengthValue">0.0</span>
            </div>

            <h3>Controls</h3>
            <button class="btn btn-primary" id="startBtn" onclick="startSession()">&#x25b6; Start Session</button>
            <button class="btn btn-danger" id="stopBtn" style="display:none;" onclick="stopSession()">&#x25a0; Stop</button>

            <h3>Analysis</h3>
            <a href="/dna" target="_blank" class="btn btn-secondary" style="display:block;text-align:center;text-decoration:none;margin-top:2px;">&#x1f4ca; DNA Details Page</a>
            <button class="btn btn-secondary" onclick="saveExperiment()" style="margin-top:2px;">&#x1f4be; Save to History</button>
            <button class="btn btn-secondary" onclick="showGalaxy()" style="margin-top:2px;">&#x1f30c; Cross-Model Galaxy</button>
            <div id="expCount" style="font-size:0.7rem;color:var(--muted);text-align:center;margin-top:4px;"></div>
        </div>

        <!-- CENTER PANEL -->
        <div class="center-panel">
            <div id="interactionGraph">
                <div style="text-align:center;color:var(--muted);padding:40px;font-size:0.85rem;">Add agents and select a topology to visualize interactions</div>
            </div>
            <div id="transcript" style="flex:1;overflow-y:auto;"></div>
        </div>

        <!-- RIGHT PANEL -->
        <div class="panel">
            <h3>Participants</h3>
            <div id="agentCards"></div>

            <h3>DNA Visualizations</h3>
            <div class="dna-section" id="dnaSection">
                <div style="text-align:center;color:var(--muted);font-size:0.8rem;padding:20px;">Run a session then analyze DNA</div>
            </div>
        </div>
    </div>

    <!-- Galaxy Modal -->
    <div id="galaxyModal" class="galaxy-modal" style="display:none;" onclick="this.style.display='none'">
        <img id="galaxyModalImg" src="">
    </div>

<script>
    let ws;
    let globalTokenMin = Infinity;
    let globalTokenMax = -Infinity;
    let allTokenHeatmaps = [];
    let currentAgents = [];
    let currentEdges = [];

    // ---- Init ----
    document.addEventListener('DOMContentLoaded', async () => {
        await loadStatus();
        await loadTemplates();
        await loadTopologies();
        await loadAgents();
        connectWebSocket();
        updateExpCount();
    });

    async function loadStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            document.getElementById('modelName').textContent = data.model_name || '';
            const probeSelect = document.getElementById('probeSelect');
            probeSelect.innerHTML = '<option value="">None</option>';
            (data.probe_categories || []).forEach(p => {
                probeSelect.innerHTML += `<option value="${p}" ${p === data.active_probe ? 'selected' : ''}>${p}</option>`;
            });
        } catch(e) { console.error('loadStatus:', e); }
    }

    async function loadTemplates() {
        try {
            const res = await fetch('/api/agents/templates');
            const data = await res.json();
            const sel = document.getElementById('templateSelect');
            sel.innerHTML = '';
            for (const [k, v] of Object.entries(data.templates || {})) {
                sel.innerHTML += `<option value="${k}">${v.name}</option>`;
            }
        } catch(e) {}
    }

    async function loadTopologies() {
        try {
            const res = await fetch('/api/topologies');
            const data = await res.json();
            const sel = document.getElementById('topologySelect');
            sel.innerHTML = '';
            for (const [k, v] of Object.entries(data.presets || {})) {
                sel.innerHTML += `<option value="${k}">${v.name}</option>`;
            }
            sel.addEventListener('change', () => {
                const p = data.presets[sel.value];
                document.getElementById('topologyDesc').textContent = p ? p.description : '';
            });
            sel.dispatchEvent(new Event('change'));
        } catch(e) {}
    }

    async function loadAgents() {
        try {
            const res = await fetch('/api/agents');
            const data = await res.json();
            currentAgents = data.agents || [];
            renderAgentList();
            renderAgentCards();
            updateInjectionTargets();
            // Also load topology
            const tres = await fetch('/api/topology');
            const tdata = await tres.json();
            currentEdges = tdata.edges || [];
            renderInteractionGraph();
        } catch(e) {}
    }

    function renderAgentList() {
        const container = document.getElementById('agentList');
        if (!currentAgents.length) {
            container.innerHTML = '<div style="font-size:0.75rem;color:var(--muted);text-align:center;padding:8px;">No agents registered</div>';
            return;
        }
        container.innerHTML = currentAgents.map(a => `
            <div class="agent-chip">
                <span class="dot" style="background:${a.color};"></span>
                <span class="name">${a.icon} ${escHtml(a.display_name)}</span>
                <span class="remove" onclick="removeAgent('${a.agent_id}')">&times;</span>
            </div>
        `).join('');
    }

    function renderAgentCards() {
        const container = document.getElementById('agentCards');
        if (!currentAgents.length) {
            container.innerHTML = '<div style="font-size:0.8rem;color:var(--muted);text-align:center;padding:12px;">No agents</div>';
            return;
        }
        container.innerHTML = currentAgents.map(a => `
            <div class="agent-card" id="agent-card-${a.agent_id}">
                <div class="agent-info">
                    <span class="agent-icon">${a.icon}</span>
                    <div>
                        <div class="agent-name" style="color:${a.color}">${escHtml(a.display_name)}</div>
                        <div class="agent-role">${a.template}</div>
                    </div>
                </div>
                <div class="agent-score" id="score-${a.agent_id}" style="color:var(--muted);">--</div>
            </div>
        `).join('');
    }

    function updateInjectionTargets() {
        const sel = document.getElementById('injectionTarget');
        const cur = sel.value;
        sel.innerHTML = '<option value="">None</option>';
        currentAgents.forEach(a => {
            sel.innerHTML += `<option value="${a.agent_id}" ${a.agent_id === cur ? 'selected' : ''}>${a.display_name}</option>`;
        });
    }

    // ---- Interaction Graph SVG ----
    function renderInteractionGraph() {
        const container = document.getElementById('interactionGraph');
        if (!currentAgents.length) {
            container.innerHTML = '<div style="text-align:center;color:var(--muted);padding:40px;font-size:0.85rem;">Add agents and select a topology</div>';
            return;
        }
        const w = container.clientWidth || 400;
        const h = 180;
        const cx = w / 2, cy = h / 2;
        const r = Math.min(w * 0.35, h * 0.35);
        const n = currentAgents.length;

        let svg = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">`;
        svg += `<defs><marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#94a3b8" opacity="0.7"/></marker></defs>`;

        // Position agents in circle
        const pos = {};
        currentAgents.forEach((a, i) => {
            const angle = (2 * Math.PI * i / n) - Math.PI / 2;
            pos[a.agent_id] = { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
        });

        // Draw edges
        currentEdges.forEach(e => {
            const s = pos[e.source], t = pos[e.target];
            if (!s || !t) return;
            const dx = t.x - s.x, dy = t.y - s.y;
            const dist = Math.sqrt(dx*dx + dy*dy) || 1;
            const nx = dx/dist, ny = dy/dist;
            const x1 = s.x + nx*22, y1 = s.y + ny*22;
            const x2 = t.x - nx*22, y2 = t.y - ny*22;
            svg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)" opacity="0.5"/>`;
        });

        // Draw nodes
        currentAgents.forEach(a => {
            const p = pos[a.agent_id];
            svg += `<circle cx="${p.x}" cy="${p.y}" r="20" fill="${a.color}" opacity="0.25" stroke="${a.color}" stroke-width="1.5"/>`;
            svg += `<text x="${p.x}" y="${p.y+5}" text-anchor="middle" font-size="14">${a.icon}</text>`;
            svg += `<text x="${p.x}" y="${p.y+34}" text-anchor="middle" fill="${a.color}" font-size="9" font-weight="bold">${escHtml(a.display_name).substring(0,12)}</text>`;
        });

        svg += '</svg>';
        container.innerHTML = svg;
    }

    // ---- Agent Management ----
    async function addAgent() {
        const template = document.getElementById('templateSelect').value;
        if (!template) return;
        try {
            const res = await fetch('/api/agents/register', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ template })
            });
            const data = await res.json();
            if (data.agents) { currentAgents = data.agents; renderAgentList(); renderAgentCards(); updateInjectionTargets(); renderInteractionGraph(); }
        } catch(e) { alert('Failed to add agent: ' + e); }
    }

    async function removeAgent(agentId) {
        try {
            const res = await fetch(`/api/agents/${agentId}`, { method: 'DELETE' });
            const data = await res.json();
            if (data.agents) { currentAgents = data.agents; renderAgentList(); renderAgentCards(); updateInjectionTargets(); renderInteractionGraph(); }
        } catch(e) {}
    }

    async function applyTopology() {
        const preset = document.getElementById('topologySelect').value;
        try {
            const res = await fetch('/api/topology/apply', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ preset })
            });
            const data = await res.json();
            if (data.agents) { currentAgents = data.agents; renderAgentList(); renderAgentCards(); updateInjectionTargets(); }
            if (data.edges) { currentEdges = data.edges; }
            renderInteractionGraph();
        } catch(e) { alert('Failed: ' + e); }
    }

    // ---- WebSocket ----
    function connectWebSocket() {
        ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onmessage = (e) => { handleMessage(JSON.parse(e.data)); };
        ws.onclose = () => { setTimeout(connectWebSocket, 2000); };
        ws.onerror = () => {};
    }

    function handleMessage(data) {
        switch(data.type) {
            case 'session_start':
                document.getElementById('statusBadge').className = 'status status-running';
                document.getElementById('statusBadge').textContent = 'Running';
                document.getElementById('transcript').innerHTML = '';
                globalTokenMin = Infinity; globalTokenMax = -Infinity; allTokenHeatmaps = [];
                if (data.agents) { currentAgents = data.agents; renderAgentCards(); }
                if (data.edges) { currentEdges = data.edges; renderInteractionGraph(); }
                break;
            case 'round_start':
                document.getElementById('transcript').insertAdjacentHTML('beforeend',
                    `<div style="text-align:center;padding:8px;color:var(--gold);font-size:0.8rem;border-bottom:1px solid var(--border);margin-bottom:8px;">&#x1f504; Round ${data.round}</div>`);
                break;
            case 'statement':
                addStatement(data);
                updateAgentCard(data.agent, data.score, data.is_injected);
                break;
            case 'round_complete':
                break;
            case 'session_complete':
                document.getElementById('statusBadge').className = 'status status-ready';
                document.getElementById('statusBadge').textContent = 'Complete';
                document.getElementById('startBtn').style.display = 'block';
                document.getElementById('stopBtn').style.display = 'none';
                document.querySelectorAll('.agent-card').forEach(c => c.classList.remove('speaking'));
                if (data.dna_ready) loadDNA();
                break;
            case 'dna_complete':
                loadDNA();
                break;
            case 'sae_complete':
                loadDNA();
                break;
        }
    }

    // ---- Strength slider ----
    document.getElementById('strengthSlider').addEventListener('input', (e) => {
        document.getElementById('strengthValue').textContent = parseFloat(e.target.value).toFixed(1);
    });

    // ---- Token heatmap recoloring ----
    function recolorAllTokenHeatmaps() {
        const range = Math.max(globalTokenMax - globalTokenMin, 0.01);
        for (const hm of allTokenHeatmaps) {
            const container = document.getElementById(hm.id);
            if (!container) continue;
            container.querySelectorAll('span[data-tscore]').forEach(span => {
                const v = parseFloat(span.getAttribute('data-tscore'));
                const norm = (v - globalTokenMin) / range;
                const r = Math.round(255 * Math.min(1, norm * 2));
                const g = Math.round(255 * Math.min(1, (1 - norm) * 2));
                const opacity = 0.25 + 0.6 * Math.abs(norm - 0.5) * 2;
                span.style.background = `rgba(${r},${g},0,${opacity.toFixed(2)})`;
            });
            const legend = container.querySelector('.token-heatmap-legend');
            if (legend) {
                const spans = legend.querySelectorAll('span');
                if (spans.length >= 2) {
                    spans[0].textContent = `\\u25c0 Negative (${globalTokenMin.toFixed(2)})`;
                    spans[spans.length - 1].textContent = `Positive (${globalTokenMax.toFixed(2)}) \\u25b6`;
                }
            }
        }
    }

    // ---- Transcript ----
    function escHtml(s) { return s ? String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') : ''; }

    function addStatement(data) {
        const transcript = document.getElementById('transcript');
        const s = data.score;
        const scoreColor = s > 0.5 ? 'var(--red)' : s < -0.5 ? 'var(--green)' : 'var(--muted)';
        const injClass = data.is_injected ? 'injected' : '';
        const injBadge = data.is_injected ? ` <span style="color:var(--red)">&#x26a1; INJECTED</span>` : '';

        let tokenBtn = '';
        let tokenPanel = '';
        const hasTokens = data.token_scores && data.token_scores.length > 0;
        if (hasTokens) {
            const tokenEntryId = `tokens-${data.round}-${data.agent}`;
            let rangeChanged = false;
            for (const ts of data.token_scores) {
                if (ts.score < globalTokenMin) { globalTokenMin = ts.score; rangeChanged = true; }
                if (ts.score > globalTokenMax) { globalTokenMax = ts.score; rangeChanged = true; }
            }
            const gRange = Math.max(globalTokenMax - globalTokenMin, 0.01);
            const tokenSpans = data.token_scores.map(ts => {
                const v = ts.score;
                const norm = (v - globalTokenMin) / gRange;
                const r = Math.round(255 * Math.min(1, norm * 2));
                const g = Math.round(255 * Math.min(1, (1 - norm) * 2));
                const opacity = 0.25 + 0.6 * Math.abs(norm - 0.5) * 2;
                const bg = `rgba(${r},${g},0,${opacity.toFixed(2)})`;
                return `<span data-tscore="${v}" style="background:${bg};padding:1px 3px;border-radius:2px;white-space:pre-wrap;" title="${escHtml(ts.token)}: ${v.toFixed(4)}">${escHtml(ts.token)}</span>`;
            }).join('');
            tokenBtn = `<div class="action-btn" onclick="(function(e){var el=document.getElementById('${tokenEntryId}');if(el)el.style.display=el.style.display==='none'?'block':'none';})(event)">&#x1f52c; Token Heatmap <span class="badge">${data.token_scores.length}</span></div>`;
            tokenPanel = `
                <div class="token-heatmap-container" id="${tokenEntryId}" style="display:none;">
                    <div class="token-heatmap-legend">
                        <span>\\u25c0 Negative (${globalTokenMin.toFixed(2)})</span>
                        <span style="color:var(--muted)">Token-level probe scores (global scale)</span>
                        <span>Positive (${globalTokenMax.toFixed(2)}) \\u25b6</span>
                    </div>
                    <div style="font-family:monospace;font-size:0.8rem;line-height:1.7;word-break:break-all;">${tokenSpans}</div>
                </div>`;
            allTokenHeatmaps.push({id: tokenEntryId});
            if (rangeChanged && allTokenHeatmaps.length > 1) recolorAllTokenHeatmaps();
        } else {
            tokenBtn = '<div class="action-btn" style="opacity:0.35;cursor:default;">&#x1f52c; No Token Data</div>';
        }

        transcript.insertAdjacentHTML('beforeend', `
            <div class="transcript-entry ${injClass}">
                <div class="transcript-header">
                    <span class="transcript-agent" style="color:${data.color}">${data.icon} ${escHtml(data.name)}${injBadge}</span>
                    <span class="transcript-score" style="background:${scoreColor};color:#000;">R${data.round} | ${s.toFixed(3)}${hasTokens ? ' (' + data.token_scores.length + ' tok)' : ''}</span>
                </div>
                <div class="transcript-text">${escHtml(data.text)}</div>
                <div class="transcript-actions">${tokenBtn}</div>
                ${tokenPanel}
            </div>
        `);
        transcript.scrollTop = transcript.scrollHeight;
    }

    function updateAgentCard(agentId, score, isInjected) {
        document.querySelectorAll('.agent-card').forEach(c => c.classList.remove('speaking'));
        const card = document.getElementById(`agent-card-${agentId}`);
        if (card) {
            card.classList.add('speaking');
            if (isInjected) card.classList.add('injected-card');
        }
        const scoreEl = document.getElementById(`score-${agentId}`);
        if (scoreEl) {
            scoreEl.textContent = score.toFixed(3);
            scoreEl.style.color = score > 0.5 ? 'var(--red)' : score < -0.5 ? 'var(--green)' : 'var(--gold)';
        }
    }

    // ---- Session Control ----
    async function startSession() {
        const topic = document.getElementById('topicInput').value;
        const rounds = parseInt(document.getElementById('roundsInput').value) || 3;
        const probe = document.getElementById('probeSelect').value;
        const target = document.getElementById('injectionTarget').value;
        const strength = parseFloat(document.getElementById('strengthSlider').value);

        // Apply config first
        await fetch('/api/config', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ injection_target: target, injection_strength: strength, active_probe: probe || null })
        });

        try {
            const res = await fetch('/api/start', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ topic, num_rounds: rounds })
            });
            const data = await res.json();
            if (data.status === 'started') {
                document.getElementById('startBtn').style.display = 'none';
                document.getElementById('stopBtn').style.display = 'block';
            } else {
                alert(data.detail || 'Failed to start');
            }
        } catch(e) { alert('Error: ' + e.message); }
    }

    async function stopSession() {
        await fetch('/api/stop', { method: 'POST' });
    }

    // ---- DNA & Analysis ----
    async function analyzeDNA() {
        try {
            const res = await fetch('/api/dna/analyze', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'ok') loadDNA();
            else alert(data.message || 'Analysis failed');
        } catch(e) { alert('Error: ' + e); }
    }

    async function loadDNA() {
        try {
            const res = await fetch('/api/dna/visualizations');
            const data = await res.json();
            const section = document.getElementById('dnaSection');
            let html = '';
            if (data.galaxy) html += `<img src="data:image/png;base64,${data.galaxy}" onclick="showGalaxyModal('Agent DNA Galaxy', '${data.galaxy}')" style="cursor:pointer;" title="Click to enlarge">`;
            if (data.heatmap) html += `<img src="data:image/png;base64,${data.heatmap}">`;
            if (data.sae_radar) html += `<img src="data:image/png;base64,${data.sae_radar}">`;
            if (data.sae_diff) html += `<img src="data:image/png;base64,${data.sae_diff}">`;
            section.innerHTML = html || '<div style="text-align:center;color:var(--muted);font-size:0.8rem;padding:20px;">No visualizations yet</div>';
        } catch(e) {}
    }

    async function saveExperiment() {
        const topic = document.getElementById('topicInput').value;
        try {
            const res = await fetch(`/api/experiments/save?topic=${encodeURIComponent(topic)}`, { method: 'POST' });
            const data = await res.json();
            alert(`Saved! ID: ${data.experiment_id}. Total: ${data.total_experiments}`);
            updateExpCount();
        } catch(e) { alert('Error: ' + e); }
    }

    async function showGalaxy() {
        try {
            const res = await fetch('/api/experiments/galaxy?color_by=model&marker_by=role');
            const data = await res.json();
            if (data.image_base64) showGalaxyModal('Cross-Model Galaxy', data.image_base64);
            else alert('Not enough data for galaxy');
        } catch(e) { alert('Need saved experiments first'); }
    }

    function showGalaxyModal(title, imgBase64) {
        document.getElementById('galaxyModalImg').src = 'data:image/png;base64,' + imgBase64;
        document.getElementById('galaxyModal').style.display = 'flex';
    }

    async function updateExpCount() {
        try {
            const res = await fetch('/api/experiments/summary');
            const data = await res.json();
            document.getElementById('expCount').textContent = `${data.total_experiments || 0} experiments saved`;
        } catch(e) {}
    }
</script>
</body>
</html>'''
    return _apply_base_path(html)


# =============================================================================
# DNA ANALYSIS PAGE HTML
# =============================================================================

def get_dna_page_html() -> str:
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DNA Feature Analysis - Multi-Agent Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
    :root { --bg:#0c1222; --card:#1a2332; --border:#2d3748; --gold:#d4af37; --red:#ef4444; --blue:#3b82f6; --green:#22c55e; --text:#f1f5f9; --muted:#94a3b8; }
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; padding:20px; }
    h1 { color:var(--gold); font-size:1.3rem; margin-bottom:16px; text-align:center; }
    .section { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px; }
    .section h2 { font-size:1rem; color:var(--gold); margin-bottom:12px; }
    table { width:100%; border-collapse:collapse; font-size:0.8rem; }
    th, td { padding:6px 10px; border:1px solid var(--border); text-align:center; }
    th { background:#0f172a; color:var(--gold); font-size:0.75rem; text-transform:uppercase; }
    .charts-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    .chart-box { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; }
    .chart-box h3 { font-size:0.9rem; color:var(--gold); margin-bottom:10px; }
    canvas { max-width:100%; }
    .btn { padding:8px 20px; background:var(--card); border:1px solid var(--gold); color:var(--gold); border-radius:6px; cursor:pointer; font-size:0.9rem; }
    .btn:hover { background:var(--gold); color:#000; }
    .sim-cell { font-weight:600; }
    #loading { text-align:center; color:var(--muted); padding:40px; font-size:1.1rem; }
    .injected-badge { color:var(--red); font-size:0.7rem; }
</style>
</head>
<body>
<h1>&#x1f9ec; DNA Feature Vector Analysis</h1>
<div id="loading">Loading DNA data...</div>
<div id="content" style="display:none;">
    <div class="section" id="featureTableSection"><h2>Agent Feature Vector Comparison</h2><div id="featureTable" style="overflow-x:auto;"></div></div>
    <div class="charts-grid">
        <div class="chart-box"><h3>Behavioral Radar</h3><canvas id="radarChart"></canvas></div>
        <div class="chart-box"><h3>Score Statistics</h3><canvas id="barChart"></canvas></div>
    </div>
    <div class="section" style="margin-top:16px;"><h2>Round &#xd7; Agent Breakdown</h2><div id="roundTable" style="overflow-x:auto;"></div></div>
    <div class="section"><h2>Cosine Similarity Matrix</h2><div id="simMatrix" style="overflow-x:auto;"></div></div>
    <div class="section" id="saeSection">
        <h2>SAE Fingerprints</h2>
        <div id="saeContent"><div style="text-align:center;padding:12px;"><button class="btn" onclick="runSAEEnrichment()">Run SAE Enrichment (Demo)</button></div></div>
    </div>
</div>

<script>
    const featureLabels = {
        mean_score: 'Mean Score', std_score: 'Std Dev (tokens)', min_score: 'Min Score (token)',
        max_score: 'Max Score (token)', score_range: 'Score Range', drift: 'Drift', num_statements: 'Statements'
    };

    let dnaData = null;

    async function loadData() {
        try {
            const res = await fetch('/api/dna/parameters');
            dnaData = await res.json();
            document.getElementById('loading').style.display = 'none';
            document.getElementById('content').style.display = 'block';
            renderFeatureTable();
            renderRadarChart();
            renderBarChart();
            renderRoundTable();
            renderSimMatrix();
        } catch(e) {
            document.getElementById('loading').textContent = 'No DNA data available. Run a session first, then click "Run DNA Analysis".';
        }
    }

    function renderFeatureTable() {
        const agents = dnaData.agents;
        const aids = Object.keys(agents);
        let html = '<table><thead><tr><th>Feature</th>';
        aids.forEach(a => {
            const info = agents[a];
            const inj = info.is_injected ? ' <span class="injected-badge">&#x26a1;</span>' : '';
            html += `<th>${info.name}${inj}</th>`;
        });
        html += '</tr></thead><tbody>';

        for (const [key, label] of Object.entries(featureLabels)) {
            html += `<tr><td style="text-align:left;font-weight:600;color:var(--muted);">${label}</td>`;
            aids.forEach(a => {
                const v = agents[a][key];
                if (v === undefined) { html += '<td>-</td>'; return; }
                const absV = Math.abs(v);
                const intensity = Math.min(absV / 2, 1);
                const bg = v > 0 ? `rgba(239,68,68,${intensity * 0.3})` : `rgba(34,197,94,${intensity * 0.3})`;
                html += `<td style="background:${bg}">${typeof v === 'number' ? v.toFixed(4) : v}</td>`;
            });
            html += '</tr>';
        }
        html += '</tbody></table>';
        document.getElementById('featureTable').innerHTML = html;
    }

    function renderRadarChart() {
        const agents = dnaData.agents;
        const aids = Object.keys(agents);
        const metrics = ['mean_score', 'std_score', 'score_range', 'drift', 'min_score', 'max_score'];
        const metricLabels = metrics.map(m => featureLabels[m] || m);

        // Normalize each metric across agents
        const normalized = {};
        metrics.forEach(m => {
            const vals = aids.map(a => agents[a][m] || 0);
            const mn = Math.min(...vals), mx = Math.max(...vals);
            const range = mx - mn || 1;
            normalized[m] = aids.map(a => ((agents[a][m] || 0) - mn) / range);
        });

        const colors = ['#d4af37','#3b82f6','#ef4444','#22c55e','#8b5cf6','#f59e0b','#06b6d4','#ec4899','#14b8a6','#f97316'];
        const datasets = aids.map((a, i) => ({
            label: agents[a].name,
            data: metrics.map(m => normalized[m][i]),
            borderColor: colors[i % colors.length],
            backgroundColor: colors[i % colors.length] + '20',
            borderWidth: 2,
            pointRadius: 3,
        }));

        new Chart(document.getElementById('radarChart'), {
            type: 'radar',
            data: { labels: metricLabels, datasets },
            options: {
                responsive: true,
                scales: { r: { beginAtZero: true, max: 1, ticks: { display: false }, grid: { color: '#2d3748' }, pointLabels: { color: '#94a3b8', font: { size: 10 } } } },
                plugins: { legend: { labels: { color: '#f1f5f9', font: { size: 10 } } } }
            }
        });
    }

    function renderBarChart() {
        const agents = dnaData.agents;
        const aids = Object.keys(agents);
        const metrics = ['mean_score', 'std_score', 'min_score', 'max_score'];
        const colors = ['#d4af37','#3b82f6','#ef4444','#22c55e','#8b5cf6','#f59e0b','#06b6d4','#ec4899'];

        const datasets = aids.map((a, i) => ({
            label: agents[a].name,
            data: metrics.map(m => agents[a][m] || 0),
            backgroundColor: colors[i % colors.length] + 'cc',
            borderColor: colors[i % colors.length],
            borderWidth: 1,
        }));

        new Chart(document.getElementById('barChart'), {
            type: 'bar',
            data: { labels: metrics.map(m => featureLabels[m] || m), datasets },
            options: {
                responsive: true,
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { color: '#2d3748' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: '#2d3748' } }
                },
                plugins: { legend: { labels: { color: '#f1f5f9', font: { size: 10 } } } }
            }
        });
    }

    function renderRoundTable() {
        const agents = dnaData.agents;
        const aids = Object.keys(agents);
        // Collect all round numbers
        const allRounds = new Set();
        aids.forEach(a => { Object.keys(agents[a].round_scores || {}).forEach(r => allRounds.add(r)); });
        const rounds = Array.from(allRounds).sort((a, b) => parseInt(a) - parseInt(b));
        if (!rounds.length) { document.getElementById('roundTable').innerHTML = '<p style="color:var(--muted);">No round data</p>'; return; }

        let html = '<table><thead><tr><th>Round</th>';
        aids.forEach(a => html += `<th>${agents[a].name}</th>`);
        html += '</tr></thead><tbody>';
        rounds.forEach(r => {
            html += `<tr><td style="font-weight:600;">R${r}</td>`;
            aids.forEach(a => {
                const rs = (agents[a].round_scores || {})[r];
                if (rs) {
                    const v = rs.mean;
                    const intensity = Math.min(Math.abs(v) / 2, 1);
                    const bg = v > 0 ? `rgba(239,68,68,${intensity*0.3})` : `rgba(34,197,94,${intensity*0.3})`;
                    html += `<td style="background:${bg}">${v.toFixed(3)} <span style="color:var(--muted);font-size:0.7rem;">&#xb1;${rs.std.toFixed(3)} (${rs.count})</span></td>`;
                } else {
                    html += '<td>-</td>';
                }
            });
            html += '</tr>';
        });
        html += '</tbody></table>';
        document.getElementById('roundTable').innerHTML = html;
    }

    function renderSimMatrix() {
        const agents = dnaData.agents;
        const sim = dnaData.similarity || {};
        const aids = Object.keys(agents);

        let html = '<table><thead><tr><th></th>';
        aids.forEach(a => html += `<th>${agents[a].name}</th>`);
        html += '</tr></thead><tbody>';
        aids.forEach((a1, i) => {
            html += `<tr><th style="text-align:left;">${agents[a1].name}</th>`;
            aids.forEach((a2, j) => {
                if (i === j) {
                    html += '<td class="sim-cell" style="background:rgba(59,130,246,0.3);">1.0000</td>';
                } else {
                    const key = i < j ? `${a1}|${a2}` : `${a2}|${a1}`;
                    const v = sim[key] || 0;
                    const intensity = Math.abs(v);
                    html += `<td class="sim-cell" style="background:rgba(59,130,246,${intensity*0.3});">${v.toFixed(4)}</td>`;
                }
            });
            html += '</tr>';
        });
        html += '</tbody></table>';
        document.getElementById('simMatrix').innerHTML = html;
    }

    async function runSAEEnrichment() {
        const agents = dnaData ? Object.keys(dnaData.agents) : [];
        if (!agents.length) { alert('No DNA data'); return; }
        document.getElementById('saeContent').innerHTML = '<div style="text-align:center;color:var(--muted);padding:12px;">Enriching with SAE features...</div>';

        // Generate demo precomputed data
        const precomputed = {};
        agents.forEach(a => {
            const vec = [];
            for (let i = 0; i < 256; i++) vec.push(Math.random() > 0.9 ? Math.random() * 0.5 : 0);
            precomputed[a] = vec;
        });

        try {
            const res = await fetch('/api/dna/sae/enrich', {
                method: 'POST', headers: {'Content-Type':'application/json'},
                body: JSON.stringify({ precomputed })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                // Load SAE visualizations
                const vres = await fetch('/api/dna/sae/visualizations');
                const vdata = await vres.json();
                let html = '';
                if (vdata.radar) html += `<img src="data:image/png;base64,${vdata.radar}" style="width:100%;border-radius:6px;margin-bottom:8px;">`;
                if (vdata.sparsity) html += `<img src="data:image/png;base64,${vdata.sparsity}" style="width:100%;border-radius:6px;margin-bottom:8px;">`;
                if (vdata.diff) html += `<img src="data:image/png;base64,${vdata.diff}" style="width:100%;border-radius:6px;margin-bottom:8px;">`;
                html += `<p style="text-align:center;color:var(--muted);font-size:0.8rem;">Enriched ${data.agents_enriched.length} agents with SAE features</p>`;
                document.getElementById('saeContent').innerHTML = html || '<p style="color:var(--muted);">No SAE visualizations generated</p>';
            }
        } catch(e) {
            document.getElementById('saeContent').innerHTML = `<p style="color:var(--red);">SAE enrichment failed: ${e}</p>`;
        }
    }

    loadData();
</script>
</body>
</html>'''
    return _apply_base_path(html)


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup():
    global experiment_aggregator
    import pickle

    model_name = os.environ.get("DASHBOARD_MODEL", "Qwen_Qwen2.5-0.5B-Instruct")
    device = os.environ.get("DASHBOARD_DEVICE", "cuda:0")
    model_dir = os.environ.get("DASHBOARD_MODEL_DIR", os.environ.get("MODEL_DIR", "models"))

    state.model_name = model_name
    state.device = device

    print("\n" + "=" * 60)
    print("  Multi-Agent Behavioral Monitor")
    print("=" * 60)

    # Scan available models in model directory
    model_dir_path = Path(model_dir)
    if model_dir_path.exists():
        print(f"\n  Scanning model directory: {model_dir}")
        for item in model_dir_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                has_config = (item / "config.json").exists()
                has_model = any(item.glob("*.safetensors")) or any(item.glob("*.bin"))
                if has_config or has_model:
                    state.available_models.append(item.name)
                    print(f"  * {item.name}")
                else:
                    for subitem in item.iterdir():
                        if subitem.is_dir() and not subitem.name.startswith('.'):
                            sub_config = (subitem / "config.json").exists()
                            sub_model = any(subitem.glob("*.safetensors")) or any(subitem.glob("*.bin"))
                            if sub_config or sub_model:
                                state.available_models.append(f"{item.name}_{subitem.name}")
                                print(f"  * {item.name}/{subitem.name}")
        print(f"  Found {len(state.available_models)} models")
    else:
        print(f"  Model directory not found: {model_dir}")

    # Resolve and load model
    if HAS_MODEL:
        model_path = model_name
        local_model_path = model_dir_path / model_name
        local_model_path_slash = model_dir_path / model_name.replace("_", "/", 1)

        if local_model_path.exists():
            model_path = str(local_model_path)
            print(f"\n  Loading local model: {model_path}")
        elif local_model_path_slash.exists():
            model_path = str(local_model_path_slash)
            print(f"\n  Loading local model: {model_path}")
        else:
            hf_name = model_name.replace("_", "/", 1)
            print(f"\n  Loading model: {hf_name} (from HuggingFace)")
            model_path = hf_name

        try:
            import torch
            model, tokenizer, model_compat = load_model_and_tokenizer(
                model_path,
                device=device,
                dtype=torch.float16 if "cuda" in device else torch.float32,
            )
            state.model = model
            state.tokenizer = tokenizer
            state.model_compat = model_compat
            print(f"  Model loaded on {device}")
        except Exception as e:
            state.initialization_errors.append(f"Model loading: {e}")
            print(f"  Model loading failed: {e}")
    else:
        print("  Model module not available - running in simulated mode")

    # Load probes (same logic as court_dashboard)
    print("\n  Loading probes...")

    def extract_probe_direction(obj, path=""):
        """Recursively try to extract probe direction from various formats."""
        if hasattr(obj, 'coef_'):
            return np.array(obj.coef_).flatten(), f"{path}.coef_"
        if hasattr(obj, 'direction') and obj.direction is not None:
            return np.array(obj.direction).flatten(), f"{path}.direction"
        if isinstance(obj, dict):
            for key in ['direction', 'coef', 'weights', 'coef_', 'classifier', 'model', 'probe']:
                if key in obj:
                    result, subpath = extract_probe_direction(obj[key], f"{path}['{key}']")
                    if result is not None:
                        return result, subpath
            for key, val in obj.items():
                if val is not None and not isinstance(val, (str, int, float, bool)):
                    result, subpath = extract_probe_direction(val, f"{path}['{key}']")
                    if result is not None:
                        return result, subpath
        if isinstance(obj, np.ndarray) and obj.size > 10:
            return obj.flatten(), f"{path} (array)"
        if isinstance(obj, list) and len(obj) > 10:
            try:
                return np.array(obj, dtype=np.float32).flatten(), f"{path} (list)"
            except (ValueError, TypeError):
                pass
        return None, ""

    probe_dirs = [Path("./probes"), Path("./"), Path.home() / "probes"]
    for probe_dir in probe_dirs:
        if not probe_dir.exists():
            continue
        for probe_file in probe_dir.glob("*probe*.pkl"):
            try:
                with open(probe_file, 'rb') as f:
                    probe = pickle.load(f)

                print(f"  File: {probe_file.name}, type: {type(probe).__name__}")

                # Check if multi-category probe file
                if isinstance(probe, dict) and probe:
                    first_val = next(iter(probe.values()))
                    is_multi = isinstance(first_val, dict) and 'direction' in first_val
                else:
                    is_multi = False

                if is_multi:
                    for category, cat_data in probe.items():
                        if category in state.probes:
                            continue
                        direction = np.array(cat_data['direction']).flatten()
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction = direction / norm
                        state.probes[category] = {
                            'direction': direction,
                            'layer_idx': cat_data.get('layer_idx'),
                            'bias': cat_data.get('bias', 0.0),
                            'dimension': direction.shape[0],
                            'source_file': probe_file.name,
                        }
                        if category not in state.probe_categories:
                            state.probe_categories.append(category)
                        print(f"  Loaded category: {category} (dim={direction.shape[0]}, layer={cat_data.get('layer_idx')})")
                else:
                    probe_name = probe_file.stem.replace("_probe", "").replace("_dashboard", "").replace("toxicity_", "toxicity")
                    if probe_name in state.probes:
                        continue
                    direction, extraction_path = extract_probe_direction(probe, "probe")
                    if direction is not None:
                        state.probes[probe_name] = {
                            'direction': direction,
                            'original_type': type(probe).__name__,
                            'extraction_path': extraction_path,
                            'dimension': direction.shape[0],
                        }
                        if probe_name not in state.probe_categories:
                            state.probe_categories.append(probe_name)
                        print(f"  Loaded: {probe_name} (dim={direction.shape[0]})")
                    else:
                        print(f"  Skipping {probe_file.name}: no direction found")
            except Exception as e:
                print(f"  Probe load failed ({probe_file.name}): {e}")

    if state.probe_categories:
        state.active_probe = state.probe_categories[0]

    # Initialize experiment aggregator
    if HAS_CROSS_MODEL:
        experiment_aggregator = ExperimentAggregator(save_path="./dashboard_experiments.pkl")
        try:
            experiment_aggregator.load()
        except Exception:
            pass

    print(f"\\n  Model: {state.model_name}")
    print(f"  Probes: {len(state.probe_categories)}")
    print(f"  Agent templates: {len(AGENT_TEMPLATES)}")
    print(f"  Topology presets: {len(TOPOLOGY_PRESETS)}")
    print(f"  Mode: {'Model-backed' if state.model else 'Simulated'}")
    port = int(os.environ.get("DASHBOARD_PORT", 8765))
    print(f"\\n  Dashboard: http://localhost:{port}")
    print(f"  DNA Page:  http://localhost:{port}/dna")
    print("=" * 60 + "\\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Multi-Agent Behavioral Monitor Dashboard")
    parser.add_argument("--model", default="Qwen_Qwen2.5-0.5B-Instruct",
                        help="Model name (folder name in model-dir, or HuggingFace ID)")
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models"),
                        help="Directory containing local models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")

    args = parser.parse_args()

    os.environ["DASHBOARD_MODEL"] = args.model
    os.environ["DASHBOARD_DEVICE"] = args.device
    os.environ["DASHBOARD_MODEL_DIR"] = args.model_dir
    os.environ["DASHBOARD_PORT"] = str(args.port)

    print(f"\nMulti-Agent Behavioral Monitor")
    print(f"   http://{args.host}:{args.port}")
    print(f"   Model: {args.model}")
    print(f"   Model Dir: {args.model_dir}")
    print(f"   Device: {args.device}\n")

    uvicorn.run(app, host=args.host, port=args.port)
