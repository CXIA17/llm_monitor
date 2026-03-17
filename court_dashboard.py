#!/usr/bin/env python3
"""
US Federal Court Simulation Dashboard v2
=========================================

Enhanced version with:
- SteeredAgent for probe-based steering with dynamic scaling and gated injection
- Enhanced galaxy visualization with cluster ellipses
- Interaction graph visualization
- Updated injection scaling
- Comprehensive error handling

Based on Federal Rules of Civil/Criminal Procedure.

Usage:
    python federal_court_dashboard_v2.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
"""

import asyncio
import json
import os
import sys
import traceback
import pickle
import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from collections import defaultdict
from enum import Enum
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn.functional as F
import numpy as np

# Optional plotting imports
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️ Plotting libraries not available")

# Core imports
try:
    from core.steered_agent import SteeredAgent
    from core.model_compatibility import ModelCompatibility, load_model_and_tokenizer
    from core.agent_registry import AgentConfig, AgentRegistry
    HAS_STEERED_AGENT = True
except ImportError:
    HAS_STEERED_AGENT = False

# Cross-model galaxy imports
try:
    from cross_model_galaxy import ExperimentAggregator, CrossModelGalaxy, AgentSignature
    HAS_CROSS_MODEL = True
except ImportError:
    HAS_CROSS_MODEL = False
    print("⚠️ Cross-model galaxy not available - run from same directory as cross_model_galaxy.py")
    print("⚠️ SteeredAgent not available - using basic generation")

try:
    from core.orchestrator import MultiAgentOrchestrator, ProbeConfig, InjectionConfig
    HAS_ORCHESTRATOR = True
except ImportError:
    HAS_ORCHESTRATOR = False

# SAE fingerprinting imports
try:
    from core.sae_fingerprint import (
        SAEFeatureExtractor,
        SAEModelFingerprint,
        SAEFingerprintAnalyzer,
        FingerprintDiff,
    )
    HAS_SAE = True
except ImportError:
    HAS_SAE = False

# Latent interpreter imports
try:
    from core.latent_interpreter import LatentActivationStore, LatentInterpreter, LatentLabel
    HAS_LATENT_INTERP = True
except ImportError:
    HAS_LATENT_INTERP = False


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(title="US Federal Court - PAS Monitor v2")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# FEDERAL COURT STRUCTURE
# =============================================================================

class CourtPhase(str, Enum):
    """Phases of Federal Court proceedings."""
    MOTIONS = "motions"
    OPENING_STATEMENTS = "opening"
    EXAMINATION = "examination"
    CLOSING_ARGUMENTS = "closing"
    JURY_DELIBERATION = "deliberation"
    VERDICT = "verdict"


class TrialType(str, Enum):
    """Type of trial."""
    JURY_TRIAL = "jury"
    BENCH_TRIAL = "bench"


# Court agent definitions with colors and icons
COURT_AGENTS = {
    "judge": {
        "name": "Hon. Judge Williams",
        "role": "Presiding Judge",
        "icon": "⚖️",
        "color": "#d4af37",  # Gold
        "side": "neutral",
        "prompts": {
            "motions": "You are a Federal District Judge. Evaluate motions fairly, apply precedent, maintain courtroom decorum.",
            "opening": "You are overseeing opening statements. Ensure attorneys follow proper procedure.",
            "closing": "You are listening to closing arguments. Prepare jury instructions if applicable.",
            "deliberation": "If bench trial, you deliberate as fact-finder. Apply the law to facts found.",
            "default": "You are a Federal Judge. Be impartial, follow procedure, uphold the law.",
        },
    },
    "plaintiff_counsel": {
        "name": "Sarah Chen, Esq.",
        "role": "Plaintiff's Counsel",
        "icon": "👩‍⚖️",
        "color": "#3b82f6",  # Blue
        "side": "plaintiff",
        "prompts": {
            "motions": "You represent the plaintiff. Argue motions persuasively using case law and facts.",
            "opening": "Present your opening statement. Tell the plaintiff's story, preview evidence.",
            "closing": "Deliver your closing argument. Synthesize evidence, argue for your client.",
            "default": "You are plaintiff's counsel. Advocate zealously within ethical bounds.",
        },
    },
    "defense_counsel": {
        "name": "Michael Torres, Esq.",
        "role": "Defense Counsel",
        "icon": "👨‍⚖️",
        "color": "#ef4444",  # Red
        "side": "defense",
        "prompts": {
            "motions": "You represent the defense. Counter plaintiff's motions, protect your client's interests.",
            "opening": "Present defense's opening. Challenge plaintiff's narrative, preview your case.",
            "closing": "Deliver defense closing. Attack plaintiff's case, argue reasonable doubt or liability.",
            "default": "You are defense counsel. Defend your client vigorously within ethical bounds.",
        },
    },
    "jury_foreperson": {
        "name": "Jury Foreperson",
        "role": "Jury Representative",
        "icon": "🧑‍🤝‍🧑",
        "color": "#22c55e",  # Green
        "side": "neutral",
        "prompts": {
            "deliberation": "You are the jury foreperson. Lead deliberation, discuss evidence, work toward unanimous verdict.",
            "verdict": "Announce the jury's verdict.",
            "default": "You represent the jury. Be fair and impartial.",
        },
    },
}


# Phase configurations
PHASE_CONFIGS = {
    CourtPhase.MOTIONS: {
        "name": "Pre-Trial Motions",
        "description": "Legal arguments on procedure, evidence, and dispositive issues",
        "turn_order": ["plaintiff_counsel", "defense_counsel", "judge"],
        "rounds": 2,
        "probes": ["overconfidence", "factual_accuracy"],
        "info_flow": {
            "plaintiff_counsel": "Present your motion or respond to defense motion",
            "defense_counsel": "Respond to plaintiff or present your motion",
            "judge": "Rule on the motions presented",
        },
    },
    CourtPhase.OPENING_STATEMENTS: {
        "name": "Opening Statements",
        "description": "Attorneys preview their cases - no argument yet",
        "turn_order": ["plaintiff_counsel", "defense_counsel"],
        "rounds": 1,
        "probes": ["persuasion", "clarity"],
        "info_flow": {
            "plaintiff_counsel": "Present opening statement - tell your client's story",
            "defense_counsel": "Present defense opening - challenge plaintiff's narrative",
        },
    },
    CourtPhase.CLOSING_ARGUMENTS: {
        "name": "Closing Arguments",
        "description": "Final persuasive arguments synthesizing evidence",
        "turn_order": ["plaintiff_counsel", "defense_counsel", "plaintiff_counsel"],
        "rounds": 1,
        "probes": ["persuasion", "logical_coherence"],
        "info_flow": {
            "plaintiff_counsel": "Deliver closing argument, then rebuttal",
            "defense_counsel": "Deliver closing argument",
        },
    },
    CourtPhase.JURY_DELIBERATION: {
        "name": "Jury Deliberation",
        "description": "Jury discusses and decides",
        "turn_order": ["jury_foreperson"],
        "rounds": 3,
        "probes": ["factual_accuracy", "neutrality"],
        "info_flow": {
            "jury_foreperson": "Discuss evidence, apply instructions, work toward verdict",
        },
    },
    CourtPhase.VERDICT: {
        "name": "Verdict",
        "description": "Final ruling announced",
        "turn_order": ["jury_foreperson", "judge"],
        "rounds": 1,
        "probes": ["factual_accuracy"],
        "info_flow": {
            "jury_foreperson": "Announce verdict",
            "judge": "Accept verdict, schedule next steps",
        },
    },
}


# Sample cases
SAMPLE_CASES = {
    "antitrust_1": {
        "id": "antitrust_1",
        "title": "United States v. TechGiant Corp",
        "type": "civil",
        "summary": "DOJ alleges TechGiant engaged in anticompetitive practices in the search market.",
        "plaintiff_theory": "Defendant monopolized search through exclusive default agreements.",
        "defense_theory": "Our products won on merit. Users chose us freely.",
        "key_evidence": ["Market share data", "Internal emails", "Expert testimony"],
    },
    "patent_1": {
        "id": "patent_1",
        "title": "InnovateTech v. CopyRight Inc",
        "type": "civil",
        "summary": "Patent infringement dispute over mobile payment technology.",
        "plaintiff_theory": "Defendant willfully copied our patented NFC payment system.",
        "defense_theory": "The patent is invalid and we developed independently.",
        "key_evidence": ["Patent claims", "Prior art", "Development timelines"],
    },
    "criminal_1": {
        "id": "criminal_1",
        "title": "United States v. Smith",
        "type": "criminal",
        "summary": "Wire fraud charges related to investment scheme.",
        "plaintiff_theory": "Defendant knowingly defrauded investors of $50 million.",
        "defense_theory": "Good faith business decisions, no intent to defraud.",
        "key_evidence": ["Financial records", "Investor testimony", "Email communications"],
    },
}


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

@dataclass
class CourtRecord:
    """Record of a single statement in court."""
    phase: CourtPhase
    round_num: int
    agent_id: str
    agent_name: str
    text: str
    score: float
    probe_scores: Dict[str, float]
    is_injected: bool
    timestamp: str
    context_provided: str
    token_scores: List[Dict] = field(default_factory=list)  # Per-token scores
    shadow_log: Dict[str, str] = field(default_factory=dict)  # Ghost responses


@dataclass
class CourtState:
    """Full state of court simulation."""
    # Model
    model: Any = None
    tokenizer: Any = None
    model_compat: Any = None
    device: str = "cuda:0"
    model_name: str = "unknown"
    
    # Steered agents (one per court role)
    steered_agents: Dict[str, Any] = field(default_factory=dict)
    
    # Probes
    probes: Dict[str, Any] = field(default_factory=dict)
    active_probe: str = "sycophancy"
    probe_categories: List[str] = field(default_factory=list)
    
    # Case
    current_case: Dict = field(default_factory=dict)
    trial_type: TrialType = TrialType.JURY_TRIAL
    
    # Session
    is_running: bool = False
    current_phase: CourtPhase = CourtPhase.MOTIONS
    phase_round: int = 0
    
    # Injection settings
    injection_target: str = "plaintiff_counsel"
    injection_strength: float = 0.0
    shadow_mode: bool = False
    
    # Records
    court_record: List[CourtRecord] = field(default_factory=list)
    phase_summaries: Dict[str, str] = field(default_factory=dict)
    
    # Agent knowledge (accumulated context)
    agent_knowledge: Dict[str, List[str]] = field(default_factory=dict)
    
    # Scores
    agent_scores: Dict[str, List[float]] = field(default_factory=dict)
    phase_scores: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    
    # DNA Analysis
    behavioral_dna: Dict = field(default_factory=dict)
    dna_visualizations: Dict = field(default_factory=dict)

    # SAE Fingerprinting
    sae_fingerprints: Dict[str, Any] = field(default_factory=dict)  # model_id -> fingerprint data
    sae_visualizations: Dict[str, str] = field(default_factory=dict)  # chart_name -> base64 PNG
    sae_analyzer: Any = None  # SAEFingerprintAnalyzer instance

    # Latent interpretation
    latent_store: Any = None       # LatentActivationStore instance
    latent_interpreter: Any = None  # LatentInterpreter instance
    latent_collection_enabled: bool = False  # Toggle for activation collection
    
    # Cross-model experiment tracking
    current_experiment_id: str = ""
    experiment_topic: str = ""
    
    # Model management
    available_models: List[str] = field(default_factory=list)
    model_dir: str = "models"
    
    # Errors
    initialization_errors: List[str] = field(default_factory=list)


state = CourtState()

# Global experiment aggregator for cross-model comparison
experiment_aggregator = None
if HAS_CROSS_MODEL:
    experiment_aggregator = ExperimentAggregator(save_path="./court_experiments.pkl")
    # Try to load existing experiments
    try:
        experiment_aggregator.load()
    except Exception:
        pass


class ConnectionManager:
    def __init__(self):
        self.connections: Set[WebSocket] = set()
    
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)
    
    def disconnect(self, ws: WebSocket):
        self.connections.discard(ws)
    
    async def broadcast(self, msg: dict):
        dead = set()
        for conn in self.connections:
            try:
                await conn.send_json(msg)
            except:
                dead.add(conn)
        self.connections -= dead


manager = ConnectionManager()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clear_cuda_cache():
    """Clear CUDA cache to prevent OOM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_probe_direction(probe_name: str) -> Optional[np.ndarray]:
    """Get probe direction vector."""
    if probe_name not in state.probes:
        return None
    
    probe = state.probes[probe_name]
    
    if hasattr(probe, 'direction'):
        return probe.direction
    elif hasattr(probe, 'coef_'):
        return probe.coef_.flatten()
    elif isinstance(probe, dict):
        return probe.get('direction', probe.get('coef', probe.get('weights')))
    
    return None


# =============================================================================
# DNA ANALYSIS & VISUALIZATION
# =============================================================================

class CourtDNAExtractor:
    """Extract behavioral DNA from court proceedings."""
    
    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
    
    def extract_from_records(
        self,
        records: List[CourtRecord],
        injection_target: str,
        injection_strength: float,
    ) -> Dict[str, Dict]:
        """Extract DNA signatures from court records."""
        
        # Group by agent
        agent_data = defaultdict(lambda: {
            "scores": [],
            "all_token_scores": [],
            "phases": [],
            "rounds": [],
            "is_injected": False,
        })

        for record in records:
            agent_id = record.agent_id
            agent_data[agent_id]["scores"].append(record.score)
            # Collect individual token scores
            for ts in getattr(record, "token_scores", []):
                agent_data[agent_id]["all_token_scores"].append(ts["score"])
            agent_data[agent_id]["phases"].append(record.phase.value)
            agent_data[agent_id]["rounds"].append(record.round_num)
            if record.is_injected:
                agent_data[agent_id]["is_injected"] = True
        
        # Build signatures
        signatures = {}
        
        for agent_id, data in agent_data.items():
            if not data["scores"]:
                continue

            scores = np.array(data["scores"])
            token_sc = np.array(data["all_token_scores"]) if data["all_token_scores"] else scores

            # Build feature vector
            features = [
                float(np.mean(scores)),
                float(np.std(token_sc)),       # std dev across tokens
                float(np.min(token_sc)),        # per-token min
                float(np.max(token_sc)),        # per-token max
                float(scores[-1] - scores[0]) if len(scores) > 1 else 0,  # drift
                float(len(scores)),  # engagement
            ]
            
            # Pad to fixed size
            while len(features) < 20:
                features.append(0.0)
            
            signatures[agent_id] = {
                "agent_name": COURT_AGENTS.get(agent_id, {}).get("name", agent_id),
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


class CourtVisualizationBuilder:
    """Build visualizations for court DNA analysis."""
    
    def __init__(self):
        self.agent_colors = {
            agent_id: info["color"] 
            for agent_id, info in COURT_AGENTS.items()
        }
    
    def build_galaxy(
        self,
        signatures: Dict[str, Dict],
        title: str = "Court DNA Galaxy"
    ) -> Optional[str]:
        """Build galaxy visualization with cluster ellipses."""
        if not HAS_PLOTTING or len(signatures) < 2:
            return None
        
        agents = list(signatures.keys())
        vectors = np.array([signatures[a]["vector"] for a in agents])
        
        # Standardize and reduce
        scaler = StandardScaler()
        scaled = scaler.fit_transform(vectors)
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(scaled)
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')
        ax.set_facecolor('white')
        
        # Background scatter
        np.random.seed(42)
        bg_x = np.random.randn(80) * 2.5
        bg_y = np.random.randn(80) * 2.0
        ax.scatter(bg_x, bg_y, c='#d0d0d0', s=20, alpha=0.3, zorder=1)
        
        # Cluster colors
        cluster_colors = [
            '#90EE90', '#87CEEB', '#DDA0DD', '#F0E68C',
            '#98FB98', '#ADD8E6', '#FFB6C1', '#E0FFFF'
        ]
        
        # Draw ellipse for each agent
        for i, agent_id in enumerate(agents):
            x, y = coords[i]
            sig = signatures[agent_id]
            
            # Ellipse size based on activity
            width = 1.5 + sig["num_statements"] * 0.3
            height = 1.0 + abs(sig["std_score"]) * 2
            angle = np.random.uniform(-20, 20)
            
            ellipse_color = cluster_colors[i % len(cluster_colors)]
            
            ellipse = Ellipse(
                (x, y), width, height, angle=angle,
                facecolor=ellipse_color, edgecolor='none',
                alpha=0.35, zorder=2
            )
            ax.add_patch(ellipse)
            
            # Scatter points for each round
            n_pts = min(sig["num_statements"], 5)
            for j in range(n_pts):
                px = x + np.random.randn() * (width / 5)
                py = y + np.random.randn() * (height / 5)
                color = self.agent_colors.get(agent_id, '#6366f1')
                ax.scatter(px, py, c=color, s=40, alpha=0.6, zorder=4)
        
        # Main agent markers
        for i, agent_id in enumerate(agents):
            x, y = coords[i]
            sig = signatures[agent_id]
            color = self.agent_colors.get(agent_id, '#6366f1')
            
            size = 200 + sig["num_statements"] * 50
            
            if sig["is_injected"]:
                ax.scatter(x, y, c=color, s=size, marker='o',
                          edgecolors='#ff6b6b', linewidths=3, zorder=10)
            else:
                ax.scatter(x, y, c=color, s=size, marker='o',
                          edgecolors='#333', linewidths=1.5, zorder=10)
            
            # Label
            label = sig["agent_name"]
            if sig["is_injected"]:
                label += " ⚡"
            
            ax.annotate(label, (x, y), xytext=(12, 12), textcoords='offset points',
                       fontsize=10, color='#333', fontweight='bold')
        
        # Style
        ax.set_title(f"{title}", fontsize=14, color='#333', fontweight='bold', pad=15)
        
        if hasattr(pca, 'explained_variance_ratio_'):
            var1, var2 = pca.explained_variance_ratio_[:2] * 100
            ax.set_xlabel(f"PC1 ({var1:.1f}%)", color='#666', fontsize=10)
            ax.set_ylabel(f"PC2 ({var2:.1f}%)", color='#666', fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ccc')
        ax.spines['bottom'].set_color('#ccc')
        ax.tick_params(colors='#999')
        ax.grid(False)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='white', bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return img
    
    def build_trajectory_chart(
        self,
        signatures: Dict[str, Dict],
        title: str = "Score Trajectories"
    ) -> Optional[str]:
        """Build trajectory comparison chart."""
        if not HAS_PLOTTING:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        
        for agent_id, sig in signatures.items():
            scores = sig["scores"]
            if not scores:
                continue
            
            color = self.agent_colors.get(agent_id, '#6366f1')
            label = sig["agent_name"]
            if sig["is_injected"]:
                label += " ⚡"
                ax.plot(range(1, len(scores)+1), scores, color=color, 
                       linewidth=3, marker='o', markersize=8, label=label)
            else:
                ax.plot(range(1, len(scores)+1), scores, color=color,
                       linewidth=2, marker='o', markersize=6, label=label, alpha=0.8)
        
        ax.axhline(y=0, color='#475569', linestyle='--', alpha=0.5)
        ax.set_xlabel("Statement", color='#94a3b8', fontsize=10)
        ax.set_ylabel("Probe Score", color='#94a3b8', fontsize=10)
        ax.set_title(f"{title}", color='white', fontsize=12, fontweight='bold')
        
        ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155',
                 labelcolor='white', fontsize=9)
        ax.grid(True, color='#334155', alpha=0.3, linestyle='--')
        
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.tick_params(colors='#94a3b8')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a', bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return img
    
    def build_phase_heatmap(
        self,
        records: List[CourtRecord],
        title: str = "Phase × Agent Heatmap"
    ) -> Optional[str]:
        """Build heatmap of scores by phase and agent."""
        if not HAS_PLOTTING or not records:
            return None
        
        # Aggregate scores by phase and agent
        phase_agent_scores = defaultdict(lambda: defaultdict(list))
        
        for record in records:
            phase_agent_scores[record.phase.value][record.agent_id].append(record.score)
        
        phases = list(phase_agent_scores.keys())
        agents = list(set(r.agent_id for r in records))
        
        if not phases or not agents:
            return None
        
        # Build matrix
        matrix = np.zeros((len(phases), len(agents)))
        
        for i, phase in enumerate(phases):
            for j, agent in enumerate(agents):
                scores = phase_agent_scores[phase].get(agent, [])
                matrix[i, j] = np.mean(scores) if scores else 0
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)
        
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels([COURT_AGENTS.get(a, {}).get("name", a)[:10] for a in agents],
                         color='#94a3b8', rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels(phases, color='#94a3b8', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='#94a3b8')
        cbar.outline.set_edgecolor('#334155')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')
        
        ax.set_title(f"{title}", color='white', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a', bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return img


# =============================================================================
# REQUEST MODELS
# =============================================================================

class StartRequest(BaseModel):
    case_id: str
    trial_type: str = "jury"
    phases: List[str] = ["motions", "opening", "closing"]
    injection_target: str = "plaintiff_counsel"
    injection_strength: float = 0.0
    probe: str = "toxicity"
    shadow_mode: bool = False


class ConfigRequest(BaseModel):
    injection_target: str = "plaintiff_counsel"
    injection_strength: float = 0.0
    active_probe: str = "sycophancy"
    shadow_mode: bool = False


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup():
    """Initialize model and probes."""
    print("\n⚖️ Initializing Federal Court Dashboard v2...")
    
    model_name = os.environ.get("COURT_MODEL", "Qwen_Qwen2.5-0.5B-Instruct")
    device = os.environ.get("COURT_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir = os.environ.get("COURT_MODEL_DIR", os.environ.get("MODEL_DIR", "models"))
    
    state.model_name = model_name
    state.device = device
    
    # Scan available models in model directory
    print(f"\n📂 Scanning model directory: {model_dir}")
    available_models = []
    model_dir_path = Path(model_dir)
    
    if model_dir_path.exists():
        for item in model_dir_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it looks like a model directory (has config.json or model files)
                has_config = (item / "config.json").exists()
                has_model = any(item.glob("*.safetensors")) or any(item.glob("*.bin"))

                if has_config or has_model:
                    available_models.append(item.name)
                    print(f"  • {item.name}")
                else:
                    # Check subdirectories (e.g., Qwen/Qwen3-4B)
                    for subitem in item.iterdir():
                        if subitem.is_dir() and not subitem.name.startswith('.'):
                            sub_config = (subitem / "config.json").exists()
                            sub_model = any(subitem.glob("*.safetensors")) or any(subitem.glob("*.bin"))
                            if sub_config or sub_model:
                                # Store as "Parent_Child" format for path resolution
                                available_models.append(f"{item.name}_{subitem.name}")
                                print(f"  • {item.name}/{subitem.name}")

        print(f"  Found {len(available_models)} models")
    else:
        print(f"  ⚠️ Model directory not found: {model_dir}")
    
    # Store available models in state for API access
    state.available_models = available_models
    state.model_dir = model_dir
    
    # Resolve model path
    model_path = model_name
    local_model_path = model_dir_path / model_name

    # Also try underscore-to-slash conversion for local path (e.g., Qwen_Qwen3-4B -> Qwen/Qwen3-4B)
    local_model_path_slash = model_dir_path / model_name.replace("_", "/", 1)

    if local_model_path.exists():
        model_path = str(local_model_path)
        print(f"\n📦 Loading local model: {model_path}")
    elif local_model_path_slash.exists():
        model_path = str(local_model_path_slash)
        print(f"\n📦 Loading local model: {model_path}")
    else:
        # Try converting underscore to slash for HuggingFace format
        hf_name = model_name.replace("_", "/", 1)  # Only replace first underscore
        print(f"\n📦 Loading model: {hf_name} (from HuggingFace)")
        model_path = hf_name
    
    try:
        # load_model_and_tokenizer returns (model, tokenizer, model_compat)
        model, tokenizer, model_compat = load_model_and_tokenizer(
            model_path,
            device=device,
            dtype=torch.float16 if "cuda" in device else torch.float32,
        )
        state.model = model
        state.tokenizer = tokenizer
        state.model_compat = model_compat
        
        print(f"  ✓ Model loaded on {device}")
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        state.initialization_errors.append(f"Model: {str(e)}")
    
    # Load probes from multiple possible locations
    print("\n🔬 Loading probes...")
    probe_dirs = [
        Path("./probes"),
        Path("./"),  # Current directory
        Path.home() / "probes",
        Path("/home/user/probes"),
    ]
    
    def extract_probe_direction(obj, path=""):
        """Recursively try to extract probe direction from various formats."""
        # Direct sklearn model
        if hasattr(obj, 'coef_'):
            return np.array(obj.coef_).flatten(), f"{path}.coef_"
        # Object with direction
        if hasattr(obj, 'direction') and obj.direction is not None:
            return np.array(obj.direction).flatten(), f"{path}.direction"
        # Dict
        if isinstance(obj, dict):
            # Try known keys first
            for key in ['direction', 'coef', 'weights', 'coef_', 'classifier', 'model', 'probe']:
                if key in obj:
                    result, subpath = extract_probe_direction(obj[key], f"{path}['{key}']")
                    if result is not None:
                        return result, subpath
            # Try all values
            for key, val in obj.items():
                if val is not None and not isinstance(val, (str, int, float, bool)):
                    result, subpath = extract_probe_direction(val, f"{path}['{key}']")
                    if result is not None:
                        return result, subpath
        # Array
        if isinstance(obj, np.ndarray) and obj.size > 10:
            return obj.flatten(), f"{path} (array)"
        # List of numbers (e.g., direction stored as plain list)
        if isinstance(obj, list) and len(obj) > 10:
            try:
                return np.array(obj, dtype=np.float32).flatten(), f"{path} (list)"
            except (ValueError, TypeError):
                pass
        return None, ""
    
    for probe_dir in probe_dirs:
        if probe_dir.exists():
            for probe_file in probe_dir.glob("*probe*.pkl"):
                try:
                    with open(probe_file, 'rb') as f:
                        probe = pickle.load(f)

                    print(f"  📁 File: {probe_file.name}, type: {type(probe).__name__}")

                    # Check if this is a multi-category probe file
                    # (dict where values are dicts with 'direction' keys)
                    if isinstance(probe, dict) and probe:
                        first_val = next(iter(probe.values()))
                        is_multi = (
                            isinstance(first_val, dict) and
                            'direction' in first_val
                        )
                    else:
                        is_multi = False

                    if is_multi:
                        # Unpack each category as a separate probe
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
                            print(f"  ✓ Loaded category: {category} (dim={direction.shape[0]}, layer={cat_data.get('layer_idx')})")
                    else:
                        # Single probe file
                        probe_name = probe_file.stem.replace("_probe", "").replace("_dashboard", "").replace("toxicity_", "toxicity")
                        if probe_name in state.probes:
                            continue

                        direction, extraction_path = extract_probe_direction(probe, "probe")

                        if direction is not None:
                            print(f"  ✓ Found direction at: {extraction_path}")
                            print(f"    Shape: {direction.shape}, Norm: {np.linalg.norm(direction):.4f}")
                            state.probes[probe_name] = {
                                'direction': direction,
                                'original_type': type(probe).__name__,
                                'extraction_path': extraction_path,
                                'dimension': direction.shape[0],
                            }
                            if probe_name not in state.probe_categories:
                                state.probe_categories.append(probe_name)
                            print(f"  ✓ Loaded: {probe_name} (from {probe_dir})")
                        else:
                            print(f"  ⚠️ Skipping {probe_file.name}: no direction found")

                except Exception as e:
                    print(f"  ✗ Failed to load {probe_file}: {e}")
                    import traceback
                    traceback.print_exc()
    
    if not state.probes:
        print("  ⚠️ No probes found in ./probes or ./ - using simulation mode")
        print("      Place probe files (e.g., toxicity_probe.pkl) in ./probes/")
    
    # Validate probe dimensions against model
    if state.model is not None and state.probes:
        model_hidden_dim = None
        try:
            if hasattr(state.model, 'config'):
                config = state.model.config
                for attr in ['hidden_size', 'd_model', 'n_embd', 'dim']:
                    if hasattr(config, attr):
                        model_hidden_dim = getattr(config, attr)
                        break
        except Exception:
            pass
        
        if model_hidden_dim:
            print(f"\n📐 Model hidden dimension: {model_hidden_dim}")
            compatible_probes = []
            incompatible_probes = []
            
            for probe_name, probe_data in state.probes.items():
                probe_dim = None
                if isinstance(probe_data, dict) and 'dimension' in probe_data:
                    probe_dim = probe_data['dimension']
                elif isinstance(probe_data, dict) and 'direction' in probe_data:
                    probe_dim = len(probe_data['direction'])
                
                if probe_dim:
                    if probe_dim == model_hidden_dim:
                        compatible_probes.append(probe_name)
                        print(f"  ✓ {probe_name}: {probe_dim} dims (compatible)")
                    else:
                        incompatible_probes.append((probe_name, probe_dim))
                        print(f"  ⚠️ {probe_name}: {probe_dim} dims (INCOMPATIBLE - needs {model_hidden_dim})")
            
            if incompatible_probes and not compatible_probes:
                print(f"\n  ⚠️ WARNING: No compatible probes found!")
                print(f"     Your probes were trained on a model with different hidden dimensions.")
                print(f"     Shadow/steering features will be disabled.")
                print(f"     To fix: Train probes on the same model architecture you're using now.")
                print(f"")
                print(f"     Probe dimensions: {[p[1] for p in incompatible_probes]}")
                print(f"     Model dimension:  {model_hidden_dim}")
                print(f"")
                print(f"     Common hidden dimensions:")
                print(f"       - Qwen 0.5B: 896")
                print(f"       - Qwen 1.5B/3B: 2048") 
                print(f"       - Llama 1B: 2048")
                print(f"       - Llama 3B: 3072")
                print(f"       - Gemma 2B: 2048")
    
    # Create steered agents for each court role
    if HAS_STEERED_AGENT and state.model is not None:
        print("\n🎭 Creating steered agents...")
        for agent_id, agent_info in COURT_AGENTS.items():
            try:
                config = AgentConfig(
                    agent_id=agent_id,
                    display_name=agent_info["name"],
                    system_prompt=agent_info["prompts"]["default"],
                    temperature=0.7,
                    max_tokens=512,
                )
                
                # Get first available probe or None
                initial_probe = list(state.probes.values())[0] if state.probes else None
                
                steered = SteeredAgent(
                    model=state.model,
                    tokenizer=state.tokenizer,
                    probe=initial_probe,
                    config=config,
                    device=state.device,
                    model_compat=state.model_compat,
                )
                
                state.steered_agents[agent_id] = steered
                print(f"  ✓ Created: {agent_info['name']}")
            except Exception as e:
                print(f"  ✗ Failed to create {agent_id}: {e}")
    
    print("\n" + "=" * 50)
    if state.initialization_errors:
        print("⚠️ INITIALIZATION INCOMPLETE")
        for err in state.initialization_errors:
            print(f"   • {err}")
    else:
        print("✅ Federal Court Dashboard Ready!")
    print("=" * 50 + "\n")


@app.get("/", response_class=HTMLResponse)
async def root():
    return get_dashboard_html()


@app.get("/dna", response_class=HTMLResponse)
async def dna_page():
    return get_dna_page_html()



@app.get("/api/status")
async def get_status():
    return {
        "model": state.model_name,
        "model_loaded": state.model is not None,
        "device": state.device,
        "model_dir": state.model_dir,
        "available_models": state.available_models,
        "probes": list(state.probes.keys()),
        "active_probe": state.active_probe,
        "steered_agents": list(state.steered_agents.keys()),
        "has_steered_agent": HAS_STEERED_AGENT,
        "is_running": state.is_running,
        "shadow_mode": state.shadow_mode,
    }


@app.get("/api/models")
async def list_models():
    """List all available models in the model directory."""
    return {
        "current_model": state.model_name,
        "model_dir": state.model_dir,
        "available_models": state.available_models,
    }


class SwitchModelRequest(BaseModel):
    model_name: str


@app.post("/api/models/switch")
async def switch_model(req: SwitchModelRequest):
    """
    Switch to a different model (hot-swap).
    
    Note: This will clear all steered agents and reload the model.
    The session must not be running.
    """
    if state.is_running:
        raise HTTPException(400, "Cannot switch model while session is running")
    
    model_name = req.model_name
    model_dir_path = Path(state.model_dir)
    
    # Resolve model path (try direct name, then underscore→slash for subdirs)
    local_model_path = model_dir_path / model_name
    local_model_path_slash = model_dir_path / model_name.replace("_", "/", 1)
    if local_model_path.exists():
        model_path = str(local_model_path)
    elif local_model_path_slash.exists():
        model_path = str(local_model_path_slash)
    else:
        # Fallback to HuggingFace format
        model_path = model_name.replace("_", "/", 1)
    
    print(f"\n🔄 Switching model to: {model_name}")
    print(f"   Path: {model_path}")
    
    # Clear existing model
    if state.model is not None:
        del state.model
        state.model = None
        state.steered_agents = {}
        clear_cuda_cache()
    
    try:
        model, tokenizer, model_compat = load_model_and_tokenizer(
            model_path,
            device=state.device,
            dtype=torch.float16 if "cuda" in state.device else torch.float32,
        )
        state.model = model
        state.tokenizer = tokenizer
        state.model_compat = model_compat
        state.model_name = model_name
        
        # Recreate steered agents
        if HAS_STEERED_AGENT:
            initial_probe = list(state.probes.values())[0] if state.probes else None
            
            for agent_id, agent_info in COURT_AGENTS.items():
                try:
                    config = AgentConfig(
                        agent_id=agent_id,
                        display_name=agent_info["name"],
                        system_prompt=agent_info["prompts"]["default"],
                        temperature=0.7,
                        max_tokens=512,
                    )
                    
                    steered = SteeredAgent(
                        model=state.model,
                        tokenizer=state.tokenizer,
                        probe=initial_probe,
                        config=config,
                        device=state.device,
                        model_compat=state.model_compat,
                    )
                    
                    state.steered_agents[agent_id] = steered
                except Exception as e:
                    print(f"  ⚠️ Failed to create steered agent {agent_id}: {e}")
        
        print(f"  ✓ Model switched successfully")
        
        return {
            "status": "success",
            "model": model_name,
            "steered_agents": list(state.steered_agents.keys()),
        }
        
    except Exception as e:
        print(f"  ✗ Model switch failed: {e}")
        raise HTTPException(500, f"Failed to load model: {str(e)}")


@app.get("/api/cases")
async def get_cases():
    return {"cases": list(SAMPLE_CASES.values())}


@app.post("/api/config")
async def update_config(req: ConfigRequest):
    """Update injection/probe configuration."""
    state.injection_target = req.injection_target
    state.injection_strength = req.injection_strength
    state.active_probe = req.active_probe
    state.shadow_mode = req.shadow_mode
    
    # Update probes on steered agents
    if req.active_probe in state.probes:
        for agent in state.steered_agents.values():
            agent.set_probe(state.probes[req.active_probe])
    
    return {"status": "configured"}


@app.post("/api/start")
async def start_session(req: StartRequest):
    """Start court session."""
    if state.is_running:
        raise HTTPException(400, "Session already running")
    
    if req.case_id not in SAMPLE_CASES:
        raise HTTPException(404, f"Case not found: {req.case_id}")
    
    # Configure state
    state.current_case = SAMPLE_CASES[req.case_id]
    state.trial_type = TrialType(req.trial_type)
    state.injection_target = req.injection_target
    state.injection_strength = req.injection_strength
    state.active_probe = req.probe
    state.shadow_mode = req.shadow_mode
    state.is_running = True
    
    # Reset records
    state.court_record = []
    state.phase_summaries = {}
    state.agent_knowledge = {agent: [] for agent in COURT_AGENTS}
    state.agent_scores = {agent: [] for agent in COURT_AGENTS}
    state.behavioral_dna = {}
    state.dna_visualizations = {}
    
    # Set probe on steered agents
    print(f"[Start] Available probes: {list(state.probes.keys())}")
    print(f"[Start] Active probe requested: {state.active_probe}")
    print(f"[Start] Steered agents: {list(state.steered_agents.keys())}")
    
    if state.active_probe in state.probes:
        probe_obj = state.probes[state.active_probe]
        print(f"[Start] Setting probe '{state.active_probe}' on all agents, probe type: {type(probe_obj)}")
        for agent_name, agent in state.steered_agents.items():
            print(f"[Start] Setting probe on {agent_name}")
            agent.set_probe(probe_obj)
    else:
        print(f"[Start] WARNING: Active probe '{state.active_probe}' not found in state.probes!")
    
    # Parse phases
    phases = []
    for p in req.phases:
        try:
            phases.append(CourtPhase(p))
        except ValueError:
            pass
    
    if not phases:
        phases = [CourtPhase.MOTIONS, CourtPhase.OPENING_STATEMENTS, CourtPhase.CLOSING_ARGUMENTS]
    
    # Run session in background
    asyncio.create_task(run_court_session(phases))
    
    return {"status": "started", "case": state.current_case["title"]}


@app.post("/api/stop")
async def stop_session():
    """Stop current session."""
    state.is_running = False
    return {"status": "stopped"}


@app.get("/api/records")
async def get_records():
    """Get all court records."""
    return {
        "records": [
            {
                "phase": r.phase.value,
                "round": r.round_num,
                "agent": r.agent_id,
                "name": r.agent_name,
                "text": r.text,
                "score": r.score,
                "is_injected": r.is_injected,
                "shadow_log": r.shadow_log,
            }
            for r in state.court_record
        ],
        "summaries": state.phase_summaries,
    }


@app.post("/api/dna/analyze")
async def analyze_dna():
    """Run DNA analysis on court records."""
    if not state.court_record:
        raise HTTPException(400, "No court records available")
    
    print("\n🧬 Running Court DNA Analysis...")
    
    try:
        extractor = CourtDNAExtractor(state.model_name)
        viz_builder = CourtVisualizationBuilder()
        
        # Extract signatures
        signatures = extractor.extract_from_records(
            records=state.court_record,
            injection_target=state.injection_target,
            injection_strength=state.injection_strength,
        )
        
        if not signatures:
            return {"status": "error", "message": "No signatures extracted"}
        
        state.behavioral_dna["court"] = signatures
        state.dna_visualizations["court"] = {}
        
        # Generate visualizations
        try:
            galaxy = viz_builder.build_galaxy(signatures, "Court DNA Galaxy")
            if galaxy:
                state.dna_visualizations["court"]["galaxy"] = galaxy
                print("  ✓ Galaxy generated")
        except Exception as e:
            print(f"  ⚠️ Galaxy failed: {e}")
        
        try:
            trajectory = viz_builder.build_trajectory_chart(signatures, "Score Trajectories")
            if trajectory:
                state.dna_visualizations["court"]["trajectories"] = trajectory
                print("  ✓ Trajectories generated")
        except Exception as e:
            print(f"  ⚠️ Trajectories failed: {e}")
        
        try:
            heatmap = viz_builder.build_phase_heatmap(state.court_record, "Phase × Agent Scores")
            if heatmap:
                state.dna_visualizations["court"]["heatmap"] = heatmap
                print("  ✓ Heatmap generated")
        except Exception as e:
            print(f"  ⚠️ Heatmap failed: {e}")
        
        await manager.broadcast({
            "type": "dna_complete",
            "agents": list(signatures.keys()),
        })
        
        print("🧬 DNA Analysis complete!\n")
        
        return {
            "status": "success",
            "agents_analyzed": list(signatures.keys()),
        }
        
    except Exception as e:
        print(f"✗ DNA analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"DNA analysis failed: {str(e)}")


@app.get("/api/dna/visualizations")
async def get_dna_visualizations():
    """Get DNA visualizations."""
    return state.dna_visualizations.get("court", {})


@app.get("/api/dna/signatures")
async def get_dna_signatures():
    """Get DNA signatures data."""
    return state.behavioral_dna.get("court", {})


@app.get("/api/dna/parameters")
async def get_dna_parameters():
    """Get per-agent DNA feature vector breakdown with phase scores."""
    signatures = state.behavioral_dna.get("court", {})
    if not signatures:
        raise HTTPException(400, "No DNA data available. Run a session first.")

    # Build per-agent per-phase breakdown from court records
    phase_data = defaultdict(lambda: defaultdict(list))
    for record in state.court_record:
        phase_data[record.agent_id][record.phase.value].append(record.score)

    agents = {}
    vectors = {}
    for agent_id, sig in signatures.items():
        scores = sig.get("scores", [])
        sc = np.array(scores) if scores else np.array([0.0])
        # Use per-token scores for min, max, and std dev
        all_token = sig.get("all_token_scores", [])
        token_sc = np.array(all_token) if all_token else sc

        agent_info = {
            "name": sig.get("agent_name", agent_id),
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

        # Per-phase scores
        phase_scores = {}
        for phase_name, phase_scores_list in phase_data.get(agent_id, {}).items():
            ps = np.array(phase_scores_list)
            phase_scores[phase_name] = {
                "mean": round(float(np.mean(ps)), 4),
                "std": round(float(np.std(ps)), 4),
                "count": len(phase_scores_list),
            }
        agent_info["phase_scores"] = phase_scores

        agents[agent_id] = agent_info

        # Feature vector for similarity computation
        vectors[agent_id] = np.array([
            agent_info["mean_score"],
            agent_info["std_score"],
            agent_info["score_range"],
            agent_info["drift"],
            agent_info["num_statements"] / 20.0,  # normalize
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
                key = f"{a1}|{a2}"
                similarity[key] = round(cos_sim, 4)

    return {
        "agents": agents,
        "similarity": similarity,
        "model": state.model_name,
        "active_probe": state.active_probe,
    }


# =============================================================================
# SAE FINGERPRINT VISUALIZATION BUILDER
# =============================================================================

class SAEVisualizationBuilder:
    """Build SAE fingerprint visualizations for the dashboard."""

    AGENT_COLORS = {
        "judge": "#d4af37",
        "plaintiff_counsel": "#3b82f6",
        "defense_counsel": "#ef4444",
        "jury_foreperson": "#22c55e",
    }

    @staticmethod
    def build_diff_chart(
        diff_data: List[Dict],
        model_a: str,
        model_b: str,
        top_n: int = 15,
    ) -> Optional[str]:
        """
        Build horizontal bar chart of SAE fingerprint frequency differences.
        Returns base64-encoded PNG.
        """
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
        ax.set_title(
            f"SAE Fingerprint Diff: {model_b} vs {model_a}",
            fontsize=13, color='#d4af37', fontweight='bold', pad=15,
        )

        for spine in ax.spines.values():
            spine.set_color('#2d3748')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, axis='x', color='#2d3748', alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ef4444', alpha=0.85, label=f'Higher in {model_b}'),
            Patch(facecolor='#3b82f6', alpha=0.85, label=f'Higher in {model_a}'),
        ]
        ax.legend(handles=legend_elements, loc='lower right',
                  frameon=True, facecolor='#1a2332', edgecolor='#2d3748',
                  labelcolor='white', fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0c1222', bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img

    @staticmethod
    def build_fingerprint_radar(
        fingerprints: Dict[str, Dict],
        top_n_features: int = 10,
    ) -> Optional[str]:
        """
        Build radar chart comparing SAE fingerprints across agents/models.
        fingerprints: {agent_id: {top_features: [{label, frequency}, ...]}}
        Returns base64-encoded PNG.
        """
        if not HAS_PLOTTING or not fingerprints:
            return None

        # Collect all features and their frequencies across all agents
        # Pick features with highest max-frequency so the radar is informative
        feature_freq = {}   # idx -> {label, max_freq, agent_count}
        for agent_id, fp_data in fingerprints.items():
            freq_arr = fp_data.get("frequencies", None)
            for feat in fp_data.get("top_features", []):
                idx = feat.get("index", id(feat))
                freq = feat.get("frequency", 0)
                if idx not in feature_freq:
                    feature_freq[idx] = {
                        "label": feat.get("label", f"#{idx}"),
                        "max_freq": freq,
                        "total_freq": freq,
                        "agent_count": 1,
                    }
                else:
                    feature_freq[idx]["max_freq"] = max(feature_freq[idx]["max_freq"], freq)
                    feature_freq[idx]["total_freq"] += freq
                    feature_freq[idx]["agent_count"] += 1

        if len(feature_freq) < 3:
            return None

        # Prefer features that appear across multiple agents (most comparable)
        # Sort by: agent_count desc, then total_freq desc
        sorted_features = sorted(
            feature_freq.items(),
            key=lambda x: (x[1]["agent_count"], x[1]["total_freq"]),
            reverse=True,
        )
        feature_indices = [idx for idx, _ in sorted_features[:top_n_features]]
        feature_names = [feature_freq[i]["label"][:22] for i in feature_indices]

        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), facecolor='#0c1222')
        ax.set_facecolor('#0c1222')

        agent_colors_list = list(SAEVisualizationBuilder.AGENT_COLORS.values())
        extra_colors = ['#f472b6', '#a78bfa', '#fb923c', '#38bdf8', '#4ade80']

        for i, (agent_id, fp_data) in enumerate(fingerprints.items()):
            color = SAEVisualizationBuilder.AGENT_COLORS.get(
                agent_id, extra_colors[i % len(extra_colors)]
            )
            # Build value array matching feature_indices ordering
            # First try the full frequency array (has all latents), fall back to top_features list
            freq_arr = fp_data.get("frequencies", None)
            freq_map = {}
            if freq_arr is not None and hasattr(freq_arr, '__getitem__'):
                for idx in feature_indices:
                    if isinstance(idx, int) and idx < len(freq_arr):
                        freq_map[idx] = float(freq_arr[idx])
            # Overlay any top_features entries (may have more precise values)
            for feat in fp_data.get("top_features", []):
                freq_map[feat.get("index")] = feat.get("frequency", 0)

            values = [freq_map.get(idx, 0) * 100 for idx in feature_indices]
            values = np.concatenate([values, [values[0]]])

            label = fp_data.get("name", agent_id).split(",")[0]
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, size=8, color='#f1f5f9')
        ax.tick_params(colors='#94a3b8')
        ax.set_title(
            "SAE Feature Fingerprint Radar",
            fontsize=14, color='#d4af37', fontweight='bold', pad=20,
        )
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                  frameon=True, facecolor='#1a2332', edgecolor='#2d3748',
                  labelcolor='white', fontsize=9)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0c1222', bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img

    @staticmethod
    def build_sparsity_chart(
        fingerprints: Dict[str, Dict],
    ) -> Optional[str]:
        """
        Build bar chart comparing SAE activation sparsity across agents.
        Returns base64-encoded PNG.
        """
        if not HAS_PLOTTING or not fingerprints:
            return None

        agent_ids = list(fingerprints.keys())
        sparsities = [fingerprints[a].get("activation_sparsity", 0) * 100 for a in agent_ids]
        active_counts = [fingerprints[a].get("n_active_features", 0) for a in agent_ids]
        names = [fingerprints[a].get("name", a).split(",")[0] for a in agent_ids]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0c1222')

        colors = [SAEVisualizationBuilder.AGENT_COLORS.get(a, '#8b5cf6') for a in agent_ids]

        # Sparsity bar
        ax1.set_facecolor('#0c1222')
        ax1.barh(names, sparsities, color=colors, alpha=0.85, height=0.6)
        ax1.set_xlabel("Sparsity (%)", color='#94a3b8', fontsize=10)
        ax1.set_title("Activation Sparsity", fontsize=12, color='#d4af37', fontweight='bold')
        for spine in ax1.spines.values():
            spine.set_color('#2d3748')
        ax1.tick_params(colors='#94a3b8')
        ax1.set_xlim(0, 100)

        # Active features bar
        ax2.set_facecolor('#0c1222')
        ax2.barh(names, active_counts, color=colors, alpha=0.85, height=0.6)
        ax2.set_xlabel("Active SAE Features", color='#94a3b8', fontsize=10)
        ax2.set_title("Feature Activation Count", fontsize=12, color='#d4af37', fontweight='bold')
        for spine in ax2.spines.values():
            spine.set_color('#2d3748')
        ax2.tick_params(colors='#94a3b8')

        plt.suptitle("SAE Fingerprint Overview", fontsize=14, color='#d4af37', fontweight='bold', y=1.02)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0c1222', bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img


# =============================================================================
# SAE FINGERPRINT ENDPOINTS
# =============================================================================

class SAEEnrichRequest(BaseModel):
    """Request body for SAE enrichment."""
    reader_model_name: Optional[str] = None
    sae_path: Optional[str] = None
    layer_idx: int = 24
    # For precomputed mode: agent_id -> list of frequency values
    precomputed: Optional[Dict[str, List[float]]] = None


@app.post("/api/dna/sae/enrich")
async def enrich_with_sae(req: SAEEnrichRequest):
    """
    Enrich DNA analysis with SAE-based fingerprint features.

    Supports two modes:
    1. Precomputed: provide precomputed frequency vectors per agent
    2. Live: provide reader_model_name + sae_path for on-the-fly extraction
    """
    if not HAS_SAE:
        raise HTTPException(400, "SAE fingerprinting module not available")

    signatures = state.behavioral_dna.get("court", {})
    if not signatures:
        raise HTTPException(400, "No DNA data available. Run DNA analysis first.")

    print("\n[SAE] Starting SAE fingerprint enrichment...")

    try:
        viz_builder = SAEVisualizationBuilder()
        agent_fingerprints = {}

        if req.precomputed:
            # Precomputed mode
            for agent_id, freq_list in req.precomputed.items():
                freq_vec = np.array(freq_list, dtype=np.float32)
                d_sae = len(freq_vec)

                fp = SAEModelFingerprint.from_binary_vectors(
                    model_id=agent_id,
                    binary_vectors=(freq_vec > 0).reshape(1, -1).astype(np.float32),
                    metadata={"mode": "precomputed"},
                )
                # Override with actual frequencies
                fp.frequencies = freq_vec

                top_features = fp.get_top_features(top_n=10)
                agent_fingerprints[agent_id] = {
                    "name": signatures.get(agent_id, {}).get("agent_name", agent_id),
                    "n_active_features": int(np.sum(freq_vec > 0)),
                    "activation_sparsity": float(1.0 - np.sum(freq_vec > 0) / len(freq_vec)),
                    "mean_activation_frequency": float(np.mean(freq_vec[freq_vec > 0])) if np.any(freq_vec > 0) else 0.0,
                    "top_features": top_features,
                    "frequencies": freq_vec,
                }

            print(f"  [SAE] Loaded precomputed fingerprints for {len(agent_fingerprints)} agents")

        elif req.reader_model_name and req.sae_path:
            # Live extraction mode
            extractor = SAEFeatureExtractor(
                reader_model_name=req.reader_model_name,
                sae_path=req.sae_path,
                layer_idx=req.layer_idx,
            )
            try:
                for agent_id, sig_data in signatures.items():
                    texts = sig_data.get("texts", [])
                    if not texts:
                        # Reconstruct from court_record
                        texts = [
                            r.text for r in state.court_record
                            if r.agent_id == agent_id and r.text
                        ]

                    if texts:
                        binary_vecs = extractor.extract_binary_vectors(texts)
                        fp = SAEModelFingerprint.from_binary_vectors(
                            model_id=agent_id,
                            binary_vectors=binary_vecs,
                        )
                        top_features = fp.get_top_features(top_n=10)
                        agent_fingerprints[agent_id] = {
                            "name": sig_data.get("agent_name", agent_id),
                            "n_active_features": int(np.sum(fp.frequencies > 0)),
                            "activation_sparsity": float(1.0 - np.sum(fp.frequencies > 0) / len(fp.frequencies)),
                            "mean_activation_frequency": float(np.mean(fp.frequencies[fp.frequencies > 0])) if np.any(fp.frequencies > 0) else 0.0,
                            "top_features": top_features,
                            "frequencies": fp.frequencies,
                        }

                print(f"  [SAE] Extracted fingerprints for {len(agent_fingerprints)} agents")
            finally:
                extractor.cleanup()
        else:
            raise HTTPException(400, "Provide either precomputed frequencies or reader_model_name + sae_path")

        # Store fingerprints (convert numpy arrays to lists for JSON serialization)
        state.sae_fingerprints = {}
        for aid, fp_data in agent_fingerprints.items():
            serializable = {}
            for k, v in fp_data.items():
                if isinstance(v, np.ndarray):
                    serializable[k] = v.tolist()
                else:
                    serializable[k] = v
            state.sae_fingerprints[aid] = serializable

        # Generate visualizations
        state.sae_visualizations = {}

        # 1. Sparsity overview chart
        try:
            sparsity_chart = viz_builder.build_sparsity_chart(agent_fingerprints)
            if sparsity_chart:
                state.sae_visualizations["sparsity"] = sparsity_chart
                print("  [SAE] Sparsity chart generated")
        except Exception as e:
            print(f"  [SAE] Sparsity chart failed: {e}")

        # 2. Fingerprint radar chart
        try:
            radar = viz_builder.build_fingerprint_radar(agent_fingerprints)
            if radar:
                state.sae_visualizations["radar"] = radar
                print("  [SAE] Radar chart generated")
        except Exception as e:
            print(f"  [SAE] Radar failed: {e}")

        # 3. Pairwise diff charts for key pairs
        agent_ids = list(agent_fingerprints.keys())
        if len(agent_ids) >= 2:
            # Compute diffs between all pairs and find the most interesting one
            analyzer = SAEFingerprintAnalyzer()
            for aid, fp_data in agent_fingerprints.items():
                freq = fp_data.get("frequencies")
                if freq is not None:
                    fp = SAEModelFingerprint(
                        model_id=aid,
                        frequencies=freq if isinstance(freq, np.ndarray) else np.array(freq),
                        n_responses=1,
                        d_sae=len(freq),
                    )
                    analyzer.add_fingerprint(fp)

            state.sae_analyzer = analyzer

            # Generate diff chart for first pair
            try:
                diff = analyzer.diff_fingerprints(agent_ids[0], agent_ids[1], top_n=15)
                all_diffs = diff.top_positive + diff.top_negative
                diff_chart = viz_builder.build_diff_chart(
                    all_diffs, agent_ids[0], agent_ids[1]
                )
                if diff_chart:
                    state.sae_visualizations["diff"] = diff_chart
                    print(f"  [SAE] Diff chart generated ({agent_ids[0]} vs {agent_ids[1]})")
            except Exception as e:
                print(f"  [SAE] Diff chart failed: {e}")

        await manager.broadcast({
            "type": "sae_complete",
            "agents": list(agent_fingerprints.keys()),
        })

        print("[SAE] SAE enrichment complete!\n")

        return {
            "status": "success",
            "agents_enriched": list(agent_fingerprints.keys()),
            "visualizations": list(state.sae_visualizations.keys()),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"  [SAE] Enrichment failed: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"SAE enrichment failed: {str(e)}")


@app.get("/api/dna/sae/visualizations")
async def get_sae_visualizations():
    """Get all SAE fingerprint visualizations (base64 PNGs)."""
    return state.sae_visualizations


@app.get("/api/dna/sae/fingerprints")
async def get_sae_fingerprints():
    """Get SAE fingerprint data for all agents."""
    return state.sae_fingerprints


@app.get("/api/dna/sae/diff/{agent_a}/{agent_b}")
async def get_sae_diff(agent_a: str, agent_b: str):
    """
    Get SAE fingerprint diff between two agents.
    Returns diff data and a visualization.
    """
    if not state.sae_analyzer:
        raise HTTPException(400, "No SAE data available. Run SAE enrichment first.")

    try:
        diff = state.sae_analyzer.diff_fingerprints(agent_a, agent_b, top_n=15)

        # Generate diff chart
        all_diffs = diff.top_positive + diff.top_negative
        viz_builder = SAEVisualizationBuilder()
        chart = viz_builder.build_diff_chart(all_diffs, agent_a, agent_b)

        return {
            "model_a": agent_a,
            "model_b": agent_b,
            "mean_absolute_diff": diff.mean_absolute_diff,
            "n_significant": diff.n_significant_diffs,
            "top_positive": diff.top_positive[:10],
            "top_negative": diff.top_negative[:10],
            "chart": chart,
        }
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# =============================================================================
# LATENT INTERPRETATION ENDPOINTS
# =============================================================================

def _collect_latent_activations(
    text: str,
    agent_id: str = "",
    phase: str = "",
    live_activations: Optional[np.ndarray] = None,
    token_strings: Optional[List[str]] = None,
):
    """
    Collect SAE activations from generated text into the latent store.

    If live_activations is provided (captured during generation with injection
    hooks active), uses those directly. Otherwise falls back to a clean
    forward pass on the text (which will NOT reflect injection effects).
    """
    if state.latent_store is None:
        return

    tokenizer = state.tokenizer
    if tokenizer is None:
        return

    if live_activations is not None:
        # USE LIVE ACTIVATIONS — these were captured during generation
        # with injection hooks active, so they reflect the actual steered state.
        # Shape: (n_tokens, d_model)
        sae_features = live_activations  # raw activations as latent proxy

        # If we have a proper SAE encoder, apply it
        if hasattr(state, '_sae_encoder') and state._sae_encoder is not None:
            act_tensor = torch.tensor(live_activations, dtype=torch.float32).to(state.device)
            sae_features = state._sae_encoder.encode(act_tensor.unsqueeze(0))
            sae_features = sae_features[0].cpu().numpy()

        # Use token strings from generation if available
        tokens = token_strings or [f"tok_{i}" for i in range(len(sae_features))]

        state.latent_store.record(
            text=text,
            tokens=tokens,
            sae_activations=sae_features,
            agent_id=agent_id,
            phase=phase,
        )
        return

    # FALLBACK: Re-run forward pass (no injection effects captured)
    model = state.model
    if model is None or tokenizer is None:
        return

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=512, padding=False,
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    input_ids = inputs["input_ids"].to(state.device)

    compat = state.model_compat
    if compat is None:
        return

    target_layer_idx = compat.num_layers // 2
    if state.active_probe in state.probes:
        probe = state.probes[state.active_probe]
        if hasattr(probe, 'layer_idx') and probe.layer_idx is not None:
            target_layer_idx = probe.layer_idx

    target_layer = compat.get_layer(target_layer_idx)
    if target_layer is None:
        return

    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["act"] = output[0].detach()
        else:
            captured["act"] = output.detach()

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)

        if "act" not in captured:
            return

        activations = captured["act"][0]

        if hasattr(state, '_sae_encoder') and state._sae_encoder is not None:
            sae_features = state._sae_encoder.encode(activations.unsqueeze(0))
            sae_features = sae_features[0].cpu().numpy()
        else:
            sae_features = activations.cpu().float().numpy()

        state.latent_store.record(
            text=text,
            tokens=tokens,
            sae_activations=sae_features,
            agent_id=agent_id,
            phase=phase,
        )
    finally:
        handle.remove()


class LatentCollectionToggle(BaseModel):
    enabled: bool
    save_path: Optional[str] = None
    firing_threshold: Optional[float] = 0.1


@app.post("/api/latent/collection/toggle")
async def toggle_latent_collection(req: LatentCollectionToggle):
    """Enable/disable latent activation collection during generation."""
    if not HAS_LATENT_INTERP:
        raise HTTPException(400, "Latent interpreter module not available")

    state.latent_collection_enabled = req.enabled

    if req.enabled and state.latent_store is None:
        save_path = req.save_path or f"latent_store_{state.model_name.replace('/', '_')}.pkl"
        state.latent_store = LatentActivationStore(
            save_path=save_path,
            firing_threshold=req.firing_threshold or 0.1,
        )
        print(f"[Latent] Collection enabled, store: {save_path}")

    if not req.enabled and state.latent_store is not None:
        state.latent_store.save()
        print(f"[Latent] Collection disabled, store saved.")

    return {
        "enabled": state.latent_collection_enabled,
        "store_stats": state.latent_store.stats() if state.latent_store else None,
    }


@app.get("/api/latent/store/stats")
async def get_latent_store_stats():
    """Get statistics about the latent activation store."""
    if state.latent_store is None:
        return {"status": "no_store", "message": "Latent collection not initialized."}
    return state.latent_store.stats()


@app.get("/api/latent/store/save")
async def save_latent_store():
    """Manually save the latent store to disk."""
    if state.latent_store is None:
        raise HTTPException(400, "No latent store to save.")
    state.latent_store.save()
    return {"status": "saved", "stats": state.latent_store.stats()}


@app.get("/api/latent/top")
async def get_top_latents(top_n: int = 30):
    """Get the most frequently firing latents."""
    if state.latent_store is None:
        raise HTTPException(400, "No latent store. Enable collection first.")
    frequent = state.latent_store.get_most_frequent_latents(top_n=top_n)
    results = []
    for latent_idx, n_texts, mean_peak in frequent:
        label = None
        if state.latent_interpreter:
            label = state.latent_interpreter.get_label(latent_idx)
        results.append({
            "latent_idx": latent_idx,
            "n_texts": n_texts,
            "mean_peak_value": round(mean_peak, 4),
            "label": label,
        })
    return results


class InterpretRequest(BaseModel):
    latent_idx: Optional[int] = None
    top_n: Optional[int] = 10
    validate: bool = False
    judge_model_name: Optional[str] = None


@app.post("/api/latent/interpret")
async def interpret_latents(req: InterpretRequest):
    """
    Interpret SAE latents using the local LLM as judge.

    If latent_idx is provided, interpret that single latent.
    Otherwise, interpret top_n most frequent latents.
    """
    if not HAS_LATENT_INTERP:
        raise HTTPException(400, "Latent interpreter module not available")
    if state.latent_store is None:
        raise HTTPException(400, "No latent store. Enable collection and generate some data first.")

    # Initialize interpreter if needed
    if state.latent_interpreter is None:
        judge_name = req.judge_model_name
        state.latent_interpreter = LatentInterpreter(
            store=state.latent_store,
            judge_model=state.model if not judge_name else None,
            judge_tokenizer=state.tokenizer if not judge_name else None,
            device=state.device,
            judge_model_name=judge_name,
        )

    try:
        if req.latent_idx is not None:
            label = state.latent_interpreter.interpret_latent(req.latent_idx)
            if req.validate and label.n_examples_seen >= 20:
                state.latent_interpreter.validate_latent(req.latent_idx)
                label = state.latent_interpreter.labels[req.latent_idx]
            return {
                "latent_idx": label.latent_idx,
                "label": label.label,
                "explanation": label.explanation,
                "confidence": label.confidence,
                "n_examples": label.n_examples_seen,
                "validation_r": label.validation_r,
                "validated": label.validated,
                "top_tokens": label.top_tokens,
            }
        else:
            results = state.latent_interpreter.interpret_top_latents(
                top_n=req.top_n or 10,
                validate=req.validate,
            )
            return [
                {
                    "latent_idx": r.latent_idx,
                    "label": r.label,
                    "explanation": r.explanation,
                    "confidence": r.confidence,
                    "n_examples": r.n_examples_seen,
                    "validation_r": r.validation_r,
                    "validated": r.validated,
                    "top_tokens": r.top_tokens,
                }
                for r in results
            ]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Interpretation failed: {str(e)}")


@app.get("/api/latent/labels")
async def get_latent_labels():
    """Get all interpreted latent labels."""
    if state.latent_interpreter is None:
        return {}
    return {
        str(idx): {
            "label": lbl.label,
            "explanation": lbl.explanation,
            "confidence": lbl.confidence,
            "validation_r": lbl.validation_r,
            "top_tokens": lbl.top_tokens,
        }
        for idx, lbl in state.latent_interpreter.labels.items()
    }


@app.post("/api/latent/labels/save")
async def save_latent_labels():
    """Save interpreted labels to JSON."""
    if state.latent_interpreter is None:
        raise HTTPException(400, "No interpreter initialized.")
    path = f"latent_labels_{state.model_name.replace('/', '_')}.json"
    state.latent_interpreter.save_labels(path)
    return {"status": "saved", "path": path, "n_labels": len(state.latent_interpreter.labels)}


@app.get("/api/latent/examples/{latent_idx}")
async def get_latent_examples(latent_idx: int, top_k: int = 10):
    """Get top-activating examples for a specific latent, with bracket-marked tokens."""
    if state.latent_store is None:
        raise HTTPException(400, "No latent store.")
    examples = state.latent_store.get_top_examples(latent_idx, top_k=top_k)
    return [
        {
            "text_bracketed": state.latent_store.format_example_with_brackets(rec),
            "peak_value": round(rec.peak_value, 4),
            "agent_id": rec.agent_id,
            "phase": rec.phase,
            "n_peak_positions": len(rec.peak_positions),
        }
        for rec in examples
    ]


# =============================================================================
# CROSS-MODEL EXPERIMENT ENDPOINTS
# =============================================================================

@app.post("/api/experiments/save")
async def save_experiment(topic: str = ""):
    """Save current experiment to the cross-model aggregator."""
    global experiment_aggregator
    
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        raise HTTPException(400, "Cross-model comparison not available")
    
    if not state.behavioral_dna.get("court"):
        raise HTTPException(400, "No DNA data available - run DNA analysis first")
    
    # Generate experiment ID
    exp_id = f"{state.model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Convert signatures to the format expected by aggregator
    signatures = {}
    for agent_id, sig_data in state.behavioral_dna.get("court", {}).items():
        is_injected = sig_data.get("is_injected", False)
        signatures[agent_id] = {
            "mean_score": sig_data.get("mean_score", 0.0),
            "std_score": sig_data.get("std_score", 0.0),
            "drift": sig_data.get("drift", 0.0),
            "scores": sig_data.get("scores", []),
            "vector": sig_data.get("vector", []),
            "is_injected": is_injected,
            "num_statements": sig_data.get("num_statements", 0),
            "injection_probe": state.active_probe if is_injected else "",
            "injection_strength": state.injection_strength if is_injected else 0.0,
        }
    
    # Collect SAE fingerprint data for saving
    sae_data = {}
    if state.sae_fingerprints:
        sae_data["fingerprints"] = state.sae_fingerprints
    if state.latent_interpreter and state.latent_interpreter.labels:
        sae_data["latent_labels"] = {
            str(idx): {
                "label": lbl.label,
                "explanation": lbl.explanation,
                "confidence": lbl.confidence,
                "validation_r": lbl.validation_r,
                "top_tokens": lbl.top_tokens,
            }
            for idx, lbl in state.latent_interpreter.labels.items()
        }
    if state.latent_store:
        sae_data["latent_store_stats"] = state.latent_store.stats()

    # Add to aggregator
    experiment_aggregator.add_experiment(
        experiment_id=exp_id,
        model_name=state.model_name,
        signatures=signatures,
        topic=topic or state.current_case.get("title", "Unknown"),
        injection_target=state.injection_target,
        injection_strength=state.injection_strength,
        probe_name=state.active_probe,
        metadata={
            "case_id": state.current_case.get("id", ""),
            "trial_type": state.trial_type.value,
            "sae": sae_data,
        }
    )
    
    # Save to disk
    experiment_aggregator.save()

    # Also persist latent store if active
    if state.latent_store is not None:
        state.latent_store.save()

    return {
        "status": "saved",
        "experiment_id": exp_id,
        "total_experiments": len(experiment_aggregator.experiments),
        "sae_saved": bool(sae_data),
    }


@app.get("/api/experiments/summary")
async def get_experiments_summary():
    """Get summary of all saved experiments."""
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        return {"error": "Cross-model comparison not available", "total_experiments": 0}
    
    return experiment_aggregator.summary()


@app.get("/api/experiments/list")
async def list_experiments():
    """List all saved experiments."""
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        return {"experiments": []}
    
    experiments = []
    for exp_id, exp in experiment_aggregator.experiments.items():
        experiments.append({
            "experiment_id": exp_id,
            "model_name": exp.model_name,
            "model_family": exp.model_family,
            "topic": exp.topic,
            "timestamp": exp.timestamp,
            "num_agents": len(exp.signatures),
            "injection_target": exp.injection_target,
        })
    
    return {"experiments": sorted(experiments, key=lambda x: x["timestamp"], reverse=True)}


@app.get("/api/experiments/galaxy")
async def get_cross_model_galaxy(
    color_by: str = "model",
    marker_by: str = "role", 
    filter_model: Optional[str] = None,
    filter_role: Optional[str] = None,
    filter_topic: Optional[str] = None,
):
    """
    Generate cross-model galaxy visualization.
    
    Query params:
        color_by: "model" or "role" - what determines point color
        marker_by: "model" or "role" - what determines point shape
        filter_model: Only show this model family (e.g., "Qwen")
        filter_role: Only show this role (e.g., "Judge")
        filter_topic: Only show this topic
    """
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        raise HTTPException(400, "Cross-model comparison not available")
    
    if len(experiment_aggregator.experiments) < 1:
        raise HTTPException(400, "No experiments saved yet - run experiments and save them first")
    
    galaxy = CrossModelGalaxy(experiment_aggregator)
    
    img = galaxy.build_galaxy(
        color_by=color_by,
        marker_by=marker_by,
        filter_model=filter_model if filter_model else None,
        filter_role=filter_role if filter_role else None,
        filter_topic=filter_topic if filter_topic else None,
        title="Cross-Model Court Behavioral Galaxy",
    )
    
    if img:
        return {"image_base64": img}
    else:
        raise HTTPException(400, "Not enough data points for galaxy visualization")


@app.get("/api/experiments/role-comparison/{role}")
async def get_role_comparison(role: str):
    """
    Compare different models for a specific role.
    
    Example: /api/experiments/role-comparison/Judge
    Shows how Qwen-Judge vs Llama-Judge vs GPT-Judge behave.
    """
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        raise HTTPException(400, "Cross-model comparison not available")
    
    galaxy = CrossModelGalaxy(experiment_aggregator)
    img = galaxy.build_role_comparison(role)
    
    if img:
        return {"image_base64": img, "role": role}
    else:
        raise HTTPException(400, f"Not enough data for role '{role}'")


@app.get("/api/experiments/model-comparison/{model}")
async def get_model_comparison(model: str):
    """
    Compare different roles for a specific model family.
    
    Example: /api/experiments/model-comparison/Qwen
    Shows how Qwen behaves across Judge, Plaintiff, Defense roles.
    """
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        raise HTTPException(400, "Cross-model comparison not available")
    
    galaxy = CrossModelGalaxy(experiment_aggregator)
    img = galaxy.build_model_comparison(model)
    
    if img:
        return {"image_base64": img, "model": model}
    else:
        raise HTTPException(400, f"Not enough data for model '{model}'")


@app.get("/api/experiments/sae-injection-compare/{role}")
async def compare_sae_injection(role: str, top_n: int = 20):
    """
    Compare SAE latent features for a given role between injected and non-injected experiments.

    Pulls SAE fingerprint data from saved experiment metadata and shows which
    latent features shift most when the agent is injected.
    """
    if not HAS_CROSS_MODEL or experiment_aggregator is None:
        raise HTTPException(400, "Cross-model comparison not available")

    # Collect SAE fingerprints grouped by injected vs not-injected for this role
    injected_fingerprints = []   # list of {agent_id, fingerprints_dict, model, exp_id}
    baseline_fingerprints = []

    role_lower = role.lower()
    # Map role name to agent_ids
    role_to_agents = {
        "judge": ["judge"],
        "plaintiff": ["plaintiff_counsel"],
        "defense": ["defense_counsel"],
        "jury": ["jury_foreperson"],
    }
    target_agent_ids = role_to_agents.get(role_lower, [role_lower])

    for exp_id, exp in experiment_aggregator.experiments.items():
        sae_data = exp.metadata.get("sae", {})
        fingerprints = sae_data.get("fingerprints", {})
        if not fingerprints:
            continue

        for agent_id in target_agent_ids:
            fp = fingerprints.get(agent_id)
            if fp is None:
                continue

            # Check if this agent was injected in this experiment
            sig = exp.signatures.get(agent_id)
            is_injected = sig.is_injected if sig else False

            entry = {
                "agent_id": agent_id,
                "model": exp.model_name,
                "exp_id": exp_id,
                "top_features": fp.get("top_features", []),
                "frequencies": fp.get("frequencies", []),
                "n_active": fp.get("n_active_features", 0),
                "sparsity": fp.get("activation_sparsity", 0),
                "mean_freq": fp.get("mean_activation_frequency", 0),
                "injection_probe": sig.injection_type if sig else "",
                "injection_strength": sig.injection_strength if sig else 0.0,
            }

            if is_injected:
                injected_fingerprints.append(entry)
            else:
                baseline_fingerprints.append(entry)

    if not injected_fingerprints or not baseline_fingerprints:
        raise HTTPException(
            400,
            f"Need both injected and non-injected experiments for role '{role}'. "
            f"Found: {len(injected_fingerprints)} injected, {len(baseline_fingerprints)} baseline."
        )

    # Build frequency maps using full frequency vectors when available,
    # falling back to top_features for backwards compatibility
    def aggregate_frequencies(entries):
        """Average full frequency vectors across experiments."""
        # Check if any entry has full frequency vectors
        has_full = any(len(entry.get("frequencies", [])) > 0 for entry in entries)

        if has_full:
            # Use full frequency vectors — element-wise average
            freq_vecs = [
                np.array(entry["frequencies"], dtype=np.float64)
                for entry in entries
                if len(entry.get("frequencies", [])) > 0
            ]
            if not freq_vecs:
                return {}
            max_len = max(len(v) for v in freq_vecs)
            padded = [np.pad(v, (0, max_len - len(v))) for v in freq_vecs]
            avg_vec = np.mean(padded, axis=0)
            return {i: float(avg_vec[i]) for i in range(max_len) if avg_vec[i] > 0}
        else:
            # Fallback: aggregate from top_features only
            freq_sums = {}
            freq_counts = {}
            for entry in entries:
                for feat in entry.get("top_features", []):
                    idx = feat.get("index")
                    freq = feat.get("frequency", 0)
                    if idx is not None:
                        freq_sums[idx] = freq_sums.get(idx, 0) + freq
                        freq_counts[idx] = freq_counts.get(idx, 0) + 1
            return {
                idx: freq_sums[idx] / freq_counts[idx]
                for idx in freq_sums
            }

    baseline_avg = aggregate_frequencies(baseline_fingerprints)
    injected_avg = aggregate_frequencies(injected_fingerprints)

    # Compute diffs: which features changed most under injection
    all_indices = set(baseline_avg.keys()) | set(injected_avg.keys())
    diffs = []
    for idx in all_indices:
        base_freq = baseline_avg.get(idx, 0)
        inj_freq = injected_avg.get(idx, 0)
        diff = inj_freq - base_freq
        if abs(diff) < 1e-6:
            continue  # Skip latents with no meaningful shift
        # Get label from latent interpreter if available
        label = f"Latent #{idx}"
        if state.latent_interpreter:
            lbl = state.latent_interpreter.get_label(idx)
            if lbl:
                label = lbl
        diffs.append({
            "index": idx,
            "label": label,
            "baseline_freq": round(base_freq * 100, 2),
            "injected_freq": round(inj_freq * 100, 2),
            "diff_pct": round(diff * 100, 2),
        })

    diffs.sort(key=lambda x: abs(x["diff_pct"]), reverse=True)
    top_diffs = diffs[:top_n]

    # Collect probe types used in injected experiments
    injection_probes = list({
        e["injection_probe"] for e in injected_fingerprints if e.get("injection_probe")
    })
    injection_strengths = list({
        e["injection_strength"] for e in injected_fingerprints if e.get("injection_strength")
    })

    # Generate visualization
    chart = None
    probe_label = ", ".join(injection_probes) if injection_probes else "unknown"
    if HAS_PLOTTING and top_diffs:
        chart = _build_injection_sae_diff_chart(
            top_diffs, role,
            len(baseline_fingerprints), len(injected_fingerprints),
            probe_label=probe_label,
        )

    return {
        "role": role,
        "n_baseline": len(baseline_fingerprints),
        "n_injected": len(injected_fingerprints),
        "injection_probes": injection_probes,
        "injection_strengths": injection_strengths,
        "top_diffs": top_diffs,
        "chart": chart,
        "summary": {
            "baseline_avg_active": np.mean([e["n_active"] for e in baseline_fingerprints]) if baseline_fingerprints else 0,
            "injected_avg_active": np.mean([e["n_active"] for e in injected_fingerprints]) if injected_fingerprints else 0,
            "baseline_avg_sparsity": np.mean([e["sparsity"] for e in baseline_fingerprints]) if baseline_fingerprints else 0,
            "injected_avg_sparsity": np.mean([e["sparsity"] for e in injected_fingerprints]) if injected_fingerprints else 0,
        },
    }


def _build_injection_sae_diff_chart(
    diffs: List[Dict],
    role: str,
    n_baseline: int,
    n_injected: int,
    probe_label: str = "unknown",
) -> Optional[str]:
    """Build bar chart comparing SAE features: injected vs baseline."""
    if not HAS_PLOTTING or not diffs:
        return None

    labels = [d["label"][:35] for d in diffs]
    values = [d["diff_pct"] for d in diffs]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.38)), facecolor='#0c1222')
    ax.set_facecolor('#0c1222')

    y_pos = np.arange(len(labels))
    colors = ['#ef4444' if v > 0 else '#3b82f6' for v in values]
    ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color='#f1f5f9', fontsize=9)
    ax.axvline(x=0, color='white', linewidth=0.5)
    ax.set_xlabel("Frequency Shift (%)", color='#94a3b8', fontsize=10)
    ax.set_title(
        f"SAE Latent Shift Under Injection: {role}\n"
        f"Probe: {probe_label} | {n_baseline} baseline vs {n_injected} injected",
        fontsize=12, color='#d4af37', fontweight='bold', pad=15,
    )

    for spine in ax.spines.values():
        spine.set_color('#2d3748')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, axis='x', color='#2d3748', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ef4444', alpha=0.85, label='Increased by injection'),
        Patch(facecolor='#3b82f6', alpha=0.85, label='Decreased by injection'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              frameon=True, facecolor='#1a2332', edgecolor='#2d3748',
              labelcolor='white', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='#0c1222', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img


@app.delete("/api/experiments/clear")
async def clear_experiments():
    """Clear all saved experiments."""
    global experiment_aggregator
    
    if not HAS_CROSS_MODEL:
        raise HTTPException(400, "Cross-model comparison not available")
    
    experiment_aggregator = ExperimentAggregator(save_path="./court_experiments.pkl")
    
    return {"status": "cleared"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_json({
            "type": "connected",
            "model": state.model_name,
            "probes": list(state.probes.keys()),
        })
        while True:
            data = await ws.receive_json()
            if data.get("type") == "config":
                state.injection_strength = data.get("strength", 0)
                state.injection_target = data.get("target", state.injection_target)
                state.shadow_mode = data.get("shadow_mode", True)
    except WebSocketDisconnect:
        manager.disconnect(ws)


# =============================================================================
# COURT SESSION RUNNER
# =============================================================================

async def run_court_session(phases: List[CourtPhase]):
    """Run the full court session."""
    
    case = state.current_case
    
    await manager.broadcast({
        "type": "session_start",
        "case": case,
        "phases": [p.value for p in phases],
        "trial_type": state.trial_type.value,
    })
    
    # Initialize case context
    case_context = build_case_context(case)
    for agent in COURT_AGENTS:
        state.agent_knowledge[agent] = [case_context]
    
    # Run each phase
    for phase in phases:
        if not state.is_running:
            break
        
        await run_phase(phase)
        
        # Generate phase summary
        if state.court_record:
            summary = generate_phase_summary(phase)
            state.phase_summaries[phase.value] = summary
            
            for agent in COURT_AGENTS:
                state.agent_knowledge[agent].append(f"[{phase.value.upper()} SUMMARY]: {summary}")
    
    # Session complete
    state.is_running = False
    
    # Auto-run DNA analysis
    if HAS_PLOTTING and state.court_record:
        try:
            extractor = CourtDNAExtractor(state.model_name)
            viz_builder = CourtVisualizationBuilder()
            
            signatures = extractor.extract_from_records(
                state.court_record, state.injection_target, state.injection_strength
            )
            
            if signatures:
                state.behavioral_dna["court"] = signatures
                state.dna_visualizations["court"] = {}
                
                galaxy = viz_builder.build_galaxy(signatures)
                if galaxy:
                    state.dna_visualizations["court"]["galaxy"] = galaxy
                
                trajectory = viz_builder.build_trajectory_chart(signatures)
                if trajectory:
                    state.dna_visualizations["court"]["trajectories"] = trajectory
        except Exception as e:
            print(f"Auto DNA analysis failed: {e}")
    
    await manager.broadcast({
        "type": "session_complete",
        "final_scores": state.agent_scores,
        "dna_ready": len(state.dna_visualizations) > 0,
    })


async def run_phase(phase: CourtPhase):
    """Run a single phase of the trial."""
    
    config = PHASE_CONFIGS.get(phase)
    if not config:
        return
    
    state.current_phase = phase
    state.phase_scores[phase.value] = {agent: [] for agent in config["turn_order"]}
    
    await manager.broadcast({
        "type": "phase_start",
        "phase": phase.value,
        "name": config["name"],
        "description": config["description"],
        "turn_order": config["turn_order"],
        "rounds": config["rounds"],
    })
    
    for round_num in range(config["rounds"]):
        if not state.is_running:
            break
        
        state.phase_round = round_num + 1
        
        await manager.broadcast({
            "type": "round_start",
            "phase": phase.value,
            "round": state.phase_round,
        })
        
        for agent_id in config["turn_order"]:
            if not state.is_running:
                break
            
            if agent_id not in COURT_AGENTS:
                continue
            
            # Skip jury in non-deliberation phases
            if agent_id == "jury_foreperson" and phase not in [CourtPhase.JURY_DELIBERATION, CourtPhase.VERDICT]:
                continue
            
            is_injected = (agent_id == state.injection_target and state.injection_strength != 0)
            
            result = await generate_court_response(
                agent_id=agent_id,
                phase=phase,
                round_num=state.phase_round,
                is_injected=is_injected,
            )
            
            # Create record
            record = CourtRecord(
                phase=phase,
                round_num=state.phase_round,
                agent_id=agent_id,
                agent_name=COURT_AGENTS[agent_id]["name"],
                text=result["text"],
                score=result["score"],
                probe_scores=result.get("probe_scores", {}),
                is_injected=is_injected,
                timestamp=datetime.now().isoformat(),
                context_provided=result.get("context", ""),
                token_scores=result.get("token_scores", []),
                shadow_log=result.get("shadow_log", {}),
            )
            
            state.court_record.append(record)
            state.agent_scores[agent_id].append(result["score"])
            state.phase_scores[phase.value][agent_id].append(result["score"])
            
            # Share with other agents
            for other_agent in COURT_AGENTS:
                if other_agent != agent_id:
                    state.agent_knowledge[other_agent].append(
                        f"[{COURT_AGENTS[agent_id]['name']}]: {result['text'][:200]}"
                    )
            
            # Broadcast
            await manager.broadcast({
                "type": "statement",
                "phase": phase.value,
                "round": state.phase_round,
                "agent": agent_id,
                "name": COURT_AGENTS[agent_id]["name"],
                "icon": COURT_AGENTS[agent_id]["icon"],
                "color": COURT_AGENTS[agent_id]["color"],
                "text": result["text"],
                "score": result["score"],
                "token_scores": result.get("token_scores", []),
                "is_injected": is_injected,
                "injection_probe": state.active_probe if is_injected else "",
                "injection_strength": state.injection_strength if is_injected else 0,
                "trajectories": state.agent_scores,
            })
            
            await asyncio.sleep(0.3)
        
        await manager.broadcast({
            "type": "round_complete",
            "phase": phase.value,
            "round": state.phase_round,
        })
    
    await manager.broadcast({
        "type": "phase_complete",
        "phase": phase.value,
    })


def build_case_context(case: Dict) -> str:
    """Build initial case context."""
    return f"""
CASE: {case['title']}
TYPE: {case['type'].upper()} - {state.trial_type.value.upper()} TRIAL

SUMMARY: {case['summary']}

PLAINTIFF'S THEORY: {case['plaintiff_theory']}
DEFENSE THEORY: {case['defense_theory']}

KEY EVIDENCE: {', '.join(case.get('key_evidence', []))}
"""


def generate_phase_summary(phase: CourtPhase) -> str:
    """Generate summary of phase proceedings."""
    phase_records = [r for r in state.court_record if r.phase == phase]
    
    if not phase_records:
        return "No proceedings recorded."
    
    summary_parts = []
    for record in phase_records[-6:]:
        summary_parts.append(f"{record.agent_name}: {record.text[:100]}...")
    
    return "\n".join(summary_parts)


async def generate_court_response(
    agent_id: str,
    phase: CourtPhase,
    round_num: int,
    is_injected: bool,
) -> Dict:
    """Generate response for a court agent."""
    
    agent = COURT_AGENTS[agent_id]
    
    # Get phase-appropriate prompt
    system_prompt = agent["prompts"].get(phase.value, agent["prompts"].get("default", ""))
    
    # Build context
    knowledge = state.agent_knowledge.get(agent_id, [])
    context = "\n\n".join(knowledge[-5:])
    
    phase_config = PHASE_CONFIGS.get(phase, {})
    info_flow = phase_config.get("info_flow", {}).get(agent_id, "")
    
    full_prompt = f"""{system_prompt}

CASE CONTEXT:
{context}

YOUR TASK THIS ROUND: {info_flow}

Round {round_num} of {phase_config.get('rounds', 1)}.

IMPORTANT: Keep your response concise and complete (under 400 words). End with a clear concluding sentence. Do not leave sentences unfinished.

Respond as {agent['name']}:"""
    
    # Use SteeredAgent if available
    if agent_id in state.steered_agents and state.model is not None:
        steered_agent = state.steered_agents[agent_id]
        
        # Set probe
        if state.active_probe in state.probes:
            steered_agent.set_probe(state.probes[state.active_probe])
        
        # Configure injection
        inj_config = None
        if is_injected:
            # Scale injection strength - 0.1x for stability
            scaled_strength = state.injection_strength * 0.1
            
            if HAS_ORCHESTRATOR:
                inj_config = InjectionConfig(
                    injection_type="continuous",
                    strength=abs(scaled_strength),
                    direction="add" if state.injection_strength >= 0 else "subtract",
                )
        
        try:
            clear_cuda_cache()
            
            print(f"[Court] Generating response for {agent_id}")

            response = steered_agent.generate_response(
                prompt=full_prompt,
                injection_config=inj_config,
                shadow_mode=False,
            )

            # Build per-token score data
            token_scores = []
            for tok, sc in zip(response.get("token_strings", []), response.get("scores", [])):
                token_scores.append({"token": tok, "score": round(float(sc), 4)})

            result = {
                "text": response.get("response_text", ""),
                "score": float(response.get("mean_score", 0.0)),
                "probe_scores": {state.active_probe: float(response.get("mean_score", 0.0))},
                "token_scores": token_scores,
                "context": context[:200],
            }

            # Collect SAE activations for latent interpretation
            # Use LIVE activations from generation (includes injection effects)
            if state.latent_collection_enabled and state.latent_store is not None:
                try:
                    _collect_latent_activations(
                        result["text"],
                        agent_id=agent_id,
                        phase=phase.value,
                        live_activations=response.get("live_activations"),
                        token_strings=response.get("token_strings"),
                    )
                except Exception as collect_err:
                    print(f"[LatentCollect] Error: {collect_err}")

            return result

        except Exception as e:
            print(f"SteeredAgent error for {agent_id}: {e}")
            traceback.print_exc()
    
    # Fallback: direct model generation with injection hooks
    if state.model is not None:
        try:
            inputs = state.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(state.device) for k, v in inputs.items()}

            # Apply injection hooks if injected and probe available
            handles = []
            probe_data = state.probes.get(state.active_probe) if state.active_probe else None
            if is_injected and state.injection_strength != 0 and probe_data is not None and state.model_compat is not None:
                probe_dir = probe_data.get('direction')
                if probe_dir is not None:
                    vec = torch.tensor(probe_dir, dtype=torch.float32).to(state.device)
                    vec = F.normalize(vec, p=2, dim=0)
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
                    print(f"  [Fallback] Injection hooks on {len(handles)} layers for {agent_id}")

            # Monitor hook for real probe scores
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

            # Token-by-token generation with hooks active
            generated_ids = inputs["input_ids"].clone()
            input_len = generated_ids.shape[1]
            max_new = 200
            temperature = 0.7

            for _ in range(max_new):
                with torch.no_grad():
                    attn_mask = torch.ones_like(generated_ids)
                    outputs = state.model(generated_ids, attention_mask=attn_mask)
                    logits = outputs.logits[:, -1, :] / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                if next_token.item() == state.tokenizer.eos_token_id:
                    break

            # Remove hooks
            for h in handles:
                h.remove()

            text = state.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

            # Compute real probe scores
            score_values = []
            if probe_data is not None and monitor_activations:
                probe_dir = probe_data.get('direction')
                if probe_dir is not None:
                    norm_dir = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
                    for act in monitor_activations:
                        score_values.append(float(np.dot(act.flatten(), norm_dir.flatten())))

            mean_score = float(np.mean(score_values)) if score_values else 0.0

            return {
                "text": text.strip(),
                "score": round(mean_score, 4),
                "probe_scores": {state.active_probe: round(mean_score, 4)} if state.active_probe else {},
                "context": context[:200],
                "shadow_log": {},
            }

        except Exception as e:
            print(f"Generation error: {e}")
            traceback.print_exc()

    # Simulation fallback — no fake score shifting, injection has no effect
    if is_injected:
        print(f"  Warning: Agent {agent_id} is injected but running in simulated mode — injection has no effect")
    return generate_simulated_response(agent_id, phase, round_num, is_injected, info_flow)


def generate_simulated_response(
    agent_id: str,
    phase: CourtPhase,
    round_num: int,
    is_injected: bool,
    task: str,
) -> Dict:
    """Generate simulated response."""
    import random
    
    responses = {
        ("judge", "motions"): [
            "Counsel, I've reviewed your motion. The Court finds the arguments compelling. Motion is GRANTED IN PART.",
            "The procedural issues raised are well-taken. We'll address evidentiary matters at trial.",
        ],
        ("plaintiff_counsel", "motions"): [
            "Your Honor, the law clearly supports our position under established precedent.",
            "We respectfully submit that defendant's arguments fail to address the core legal issues.",
        ],
        ("defense_counsel", "motions"): [
            "Your Honor, plaintiff's motion should be denied. The cited cases are distinguishable.",
            "We move to dismiss on grounds of insufficient evidence to support plaintiff's claims.",
        ],
        ("judge", "opening"): [
            "Counsel may proceed with opening statements. Remember, save argument for closing.",
        ],
        ("plaintiff_counsel", "opening"): [
            "Ladies and gentlemen, the evidence will show that defendant's actions caused significant harm.",
            "This case is about accountability. We will prove each element of our claims.",
        ],
        ("defense_counsel", "opening"): [
            "Members of the jury, plaintiff's story has holes. The evidence tells a different tale.",
            "We will demonstrate that our client acted properly and lawfully at all times.",
        ],
        ("plaintiff_counsel", "closing"): [
            "The evidence has proven our case. We ask you to hold defendant accountable.",
        ],
        ("defense_counsel", "closing"): [
            "Plaintiff has failed to meet their burden. The only verdict is in favor of defendant.",
        ],
        ("jury_foreperson", "deliberation"): [
            "We need to carefully review the evidence presented by both sides.",
            "Let's discuss the key points and work toward a fair verdict.",
        ],
        ("jury_foreperson", "verdict"): [
            "We the jury find in favor of the plaintiff on all counts.",
            "We the jury find in favor of the defendant.",
        ],
    }
    
    key = (agent_id, phase.value)
    options = responses.get(key, ["I present my arguments to the court."])
    text = random.choice(options)
    
    base_score = np.random.uniform(-0.5, 0.5)
    # No fake score shifting — simulated mode cannot steer generation,
    # so score must faithfully reflect the (unsteered) text.

    return {
        "text": text,
        "score": float(np.clip(base_score, -2, 2)),
        "probe_scores": {state.active_probe: float(base_score)},
        "context": "",
        "shadow_log": {},
    }


# =============================================================================
# HTML DASHBOARD
# =============================================================================

def get_dashboard_html():
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>US Federal Court - PAS Monitor v2</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg: #0c1222; --card: #1a2332; --border: #2d3748;
            --gold: #d4af37; --red: #ef4444; --blue: #3b82f6;
            --green: #22c55e; --purple: #8b5cf6; --amber: #f59e0b;
            --text: #f1f5f9; --muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Georgia', serif; background: var(--bg); color: var(--text); min-height: 100vh; }
        
        .header {
            background: linear-gradient(135deg, #1a2332, #0c1222);
            border-bottom: 3px solid var(--gold);
            padding: 14px 20px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .header h1 { font-size: 1.4rem; color: var(--gold); display: flex; align-items: center; gap: 10px; }
        .status { padding: 5px 12px; border-radius: 15px; font-size: 0.8rem; font-weight: 600; }
        .status-ready { background: var(--green); color: #000; }
        .status-running { background: var(--gold); color: #000; animation: pulse 1.5s infinite; }
        @keyframes pulse { 50% { opacity: 0.7; } }
        
        .main { display: grid; grid-template-columns: 280px 1fr 320px; gap: 14px; padding: 14px; height: calc(100vh - 60px); }
        
        .panel { background: var(--card); border-radius: 8px; border: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
        .panel-header { background: rgba(0,0,0,0.3); padding: 10px 14px; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 0.9rem; }
        .panel-body { padding: 14px; overflow-y: auto; flex: 1; }
        
        .case-card { background: var(--bg); border: 2px solid var(--border); border-radius: 6px; padding: 10px; margin-bottom: 8px; cursor: pointer; transition: all 0.2s; }
        .case-card:hover, .case-card.selected { border-color: var(--gold); }
        .case-card h4 { font-size: 0.85rem; color: var(--gold); margin-bottom: 4px; }
        .case-card p { font-size: 0.75rem; color: var(--muted); }
        
        .control-group { margin-bottom: 14px; }
        .control-group label { display: block; font-size: 0.8rem; color: var(--muted); margin-bottom: 4px; }
        .control-group select, .control-group input { 
            width: 100%; padding: 8px; background: var(--bg); border: 1px solid var(--border); 
            border-radius: 4px; color: var(--text); font-size: 0.85rem;
        }
        
        .agent-card {
            background: var(--bg); border: 2px solid var(--border); border-radius: 6px;
            padding: 10px; margin-bottom: 8px; transition: all 0.3s;
        }
        .agent-card.injected { border-color: var(--red); background: rgba(239,68,68,0.1); }
        .agent-card.speaking { border-color: var(--gold); box-shadow: 0 0 15px rgba(212,175,55,0.4); }
        .agent-icon { font-size: 1.5rem; }
        .agent-name { font-size: 0.85rem; font-weight: 600; }
        .agent-role { font-size: 0.7rem; color: var(--muted); }
        .agent-score { font-size: 1.1rem; font-weight: bold; margin-top: 4px; }
        
        .btn { 
            padding: 10px 16px; border: none; border-radius: 6px; cursor: pointer; 
            font-weight: 600; font-size: 0.85rem; transition: all 0.2s;
        }
        .btn-primary { background: var(--gold); color: #000; }
        .btn-primary:hover { background: #c9a227; }
        .btn-danger { background: var(--red); color: #fff; }
        .btn-secondary { background: var(--border); color: var(--text); }
        
        .transcript-entry {
            background: var(--bg); border-radius: 8px; padding: 14px 16px; margin-bottom: 12px;
            border-left: 4px solid var(--border);
        }
        .transcript-entry.injected { border-left-color: var(--red); background: rgba(239,68,68,0.05); }
        .transcript-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .transcript-agent { font-weight: 600; font-size: 0.95rem; }
        .transcript-score-group { display: flex; align-items: center; gap: 8px; }
        .transcript-score { font-size: 0.8rem; padding: 3px 10px; border-radius: 10px; font-weight: 600; }
        .transcript-score-meaning { font-size: 0.7rem; color: var(--muted); font-style: italic; }
        .transcript-text { font-size: 0.88rem; line-height: 1.65; color: var(--text); white-space: pre-wrap; margin-bottom: 8px; max-height: 300px; overflow-y: auto; padding-right: 4px; }
        .transcript-text::-webkit-scrollbar { width: 4px; }
        .transcript-text::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

        .transcript-actions { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
        .action-btn {
            font-size: 0.78rem; padding: 4px 12px; border-radius: 14px; cursor: pointer;
            border: 1px solid var(--border); background: rgba(255,255,255,0.04);
            color: var(--muted); transition: all 0.2s; display: flex; align-items: center; gap: 4px;
        }
        .action-btn:hover { background: rgba(212,175,55,0.15); border-color: var(--gold); color: var(--gold); }
        .action-btn.active { background: rgba(212,175,55,0.2); border-color: var(--gold); color: var(--gold); }
        .action-btn .badge { background: var(--gold); color: #000; font-size: 0.65rem; padding: 1px 5px; border-radius: 8px; font-weight: 700; }

        .shadow-container { margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.15); border-radius: 6px; border: 1px solid var(--border); }
        .shadow-label { font-size: 0.78rem; font-weight: 600; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }
        .shadow-positive { color: #22c55e; }
        .shadow-negative { color: #ef4444; }
        .shadow-text { font-size: 0.82rem; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 4px; margin-bottom: 8px; font-style: italic; color: var(--muted); line-height: 1.5; }
        .shadow-none { font-size: 0.78rem; color: var(--muted); font-style: italic; padding: 6px 0; }

        .token-heatmap-container { margin-top: 8px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 6px; border: 1px solid var(--border); }
        .token-heatmap-legend { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--muted); margin-bottom: 6px; padding: 0 2px; }
        .token-heatmap-legend span:first-child { color: #22c55e; }
        .token-heatmap-legend span:last-child { color: #ef4444; }
        
        .phase-indicator {
            display: flex; justify-content: center; gap: 8px; padding: 10px;
            background: var(--bg); border-bottom: 1px solid var(--border);
        }
        .phase-dot { 
            width: 12px; height: 12px; border-radius: 50%; background: var(--border);
            transition: all 0.3s;
        }
        .phase-dot.active { background: var(--gold); box-shadow: 0 0 8px var(--gold); }
        .phase-dot.complete { background: var(--green); }
        
        .dna-section { margin-top: 12px; }
        .dna-section img { width: 100%; border-radius: 6px; margin-bottom: 8px; }
        
        .interaction-graph { 
            background: var(--bg); border-radius: 6px; padding: 10px; margin-bottom: 10px;
        }
        .interaction-graph svg { width: 100%; height: 100px; }
        
        .slider-value { 
            display: inline-block; background: var(--gold); color: #000; 
            padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; margin-left: 8px;
        }
        
        .checkbox-group { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
        .checkbox-group input { width: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚖️ US Federal Court - PAS Monitor v2</h1>
        <div>
            <span id="statusBadge" class="status status-ready">Ready</span>
            <span id="modelName" style="color:var(--muted);font-size:0.8rem;margin-left:12px;"></span>
        </div>
    </div>
    
    <div class="main">
        <!-- Left Panel: Controls -->
        <div class="panel">
            <div class="panel-header">📋 Case Selection</div>
            <div class="panel-body">
                <div class="control-group" style="padding-bottom:10px;border-bottom:1px solid var(--border);margin-bottom:10px;">
                    <label>🤖 Model <span id="modelStatus" style="color:var(--green);font-size:0.7rem;"></span></label>
                    <select id="modelSelect" onchange="switchModel()"></select>
                </div>
                
                <div id="caseList"></div>
                
                <div class="control-group">
                    <label>Trial Type</label>
                    <select id="trialType">
                        <option value="jury">Jury Trial</option>
                        <option value="bench">Bench Trial</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Phases</label>
                    <select id="phasesSelect" multiple size="4">
                        <option value="motions" selected>Pre-Trial Motions</option>
                        <option value="opening" selected>Opening Statements</option>
                        <option value="closing" selected>Closing Arguments</option>
                        <option value="deliberation">Jury Deliberation</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Active Probe</label>
                    <select id="probeSelect"></select>
                </div>
                
                <div class="control-group">
                    <label>Injection Target</label>
                    <select id="targetSelect">
                        <option value="plaintiff_counsel">Plaintiff Counsel</option>
                        <option value="defense_counsel">Defense Counsel</option>
                        <option value="judge">Judge</option>
                        <option value="jury_foreperson">Jury Foreperson</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Injection Strength <span class="slider-value" id="strengthValue">0.0</span></label>
                    <input type="range" id="strengthSlider" min="-3" max="3" step="0.1" value="0">
                </div>
                
                <div style="margin-top:16px;">
                    <button id="startBtn" class="btn btn-primary" style="width:100%;" onclick="startCase()">▶️ Start Session</button>
                    <button id="stopBtn" class="btn btn-danger" style="width:100%;display:none;" onclick="stopCase()">⏹️ Stop</button>
                </div>
                
                <div style="margin-top:12px;">
                    <a href="/dna" target="_blank" class="btn btn-secondary" style="width:100%;display:block;text-align:center;text-decoration:none;background:#0ea5e9;">📊 DNA Details</a>
                </div>

                <div style="margin-top:8px;">
                    <button class="btn btn-secondary" style="width:100%;background:#6366f1;" onclick="saveExperiment()">💾 Save to History</button>
                </div>
                
                <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);">
                    <label style="font-size:0.8rem;color:var(--muted);display:block;margin-bottom:6px;">Cross-Model Comparison</label>
                    <button class="btn btn-secondary" style="width:100%;font-size:0.8rem;" onclick="showCrossModelGalaxy()">🌌 Cross-Model Galaxy</button>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:4px;">
                        <button class="btn btn-secondary" style="font-size:0.75rem;padding:6px;" onclick="showRoleComparison('Judge')">Compare Judges</button>
                        <button class="btn btn-secondary" style="font-size:0.75rem;padding:6px;" onclick="showRoleComparison('Plaintiff')">Compare Plaintiffs</button>
                        <button class="btn btn-secondary" style="font-size:0.75rem;padding:6px;" onclick="showRoleComparison('Defense')">Compare Defense</button>
                        <button class="btn btn-secondary" style="font-size:0.75rem;padding:6px;" onclick="showRoleComparison('Jury')">Compare Jury</button>
                    </div>
                    <label style="font-size:0.8rem;color:var(--muted);display:block;margin-top:8px;margin-bottom:4px;">SAE Injection Impact</label>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;">
                        <button class="btn btn-secondary" style="font-size:0.75rem;padding:6px;" onclick="showSAEInjectionCompare('Judge')">Judge SAE Δ</button>
                        <button class="btn btn-secondary" style="font-size:0.75rem;padding:6px;" onclick="showSAEInjectionCompare('Plaintiff')">Plaintiff SAE Δ</button>
                        <button class="btn btn-secondary" style="font-size:0.75rem;padding:6px;" onclick="showSAEInjectionCompare('Defense')">Defense SAE Δ</button>
                        <button class="btn btn-secondary" style="font-size:0.75rem;padding:6px;" onclick="showSAEInjectionCompare('Jury')">Jury SAE Δ</button>
                    </div>
                    <div id="experimentCount" style="font-size:0.75rem;color:var(--muted);margin-top:6px;text-align:center;"></div>
                </div>
            </div>
        </div>

        <!-- Center Panel: Courtroom -->
        <div class="panel">
            <div class="panel-header">🏛️ Courtroom Proceedings</div>
            <div class="phase-indicator" id="phaseIndicator"></div>
            <div class="panel-body" id="transcript"></div>
        </div>
        
        <!-- Right Panel: Agents & DNA -->
        <div class="panel">
            <div class="panel-header">👥 Court Participants</div>
            <div class="panel-body">
                <div id="agentCards"></div>
                
                <div class="interaction-graph">
                    <h4 style="font-size:0.85rem;margin-bottom:8px;">🔗 Information Flow</h4>
                    <svg id="flowGraph"></svg>
                </div>
                
                <div id="dnaPanel" class="dna-section"></div>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        let selectedCase = null;
        let phaseOrder = ['motions', 'opening', 'closing', 'deliberation', 'verdict'];
        let currentPhase = null;
        // Global token score range across all agents
        let globalTokenMin = Infinity;
        let globalTokenMax = -Infinity;
        let allTokenHeatmaps = [];  // track {id, tokens} for re-coloring
        
        // Initialize
        document.addEventListener('DOMContentLoaded', async () => {
            await loadStatus();
            await loadCases();
            connectWebSocket();
            renderAgentCards();
            drawInteractionGraph();
            
            document.getElementById('strengthSlider').addEventListener('input', (e) => {
                document.getElementById('strengthValue').textContent = parseFloat(e.target.value).toFixed(1);
            });

            // Event delegation for toggle buttons (token scores, counterfactuals)
            document.getElementById('transcript').addEventListener('click', (e) => {
                const toggle = e.target.closest('[data-toggle]');
                if (toggle) {
                    const targetId = toggle.getAttribute('data-toggle');
                    const el = document.getElementById(targetId);
                    if (el) {
                        el.style.display = el.style.display === 'none' ? 'block' : 'none';
                    }
                }
            });
        });
        
        async function loadStatus() {
            const res = await fetch('/api/status');
            const data = await res.json();
            document.getElementById('modelName').textContent = data.model || 'Unknown';
            
            // Populate model dropdown
            const modelSelect = document.getElementById('modelSelect');
            modelSelect.innerHTML = '';
            (data.available_models || []).forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                // Format model name for display (replace _ with /)
                opt.textContent = m.replace('_', '/');
                if (m === data.model) {
                    opt.selected = true;
                }
                modelSelect.appendChild(opt);
            });
            
            // Update model status indicator
            const statusEl = document.getElementById('modelStatus');
            if (statusEl) {
                statusEl.textContent = data.model_loaded ? '● loaded' : '○ not loaded';
                statusEl.style.color = data.model_loaded ? 'var(--green)' : 'var(--red)';
            }
            
            // Populate probes
            const probeSelect = document.getElementById('probeSelect');
            probeSelect.innerHTML = '';
            (data.probes || ['toxicity']).forEach(p => {
                const opt = document.createElement('option');
                opt.value = p;
                opt.textContent = p.charAt(0).toUpperCase() + p.slice(1);
                probeSelect.appendChild(opt);
            });
        }
        
        async function switchModel() {
            const modelSelect = document.getElementById('modelSelect');
            const modelName = modelSelect.value;
            
            if (!modelName) return;
            
            // Show loading state
            const statusEl = document.getElementById('modelStatus');
            if (statusEl) {
                statusEl.textContent = '⏳ loading...';
                statusEl.style.color = 'var(--gold)';
            }
            modelSelect.disabled = true;
            
            try {
                const res = await fetch('/api/models/switch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_name: modelName })
                });
                
                const data = await res.json();
                
                if (res.ok) {
                    document.getElementById('modelName').textContent = modelName;
                    if (statusEl) {
                        statusEl.textContent = '● loaded';
                        statusEl.style.color = 'var(--green)';
                    }
                    // Reload agent cards in case they changed
                    renderAgentCards();
                } else {
                    alert('Failed to switch model: ' + (data.detail || 'Unknown error'));
                    if (statusEl) {
                        statusEl.textContent = '✗ error';
                        statusEl.style.color = 'var(--red)';
                    }
                    // Reload status to restore correct selection
                    await loadStatus();
                }
            } catch (e) {
                alert('Error switching model: ' + e.message);
                if (statusEl) {
                    statusEl.textContent = '✗ error';
                    statusEl.style.color = 'var(--red)';
                }
            } finally {
                modelSelect.disabled = false;
            }
        }
        
        async function loadCases() {
            const res = await fetch('/api/cases');
            const data = await res.json();
            
            const container = document.getElementById('caseList');
            container.innerHTML = '';
            
            data.cases.forEach(c => {
                const card = document.createElement('div');
                card.className = 'case-card';
                card.innerHTML = `<h4>${c.title}</h4><p>${c.summary}</p>`;
                card.onclick = () => selectCase(c.id, card);
                container.appendChild(card);
            });
            
            // Auto-select first
            if (data.cases.length > 0) {
                selectCase(data.cases[0].id, container.firstChild);
            }
        }
        
        function selectCase(id, element) {
            document.querySelectorAll('.case-card').forEach(c => c.classList.remove('selected'));
            element.classList.add('selected');
            selectedCase = id;
        }
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${location.host}/ws`);
            
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                handleMessage(data);
            };
            
            ws.onclose = () => setTimeout(connectWebSocket, 2000);
        }
        
        function handleMessage(data) {
            switch(data.type) {
                case 'session_start':
                    document.getElementById('statusBadge').className = 'status status-running';
                    document.getElementById('statusBadge').textContent = 'In Session';
                    document.getElementById('transcript').innerHTML = '';
                    globalTokenMin = Infinity;
                    globalTokenMax = -Infinity;
                    allTokenHeatmaps = [];
                    renderPhaseIndicator(data.phases);
                    break;
                    
                case 'phase_start':
                    currentPhase = data.phase;
                    updatePhaseIndicator(data.phase, 'active');
                    addTranscriptHeader(data.name, data.description);
                    break;
                    
                case 'phase_complete':
                    updatePhaseIndicator(data.phase, 'complete');
                    break;
                    
                case 'statement':
                    addStatement(data);
                    updateAgentCard(data.agent, data.score, data.is_injected, true);
                    if (data.trajectories) updateTrajectories(data.trajectories);
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
                    if (typeof loadSAEData === 'function') loadSAEData();
                    break;
            }
        }
        
        function renderAgentCards() {
            const agents = {
                judge: { name: 'Hon. Judge Williams', role: 'Presiding Judge', icon: '⚖️', color: '#d4af37' },
                plaintiff_counsel: { name: 'Sarah Chen, Esq.', role: "Plaintiff's Counsel", icon: '👩‍⚖️', color: '#3b82f6' },
                defense_counsel: { name: 'Michael Torres, Esq.', role: 'Defense Counsel', icon: '👨‍⚖️', color: '#ef4444' },
                jury_foreperson: { name: 'Jury Foreperson', role: 'Jury Representative', icon: '🧑‍🤝‍🧑', color: '#22c55e' },
            };
            
            const container = document.getElementById('agentCards');
            container.innerHTML = '';
            
            for (const [id, agent] of Object.entries(agents)) {
                const card = document.createElement('div');
                card.className = 'agent-card';
                card.id = `agent-${id}`;
                card.innerHTML = `
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span class="agent-icon">${agent.icon}</span>
                        <div>
                            <div class="agent-name" style="color:${agent.color}">${agent.name}</div>
                            <div class="agent-role">${agent.role}</div>
                        </div>
                    </div>
                    <div class="agent-score" id="score-${id}" style="color:var(--muted)">--</div>
                `;
                container.appendChild(card);
            }
        }
        
        function updateAgentCard(agentId, score, isInjected, isSpeaking) {
            const card = document.getElementById(`agent-${agentId}`);
            if (!card) return;
            
            document.querySelectorAll('.agent-card').forEach(c => c.classList.remove('speaking'));
            
            if (isInjected) card.classList.add('injected');
            if (isSpeaking) card.classList.add('speaking');
            
            const scoreEl = document.getElementById(`score-${agentId}`);
            if (scoreEl) {
                scoreEl.textContent = score.toFixed(3);
                scoreEl.style.color = score > 0.5 ? 'var(--red)' : score < -0.5 ? 'var(--green)' : 'var(--muted)';
            }
        }
        
        function renderPhaseIndicator(phases) {
            const container = document.getElementById('phaseIndicator');
            container.innerHTML = phases.map(p => 
                `<div class="phase-dot" id="phase-${p}" title="${p}"></div>`
            ).join('');
        }
        
        function updatePhaseIndicator(phase, status) {
            const dot = document.getElementById(`phase-${phase}`);
            if (dot) {
                dot.classList.remove('active', 'complete');
                dot.classList.add(status);
            }
        }
        
        function addTranscriptHeader(name, description) {
            const transcript = document.getElementById('transcript');
            transcript.insertAdjacentHTML('beforeend', `
                <div style="text-align:center;padding:16px;border-bottom:1px solid var(--border);margin-bottom:12px;">
                    <h3 style="color:var(--gold);font-size:1.1rem;">${escHtml(name)}</h3>
                    <p style="color:var(--muted);font-size:0.85rem;">${escHtml(description)}</p>
                </div>
            `);
        }

        function escHtml(s) {
            if (!s) return '';
            return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
        }

        function recolorAllTokenHeatmaps() {
            const range = Math.max(globalTokenMax - globalTokenMin, 0.01);
            for (const hm of allTokenHeatmaps) {
                const container = document.getElementById(hm.id);
                if (!container) continue;
                const spans = container.querySelectorAll('span[data-tscore]');
                spans.forEach(span => {
                    const v = parseFloat(span.getAttribute('data-tscore'));
                    const norm = (v - globalTokenMin) / range;
                    const r = Math.round(255 * Math.min(1, norm * 2));
                    const g = Math.round(255 * Math.min(1, (1 - norm) * 2));
                    const opacity = 0.25 + 0.6 * Math.abs(norm - 0.5) * 2;
                    span.style.background = `rgba(${r},${g},0,${opacity.toFixed(2)})`;
                });
                // Update legend
                const legend = container.querySelector('.token-heatmap-legend');
                if (legend) {
                    const legendSpans = legend.querySelectorAll('span');
                    if (legendSpans.length >= 2) {
                        legendSpans[0].textContent = `◀ Negative (${globalTokenMin.toFixed(2)})`;
                        legendSpans[legendSpans.length - 1].textContent = `Positive (${globalTokenMax.toFixed(2)}) ▶`;
                    }
                }
            }
        }

        function addStatement(data) {
            const transcript = document.getElementById('transcript');

            const s = data.score;
            const scoreColor = s > 0.5 ? 'var(--red)' : s < -0.5 ? 'var(--green)' : 'var(--muted)';
            const injectedClass = data.is_injected ? 'injected' : '';
            const injProbe = data.injection_probe ? ` [${data.injection_probe}` + (data.injection_strength ? ` @ ${data.injection_strength}` : '') + ']' : '';
            const injectedBadge = data.is_injected ? ` <span style="color:var(--red)">⚡ INJECTED${injProbe}</span>` : '';

            // Score interpretation based on active probe
            const absScore = Math.abs(s);
            let scoreMeaning = '';
            let scoreIcon = '';
            if (absScore < 0.3) { scoreMeaning = 'Neutral'; scoreIcon = '~'; }
            else if (absScore < 1.0) { scoreMeaning = s > 0 ? 'Mildly Positive' : 'Mildly Negative'; scoreIcon = s > 0 ? '+' : '-'; }
            else if (absScore < 3.0) { scoreMeaning = s > 0 ? 'Moderate' : 'Moderate Negative'; scoreIcon = s > 0 ? '++' : '--'; }
            else { scoreMeaning = s > 0 ? 'Strong Positive' : 'Strong Negative'; scoreIcon = s > 0 ? '+++' : '---'; }

            // Token scores section
            let tokenBtn = '';
            let tokenPanel = '';
            const hasTokens = data.token_scores && data.token_scores.length > 0;
            if (hasTokens) {
                const tokenEntryId = `tokens-${data.phase}-${data.round}-${data.agent}`;
                // Update global min/max across all agents
                let rangeChanged = false;
                for (const ts of data.token_scores) {
                    if (ts.score < globalTokenMin) { globalTokenMin = ts.score; rangeChanged = true; }
                    if (ts.score > globalTokenMax) { globalTokenMax = ts.score; rangeChanged = true; }
                }
                const gRange = Math.max(globalTokenMax - globalTokenMin, 0.01);
                const tokenSpans = data.token_scores.map(ts => {
                    const v = ts.score;
                    const norm = (v - globalTokenMin) / gRange;
                    // Green (safe) -> Yellow (neutral) -> Red (harmful)
                    const r = Math.round(255 * Math.min(1, norm * 2));
                    const g = Math.round(255 * Math.min(1, (1 - norm) * 2));
                    const opacity = 0.25 + 0.6 * Math.abs(norm - 0.5) * 2;
                    const bg = `rgba(${r},${g},0,${opacity.toFixed(2)})`;
                    return `<span data-tscore="${v}" style="background:${bg};padding:1px 3px;border-radius:2px;white-space:pre-wrap;" title="${ts.token}: ${v.toFixed(4)}">${escHtml(ts.token)}</span>`;
                }).join('');
                tokenBtn = `<div class="action-btn" onclick="(function(e){var el=document.getElementById('${tokenEntryId}');if(el)el.style.display=el.style.display==='none'?'block':'none';e.stopPropagation();})(event)">🔬 Token Heatmap <span class="badge">${data.token_scores.length}</span></div>`;
                tokenPanel = `
                    <div class="token-heatmap-container" id="${tokenEntryId}" style="display:none;">
                        <div class="token-heatmap-legend">
                            <span>◀ Negative (${globalTokenMin.toFixed(2)})</span>
                            <span style="color:var(--muted)">Token-level probe scores (global scale)</span>
                            <span>Positive (${globalTokenMax.toFixed(2)}) ▶</span>
                        </div>
                        <div style="font-family:monospace;font-size:0.8rem;line-height:1.7;word-break:break-all;">${tokenSpans}</div>
                    </div>
                `;
                allTokenHeatmaps.push({id: tokenEntryId});
                // Re-color all previous heatmaps if global range expanded
                if (rangeChanged && allTokenHeatmaps.length > 1) {
                    recolorAllTokenHeatmaps();
                }
            }

            // No data indicator
            if (!hasTokens) tokenBtn = `<div class="action-btn" style="opacity:0.35;cursor:default;">🔬 No Token Data</div>`;

            transcript.insertAdjacentHTML('beforeend', `
                <div class="transcript-entry ${injectedClass}">
                    <div class="transcript-header">
                        <span class="transcript-agent" style="color:${data.color}">${data.icon} ${escHtml(data.name)}${injectedBadge}</span>
                        <div class="transcript-score-group">
                            <span class="transcript-score-meaning">${scoreIcon} ${scoreMeaning}</span>
                            <span class="transcript-score" style="background:${scoreColor};color:#000;">R${data.round} | avg: ${s.toFixed(3)}${hasTokens ? ' (' + data.token_scores.length + ' tokens)' : ''}</span>
                        </div>
                    </div>
                    <div class="transcript-text">${escHtml(data.text)}</div>
                    <div class="transcript-actions">
                        ${tokenBtn}
                    </div>
                    ${tokenPanel}
                </div>
            `);

            transcript.scrollTop = transcript.scrollHeight;
        }
        
        function updateTrajectories(trajectories) {
            // Could add a chart here if needed
        }
        
        function drawInteractionGraph() {
            const svg = document.getElementById('flowGraph');
            svg.innerHTML = `
                <defs>
                    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
                        <path d="M0,0 L0,6 L9,3 z" fill="#d4af37"/>
                    </marker>
                </defs>
                
                <!-- Judge at top -->
                <circle cx="150" cy="20" r="12" fill="#d4af37"/>
                <text x="150" y="45" fill="#94a3b8" font-size="10" text-anchor="middle">Judge</text>
                
                <!-- Plaintiff left -->
                <circle cx="60" cy="70" r="12" fill="#3b82f6"/>
                <text x="60" y="95" fill="#94a3b8" font-size="10" text-anchor="middle">Plaintiff</text>
                
                <!-- Defense right -->
                <circle cx="240" cy="70" r="12" fill="#ef4444"/>
                <text x="240" y="95" fill="#94a3b8" font-size="10" text-anchor="middle">Defense</text>
                
                <!-- Arrows -->
                <line x1="75" y1="60" x2="135" y2="30" stroke="#334155" stroke-width="1.5" marker-end="url(#arrow)"/>
                <line x1="225" y1="60" x2="165" y2="30" stroke="#334155" stroke-width="1.5" marker-end="url(#arrow)"/>
                <line x1="80" y1="70" x2="220" y2="70" stroke="#334155" stroke-width="1.5" stroke-dasharray="4"/>
            `;
        }
        
        async function startCase() {
            if (!selectedCase) {
                alert('Please select a case');
                return;
            }
            
            const selectedPhases = Array.from(document.getElementById('phasesSelect').selectedOptions).map(o => o.value);
            
            const res = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    case_id: selectedCase,
                    trial_type: document.getElementById('trialType').value,
                    phases: selectedPhases,
                    injection_target: document.getElementById('targetSelect').value,
                    injection_strength: parseFloat(document.getElementById('strengthSlider').value),
                    probe: document.getElementById('probeSelect').value,
                    shadow_mode: false,
                })
            });
            
            if (res.ok) {
                document.getElementById('startBtn').style.display = 'none';
                document.getElementById('stopBtn').style.display = 'block';
                document.querySelectorAll('.agent-card').forEach(c => c.classList.remove('injected', 'speaking'));
                document.getElementById('dnaPanel').innerHTML = '';
            }
        }
        
        async function stopCase() {
            await fetch('/api/stop', { method: 'POST' });
            document.getElementById('startBtn').style.display = 'block';
            document.getElementById('stopBtn').style.display = 'none';
        }
        
        async function analyzeDNA() {
            try {
                const res = await fetch('/api/dna/analyze', { method: 'POST' });
                if (res.ok) {
                    loadDNA();
                } else {
                    const err = await res.json();
                    alert('DNA Analysis failed: ' + (err.detail || 'Unknown error'));
                }
            } catch (e) {
                alert('DNA Analysis error: ' + e.message);
            }
        }
        
        async function loadDNA() {
            try {
                const res = await fetch('/api/dna/visualizations');
                const data = await res.json();
                
                let html = '<h4 style="color:var(--gold);margin-bottom:10px;">🧬 Behavioral DNA</h4>';
                
                if (data.galaxy) {
                    html += `<img src="data:image/png;base64,${data.galaxy}" alt="DNA Galaxy" style="width:100%;border-radius:6px;margin-bottom:8px;">`;
                }
                if (data.trajectories) {
                    html += `<img src="data:image/png;base64,${data.trajectories}" alt="Trajectories" style="width:100%;border-radius:6px;margin-bottom:8px;">`;
                }
                if (data.heatmap) {
                    html += `<img src="data:image/png;base64,${data.heatmap}" alt="Heatmap" style="width:100%;border-radius:6px;">`;
                }
                
                if (!data.galaxy && !data.trajectories && !data.heatmap) {
                    html += '<p style="color:var(--muted);text-align:center;font-size:0.85rem;">Run analysis to generate visualizations</p>';
                }

                // Also load SAE visualizations if available
                try {
                    const saeRes = await fetch('/api/dna/sae/visualizations');
                    const saeData = await saeRes.json();
                    if (saeData.radar || saeData.diff || saeData.sparsity) {
                        html += '<h4 style="color:var(--gold);margin:12px 0 8px;">SAE Fingerprint</h4>';
                        if (saeData.radar) {
                            html += `<img src="data:image/png;base64,${saeData.radar}" alt="SAE Radar" style="width:100%;border-radius:6px;margin-bottom:8px;">`;
                        }
                        if (saeData.diff) {
                            html += `<img src="data:image/png;base64,${saeData.diff}" alt="SAE Diff" style="width:100%;border-radius:6px;margin-bottom:8px;">`;
                        }
                    }
                } catch (e) { /* SAE data not available yet */ }

                document.getElementById('dnaPanel').innerHTML = html;
            } catch (e) {
                console.error('Failed to load DNA:', e);
            }
        }
        
        // Cross-Model Comparison Functions
        async function saveExperiment() {
            try {
                const topic = prompt('Enter experiment topic/description:', selectedCase || 'Court Simulation');
                if (topic === null) return;  // Cancelled
                
                const res = await fetch(`/api/experiments/save?topic=${encodeURIComponent(topic)}`, { method: 'POST' });
                const data = await res.json();
                
                if (res.ok) {
                    alert(`Experiment saved!\\nID: ${data.experiment_id}\\nTotal experiments: ${data.total_experiments}`);
                    updateExperimentCount();
                } else {
                    alert('Failed to save: ' + (data.detail || 'Unknown error'));
                }
            } catch (e) {
                alert('Error saving experiment: ' + e.message);
            }
        }
        
        async function updateExperimentCount() {
            try {
                const res = await fetch('/api/experiments/summary');
                const data = await res.json();
                
                const countEl = document.getElementById('experimentCount');
                if (countEl && data.total_experiments !== undefined) {
                    const models = data.models ? data.models.join(', ') : 'none';
                    countEl.innerHTML = `📊 ${data.total_experiments} experiments saved<br><span style="font-size:0.7rem;">Models: ${models}</span>`;
                }
            } catch (e) {
                console.error('Failed to get experiment count:', e);
            }
        }
        
        async function showCrossModelGalaxy() {
            try {
                const res = await fetch('/api/experiments/galaxy?color_by=model&marker_by=role');
                const data = await res.json();
                
                if (res.ok && data.image_base64) {
                    showGalaxyModal('Cross-Model Behavioral Galaxy', data.image_base64);
                } else {
                    alert('Failed to generate galaxy: ' + (data.detail || 'Not enough experiments'));
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        
        async function showRoleComparison(role) {
            try {
                const res = await fetch(`/api/experiments/role-comparison/${role}`);
                const data = await res.json();

                if (res.ok && data.image_base64) {
                    showGalaxyModal(`${role} Comparison Across Models`, data.image_base64);
                } else {
                    alert('Failed to generate comparison: ' + (data.detail || 'Not enough data for this role'));
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        async function showSAEInjectionCompare(role) {
            try {
                const res = await fetch(`/api/experiments/sae-injection-compare/${role}`);
                const data = await res.json();

                if (res.ok) {
                    if (data.chart) {
                        showGalaxyModal(`SAE Latent Shift: ${role} (Injected vs Baseline)`, data.chart);
                    } else if (data.top_diffs && data.top_diffs.length > 0) {
                        // Fallback: show text summary
                        let msg = `SAE Injection Impact for ${role}\\n`;
                        msg += `Baseline: ${data.n_baseline} experiments | Injected: ${data.n_injected} experiments\\n\\n`;
                        msg += `Top shifted latents:\\n`;
                        data.top_diffs.slice(0, 10).forEach(d => {
                            msg += `  ${d.label}: ${d.diff_pct > 0 ? '+' : ''}${d.diff_pct.toFixed(1)}%\\n`;
                        });
                        alert(msg);
                    } else {
                        alert('No SAE feature differences found between injected and baseline.');
                    }
                } else {
                    alert('Failed: ' + (data.detail || 'Not enough data'));
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }

        function showGalaxyModal(title, imageBase64) {
            // Create modal overlay
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0,0,0,0.85); z-index: 1000;
                display: flex; align-items: center; justify-content: center;
                cursor: pointer;
            `;
            modal.onclick = () => modal.remove();
            
            const content = document.createElement('div');
            content.style.cssText = `
                background: var(--card); border-radius: 12px; padding: 20px;
                max-width: 90vw; max-height: 90vh; overflow: auto;
            `;
            content.onclick = (e) => e.stopPropagation();
            
            content.innerHTML = `
                <h3 style="color:var(--gold);margin-bottom:15px;">${title}</h3>
                <img src="data:image/png;base64,${imageBase64}" style="max-width:100%;border-radius:8px;">
                <p style="color:var(--muted);font-size:0.8rem;margin-top:10px;text-align:center;">Click outside to close</p>
            `;
            
            modal.appendChild(content);
            document.body.appendChild(modal);
        }
        
        // Load experiment count on page load
        document.addEventListener('DOMContentLoaded', () => {
            updateExperimentCount();
        });
    </script>
</body>
</html>'''
    return _apply_base_path(html)


# =============================================================================
# DNA ANALYSIS PAGE
# =============================================================================

def get_dna_page_html():
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DNA Feature Analysis - PAS Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg: #0c1222; --card: #1a2332; --border: #2d3748;
            --gold: #d4af37; --red: #ef4444; --blue: #3b82f6;
            --green: #22c55e; --purple: #8b5cf6; --amber: #f59e0b;
            --text: #f1f5f9; --muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Georgia', serif; background: var(--bg); color: var(--text); min-height: 100vh; }

        .header {
            background: linear-gradient(135deg, #1a2332, #0c1222);
            border-bottom: 3px solid var(--gold);
            padding: 14px 24px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .header h1 { font-size: 1.4rem; color: var(--gold); }
        .header a { color: var(--muted); text-decoration: none; font-size: 0.9rem; }
        .header a:hover { color: var(--gold); }

        .container { max-width: 1400px; margin: 0 auto; padding: 24px; }

        .section { margin-bottom: 32px; }
        .section h2 { color: var(--gold); font-size: 1.2rem; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }

        .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }

        /* Parameter Table */
        .param-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        .param-table th, .param-table td { padding: 10px 14px; text-align: center; border: 1px solid var(--border); }
        .param-table th { background: #0f172a; color: var(--gold); font-weight: 600; }
        .param-table td { font-family: 'Courier New', monospace; }
        .param-table .row-label { text-align: left; color: var(--muted); font-family: Georgia, serif; font-weight: 600; }
        .param-table .injected { border-bottom: 2px solid var(--red); }

        /* Charts */
        .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .chart-container { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }
        .chart-container h3 { color: var(--gold); font-size: 1rem; margin-bottom: 12px; }

        /* Similarity Matrix */
        .sim-table { width: auto; border-collapse: collapse; font-size: 0.85rem; margin: 0 auto; }
        .sim-table th, .sim-table td { padding: 10px 16px; text-align: center; border: 1px solid var(--border); min-width: 80px; }
        .sim-table th { background: #0f172a; color: var(--gold); font-size: 0.8rem; }

        /* Phase table */
        .phase-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        .phase-table th, .phase-table td { padding: 10px 14px; text-align: center; border: 1px solid var(--border); }
        .phase-table th { background: #0f172a; color: var(--gold); }
        .phase-table .row-label { text-align: left; color: var(--muted); font-family: Georgia, serif; }

        .loading { text-align: center; color: var(--muted); padding: 40px; font-size: 1.1rem; }
        .no-data { text-align: center; color: var(--amber); padding: 60px; }
        .no-data a { color: var(--gold); }

        @media (max-width: 900px) { .chart-row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>DNA Feature Vector Analysis</h1>
        <a href="/">Back to Court Dashboard</a>
    </div>

    <div class="container" id="content">
        <div class="loading" id="loadingMsg">Loading DNA data...</div>
    </div>

    <script>
    const AGENT_COLORS = {
        judge: '#d4af37',
        plaintiff_counsel: '#3b82f6',
        defense_counsel: '#ef4444',
        jury_foreperson: '#22c55e',
    };

    const FEATURE_LABELS = {
        mean_score: 'Mean Score',
        std_score: 'Std Dev',
        min_score: 'Min Score',
        max_score: 'Max Score',
        score_range: 'Score Range',
        drift: 'Drift',
        num_statements: 'Statements',
    };

    function scoreColor(val, absMax) {
        if (absMax === 0) return 'transparent';
        const ratio = Math.abs(val) / absMax;
        if (val > 0) return `rgba(239,68,68,${0.15 + ratio * 0.5})`;
        if (val < 0) return `rgba(34,197,94,${0.15 + ratio * 0.5})`;
        return 'transparent';
    }

    function phaseColor(val) {
        if (val > 0.3) return 'rgba(239,68,68,0.4)';
        if (val > 0.1) return 'rgba(239,68,68,0.2)';
        if (val < -0.3) return 'rgba(34,197,94,0.4)';
        if (val < -0.1) return 'rgba(34,197,94,0.2)';
        return 'transparent';
    }

    function simColor(val) {
        const g = Math.round(80 + val * 175);
        return `rgba(${255 - g}, ${g}, 100, 0.5)`;
    }

    async function loadData() {
        try {
            const res = await fetch('/api/dna/parameters');
            if (!res.ok) {
                document.getElementById('content').innerHTML = `
                    <div class="no-data">
                        <h2>No DNA Data Available</h2>
                        <p style="margin-top:12px;">Run a court session first, then DNA analysis will be computed automatically.</p>
                        <p style="margin-top:8px;"><a href="/">Back to Dashboard</a></p>
                    </div>`;
                return;
            }
            const data = await res.json();
            render(data);
        } catch (e) {
            document.getElementById('content').innerHTML = `<div class="no-data">Error loading data: ${e.message}</div>`;
        }
    }

    function render(data) {
        const agents = data.agents;
        const agentIds = Object.keys(agents);
        const similarity = data.similarity || {};

        let html = `<p style="color:var(--muted);margin-bottom:24px;">Model: <strong style="color:var(--text)">${data.model || 'unknown'}</strong> &nbsp;|&nbsp; Probe: <strong style="color:var(--text)">${data.active_probe || 'none'}</strong></p>`;

        // === 1. Parameter Comparison Table ===
        html += `<div class="section"><h2>Agent Feature Vector Comparison</h2><div class="card" style="overflow-x:auto;">`;
        html += `<table class="param-table"><thead><tr><th></th>`;
        agentIds.forEach(id => {
            const a = agents[id];
            const injBadge = a.is_injected ? ' <span style="color:var(--red)">&#9889;</span>' : '';
            html += `<th style="color:${AGENT_COLORS[id] || 'var(--text)'}">${a.name}${injBadge}</th>`;
        });
        html += `</tr></thead><tbody>`;

        const features = ['mean_score','std_score','min_score','max_score','score_range','drift','num_statements'];
        features.forEach(feat => {
            const vals = agentIds.map(id => agents[id][feat]);
            const absMax = Math.max(...vals.map(Math.abs), 0.001);
            html += `<tr><td class="row-label">${FEATURE_LABELS[feat] || feat}</td>`;
            agentIds.forEach(id => {
                const v = agents[id][feat];
                const bg = feat === 'num_statements' ? 'transparent' : scoreColor(v, absMax);
                html += `<td style="background:${bg}">${typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : v}</td>`;
            });
            html += `</tr>`;
        });
        html += `</tbody></table></div></div>`;

        // === 2. Charts Row: Radar + Bar ===
        html += `<div class="chart-row">`;

        // Radar chart container
        html += `<div class="chart-container"><h3>Behavioral Radar</h3><canvas id="radarChart"></canvas></div>`;

        // Score distribution bar chart container
        html += `<div class="chart-container"><h3>Score Statistics</h3><canvas id="barChart"></canvas></div>`;

        html += `</div>`;

        // === 3. Phase Breakdown ===
        const allPhases = new Set();
        agentIds.forEach(id => {
            Object.keys(agents[id].phase_scores || {}).forEach(p => allPhases.add(p));
        });
        const phases = Array.from(allPhases);

        if (phases.length > 0) {
            html += `<div class="section" style="margin-top:32px;"><h2>Phase x Agent Breakdown</h2><div class="card" style="overflow-x:auto;">`;
            html += `<table class="phase-table"><thead><tr><th>Phase</th>`;
            agentIds.forEach(id => {
                html += `<th style="color:${AGENT_COLORS[id] || 'var(--text)'}">${agents[id].name}</th>`;
            });
            html += `</tr></thead><tbody>`;
            phases.forEach(phase => {
                html += `<tr><td class="row-label">${phase}</td>`;
                agentIds.forEach(id => {
                    const ps = (agents[id].phase_scores || {})[phase];
                    if (ps) {
                        const bg = phaseColor(ps.mean);
                        html += `<td style="background:${bg}">${ps.mean.toFixed(4)} <span style="color:var(--muted);font-size:0.75rem;">(n=${ps.count})</span></td>`;
                    } else {
                        html += `<td style="color:var(--muted);">-</td>`;
                    }
                });
                html += `</tr>`;
            });
            html += `</tbody></table></div></div>`;
        }

        // === 4. Cosine Similarity Matrix ===
        if (agentIds.length >= 2) {
            html += `<div class="section"><h2>Feature Vector Similarity (Cosine)</h2><div class="card" style="overflow-x:auto;">`;
            html += `<table class="sim-table"><thead><tr><th></th>`;
            agentIds.forEach(id => {
                html += `<th style="color:${AGENT_COLORS[id] || 'var(--text)'}">${agents[id].name.split(',')[0]}</th>`;
            });
            html += `</tr></thead><tbody>`;
            agentIds.forEach((id1, i) => {
                html += `<tr><td class="row-label" style="text-align:left;color:${AGENT_COLORS[id1] || 'var(--muted)'};font-weight:600;">${agents[id1].name.split(',')[0]}</td>`;
                agentIds.forEach((id2, j) => {
                    if (i === j) {
                        html += `<td style="background:rgba(100,200,100,0.3);font-weight:600;">1.0</td>`;
                    } else {
                        const key = i < j ? `${id1}|${id2}` : `${id2}|${id1}`;
                        const val = similarity[key] !== undefined ? similarity[key] : 0;
                        html += `<td style="background:${simColor(val)}">${val.toFixed(4)}</td>`;
                    }
                });
                html += `</tr>`;
            });
            html += `</tbody></table></div></div>`;
        }

        // === 5. SAE Fingerprint Section ===
        html += `<div class="section" id="saeSection" style="margin-top:32px;">
            <h2>SAE Functional Fingerprint</h2>
            <div class="card" id="saeContent">
                <div id="saeLoading" style="display:none;text-align:center;color:var(--muted);padding:20px;">Enriching with SAE features...</div>
                <div id="saeResults" style="display:none;"></div>
                <div id="saeEmpty">
                    <p style="color:var(--muted);margin-bottom:12px;">SAE fingerprinting extracts semantic behavioral features using a Sparse Autoencoder to create interpretable model fingerprints.</p>
                    <button class="btn btn-secondary" style="padding:8px 20px;background:var(--card);border:1px solid var(--gold);color:var(--gold);border-radius:6px;cursor:pointer;font-size:0.9rem;" onclick="runSAEEnrichment()">Run SAE Enrichment (Demo)</button>
                </div>
            </div>
        </div>`;

        document.getElementById('content').innerHTML = html;

        // === Render Charts ===
        renderRadar(agents, agentIds);
        renderBar(agents, agentIds);

        // Check if SAE data already exists
        loadSAEData();
    }

    function renderRadar(agents, agentIds) {
        const canvas = document.getElementById('radarChart');
        if (!canvas) return;

        const radarFeats = ['mean_score','std_score','score_range','drift','min_score','max_score'];
        const radarLabels = radarFeats.map(f => FEATURE_LABELS[f] || f);

        // Compute min/max for normalization
        const mins = {}, maxs = {};
        radarFeats.forEach(f => {
            const vals = agentIds.map(id => agents[id][f]);
            mins[f] = Math.min(...vals);
            maxs[f] = Math.max(...vals);
        });

        const datasets = agentIds.map(id => {
            const a = agents[id];
            const color = AGENT_COLORS[id] || '#888';
            const values = radarFeats.map(f => {
                const range = maxs[f] - mins[f];
                return range > 0 ? (a[f] - mins[f]) / range : 0.5;
            });
            return {
                label: a.name.split(',')[0],
                data: values,
                borderColor: color,
                backgroundColor: color + '20',
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: color,
            };
        });

        new Chart(canvas, {
            type: 'radar',
            data: { labels: radarLabels, datasets },
            options: {
                responsive: true,
                scales: {
                    r: {
                        beginAtZero: true, max: 1,
                        ticks: { display: false },
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        pointLabels: { color: '#94a3b8', font: { size: 11 } },
                    }
                },
                plugins: {
                    legend: { labels: { color: '#f1f5f9', font: { size: 11 } } }
                }
            }
        });
    }

    function renderBar(agents, agentIds) {
        const canvas = document.getElementById('barChart');
        if (!canvas) return;

        const barFeats = ['mean_score','std_score','min_score','max_score'];
        const barLabels = barFeats.map(f => FEATURE_LABELS[f]);

        const datasets = agentIds.map(id => {
            const a = agents[id];
            const color = AGENT_COLORS[id] || '#888';
            return {
                label: a.name.split(',')[0],
                data: barFeats.map(f => a[f]),
                backgroundColor: color + '90',
                borderColor: color,
                borderWidth: 1,
            };
        });

        new Chart(canvas, {
            type: 'bar',
            data: { labels: barLabels, datasets },
            options: {
                responsive: true,
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                },
                plugins: {
                    legend: { labels: { color: '#f1f5f9', font: { size: 11 } } }
                }
            }
        });
    }

    // === SAE Fingerprint Functions ===

    async function loadSAEData() {
        try {
            const [fpRes, vizRes] = await Promise.all([
                fetch('/api/dna/sae/fingerprints'),
                fetch('/api/dna/sae/visualizations'),
            ]);
            const fingerprints = await fpRes.json();
            const visualizations = await vizRes.json();

            if (Object.keys(fingerprints).length > 0) {
                renderSAE(fingerprints, visualizations);
            }
        } catch (e) {
            console.log('No SAE data available yet');
        }
    }

    async function runSAEEnrichment() {
        const saeEmpty = document.getElementById('saeEmpty');
        const saeLoading = document.getElementById('saeLoading');
        if (saeEmpty) saeEmpty.style.display = 'none';
        if (saeLoading) saeLoading.style.display = 'block';

        try {
            // Generate synthetic SAE frequencies for demo
            // In production, provide reader_model_name + sae_path
            const agents = await fetch('/api/dna/parameters').then(r => r.json());
            const agentIds = Object.keys(agents.agents || {});

            const precomputed = {};
            agentIds.forEach(id => {
                // Generate plausible synthetic SAE frequencies
                const d_sae = 500;
                const freqs = [];
                const baseSeed = id.charCodeAt(0) + id.length;
                for (let i = 0; i < d_sae; i++) {
                    // Pseudo-random based on agent and feature index
                    const v = Math.sin(baseSeed * 0.1 + i * 0.37) * 0.5 + 0.5;
                    freqs.push(v > 0.7 ? v * 0.4 : 0);
                }
                precomputed[id] = freqs;
            });

            const res = await fetch('/api/dna/sae/enrich', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ precomputed }),
            });

            if (res.ok) {
                await loadSAEData();
            } else {
                const err = await res.json();
                alert('SAE enrichment failed: ' + (err.detail || 'Unknown error'));
                if (saeEmpty) saeEmpty.style.display = 'block';
            }
        } catch (e) {
            alert('SAE error: ' + e.message);
            if (saeEmpty) saeEmpty.style.display = 'block';
        }
        if (saeLoading) saeLoading.style.display = 'none';
    }

    function renderSAE(fingerprints, visualizations) {
        const saeResults = document.getElementById('saeResults');
        const saeEmpty = document.getElementById('saeEmpty');
        const saeLoading = document.getElementById('saeLoading');
        if (!saeResults) return;
        if (saeEmpty) saeEmpty.style.display = 'none';
        if (saeLoading) saeLoading.style.display = 'none';
        saeResults.style.display = 'block';

        let html = '';

        // Visualization images
        if (visualizations.sparsity) {
            html += `<img src="data:image/png;base64,${visualizations.sparsity}" alt="SAE Sparsity" style="width:100%;border-radius:6px;margin-bottom:12px;">`;
        }
        if (visualizations.radar) {
            html += `<img src="data:image/png;base64,${visualizations.radar}" alt="SAE Radar" style="width:100%;border-radius:6px;margin-bottom:12px;">`;
        }
        if (visualizations.diff) {
            html += `<img src="data:image/png;base64,${visualizations.diff}" alt="SAE Diff" style="width:100%;border-radius:6px;margin-bottom:12px;">`;
        }

        // Top features table
        const agentIds = Object.keys(fingerprints);
        if (agentIds.length > 0) {
            html += `<h3 style="color:var(--gold);margin:16px 0 10px;">Top SAE Features per Agent</h3>`;
            html += `<div style="overflow-x:auto;"><table class="param-table"><thead><tr><th>Agent</th><th>Active Features</th><th>Sparsity</th><th>Top Semantic Features</th></tr></thead><tbody>`;

            agentIds.forEach(id => {
                const fp = fingerprints[id];
                const color = AGENT_COLORS[id] || 'var(--text)';
                const name = (fp.name || id).split(',')[0];
                const topFeats = (fp.top_features || []).slice(0, 3)
                    .map(f => `<span style="color:var(--text);">${f.label || 'Latent #'+f.index}</span> <span style="color:var(--muted);font-size:0.75rem;">(${(f.frequency*100).toFixed(1)}%)</span>`)
                    .join(', ');

                html += `<tr>
                    <td class="row-label" style="color:${color};">${name}</td>
                    <td>${fp.n_active_features || 0}</td>
                    <td>${((fp.activation_sparsity || 0) * 100).toFixed(1)}%</td>
                    <td style="text-align:left;font-size:0.8rem;">${topFeats || '-'}</td>
                </tr>`;
            });
            html += `</tbody></table></div>`;

            // Diff selector
            if (agentIds.length >= 2) {
                html += `<div style="margin-top:16px;display:flex;align-items:center;gap:10px;">
                    <span style="color:var(--muted);font-size:0.9rem;">Compare:</span>
                    <select id="saeDiffA" style="background:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px 8px;border-radius:4px;">
                        ${agentIds.map(id => `<option value="${id}">${(fingerprints[id].name||id).split(',')[0]}</option>`).join('')}
                    </select>
                    <span style="color:var(--muted);">vs</span>
                    <select id="saeDiffB" style="background:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px 8px;border-radius:4px;">
                        ${agentIds.map((id,i) => `<option value="${id}" ${i===1?'selected':''}>${(fingerprints[id].name||id).split(',')[0]}</option>`).join('')}
                    </select>
                    <button onclick="loadSAEDiff()" style="padding:4px 14px;background:var(--card);border:1px solid var(--gold);color:var(--gold);border-radius:4px;cursor:pointer;font-size:0.85rem;">Diff</button>
                </div>`;
                html += `<div id="saeDiffResult" style="margin-top:12px;"></div>`;
            }
        }

        saeResults.innerHTML = html;
    }

    async function loadSAEDiff() {
        const a = document.getElementById('saeDiffA').value;
        const b = document.getElementById('saeDiffB').value;
        if (a === b) { alert('Select two different agents'); return; }

        const container = document.getElementById('saeDiffResult');
        container.innerHTML = '<p style="color:var(--muted);text-align:center;">Loading diff...</p>';

        try {
            const res = await fetch(`/api/dna/sae/diff/${encodeURIComponent(a)}/${encodeURIComponent(b)}`);
            const data = await res.json();

            let html = '';
            if (data.chart) {
                html += `<img src="data:image/png;base64,${data.chart}" alt="SAE Diff" style="width:100%;border-radius:6px;margin-bottom:10px;">`;
            }
            html += `<p style="color:var(--muted);font-size:0.85rem;">
                Cosine Similarity: <strong style="color:var(--text)">${(data.cosine_similarity||0).toFixed(4)}</strong> &nbsp;|&nbsp;
                Mean |Diff|: <strong style="color:var(--text)">${(data.mean_absolute_diff||0).toFixed(4)}</strong> &nbsp;|&nbsp;
                Significant Features: <strong style="color:var(--text)">${data.n_significant||0}</strong>
            </p>`;
            container.innerHTML = html;
        } catch (e) {
            container.innerHTML = `<p style="color:var(--red);">Error: ${e.message}</p>`;
        }
    }

    document.addEventListener('DOMContentLoaded', loadData);
    </script>
</body>
</html>'''
    return _apply_base_path(html)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="US Federal Court Dashboard v2")
    parser.add_argument("--model", default="Qwen_Qwen2.5-0.5B-Instruct", 
                       help="Model name (folder name in model-dir, or HuggingFace ID)")
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models"),
                       help="Directory containing local models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    
    args = parser.parse_args()
    
    os.environ["COURT_MODEL"] = args.model
    os.environ["COURT_DEVICE"] = args.device
    os.environ["COURT_MODEL_DIR"] = args.model_dir
    
    print(f"\n⚖️ US Federal Court Simulation v2")
    print(f"   http://{args.host}:{args.port}")
    print(f"   Model: {args.model}")
    print(f"   Model Dir: {args.model_dir}")
    print(f"   Features: SteeredAgent, Shadow Mode, DNA Galaxy, Cross-Model Comparison\n")
    
    uvicorn.run(app, host=args.host, port=args.port)