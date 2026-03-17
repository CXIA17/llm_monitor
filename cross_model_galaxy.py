#!/usr/bin/env python3
"""
Cross-Model Galaxy Visualization
=================================

Aggregates behavioral DNA across multiple experiments with different models
to create comparative visualizations.

Example use cases:
- Compare Qwen-Judge vs Llama-Judge vs GPT-Judge behavior
- Track how the same model behaves across different topics
- Analyze injection effects across model families

Usage:
    from cross_model_galaxy import ExperimentAggregator, CrossModelGalaxy
    
    aggregator = ExperimentAggregator()
    aggregator.add_experiment(
        experiment_id="exp_001",
        model_name="Qwen/Qwen2.5-0.5B",
        signatures=dna_signatures,
        metadata={"topic": "AI regulation", "injection": "toxicity"}
    )
    
    galaxy = CrossModelGalaxy(aggregator)
    img_base64 = galaxy.build_galaxy()
"""

import json
import pickle
import base64
import io
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AgentSignature:
    """Signature for a single agent in an experiment."""
    agent_id: str
    agent_role: str  # e.g., "judge", "plaintiff_counsel", "proposer"
    model_name: str
    model_family: str  # e.g., "Qwen", "Llama", "GPT"
    experiment_id: str
    
    # Behavioral features
    feature_vector: List[float] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    score_drift: float = 0.0
    num_rounds: int = 0
    round_scores: List[float] = field(default_factory=list)
    
    # Injection info
    is_injected: bool = False
    injection_type: str = ""
    injection_strength: float = 0.0
    
    # Metadata
    timestamp: str = ""
    topic: str = ""
    extra: Dict = field(default_factory=dict)
    
    def to_vector(self, normalize: bool = True) -> np.ndarray:
        """Convert to feature vector for dimensionality reduction."""
        if self.feature_vector:
            vec = np.array(self.feature_vector)
        else:
            # Build from available features
            vec = np.array([
                self.mean_score,
                self.std_score,
                self.score_drift,
                float(self.num_rounds),
                float(self.is_injected),
                self.injection_strength,
            ])
        
        if normalize and np.std(vec) > 0:
            vec = (vec - np.mean(vec)) / (np.std(vec) + 1e-8)
        
        return vec


@dataclass  
class ExperimentRecord:
    """Record of a single experiment."""
    experiment_id: str
    model_name: str
    model_family: str
    timestamp: str
    topic: str
    
    # Agent signatures
    signatures: Dict[str, AgentSignature] = field(default_factory=dict)
    
    # Experiment config
    injection_target: str = ""
    injection_strength: float = 0.0
    injection_type: str = ""
    probe_name: str = ""
    
    # Metadata
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# EXPERIMENT AGGREGATOR
# =============================================================================

class ExperimentAggregator:
    """
    Aggregates experiments across multiple models and sessions.
    
    Maintains a database of experiments that can be queried and visualized.
    """
    
    def __init__(self, save_path: Optional[str] = None):
        self.experiments: Dict[str, ExperimentRecord] = {}
        self.save_path = save_path or "./experiment_history.pkl"
        
        # Index for fast lookups
        self._by_model: Dict[str, List[str]] = defaultdict(list)
        self._by_role: Dict[str, List[str]] = defaultdict(list)
        self._by_topic: Dict[str, List[str]] = defaultdict(list)
    
    def add_experiment(
        self,
        experiment_id: str,
        model_name: str,
        signatures: Dict[str, Dict],  # Raw signatures from DNA extractor
        topic: str = "",
        injection_target: str = "",
        injection_strength: float = 0.0,
        injection_type: str = "",
        probe_name: str = "",
        metadata: Dict = None,
    ) -> ExperimentRecord:
        """
        Add an experiment to the aggregator.
        
        Args:
            experiment_id: Unique ID for this experiment
            model_name: Full model name (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
            signatures: Dict of agent signatures from DNA extractor
            topic: Experiment topic/prompt
            injection_target: Which agent was injected
            injection_strength: Injection strength used
            injection_type: Type of injection
            probe_name: Name of probe used
            metadata: Additional metadata
        """
        model_family = self._extract_model_family(model_name)
        timestamp = datetime.now().isoformat()
        
        # Convert raw signatures to AgentSignature objects
        agent_sigs = {}
        for agent_id, sig_data in signatures.items():
            agent_sig = AgentSignature(
                agent_id=agent_id,
                agent_role=self._normalize_role(agent_id),
                model_name=model_name,
                model_family=model_family,
                experiment_id=experiment_id,
                feature_vector=sig_data.get("vector", []),
                mean_score=sig_data.get("mean_score", 0.0),
                std_score=sig_data.get("std_score", 0.0),
                score_drift=sig_data.get("drift", 0.0),
                num_rounds=sig_data.get("num_statements", len(sig_data.get("scores", []))),
                round_scores=sig_data.get("scores", []),
                is_injected=sig_data.get("is_injected", False),
                injection_type=sig_data.get("injection_probe", "") or (injection_type if sig_data.get("is_injected") else ""),
                injection_strength=sig_data.get("injection_strength", injection_strength if sig_data.get("is_injected") else 0.0),
                timestamp=timestamp,
                topic=topic,
            )
            agent_sigs[agent_id] = agent_sig
        
        # Create experiment record
        record = ExperimentRecord(
            experiment_id=experiment_id,
            model_name=model_name,
            model_family=model_family,
            timestamp=timestamp,
            topic=topic,
            signatures=agent_sigs,
            injection_target=injection_target,
            injection_strength=injection_strength,
            injection_type=injection_type,
            probe_name=probe_name,
            metadata=metadata or {},
        )
        
        self.experiments[experiment_id] = record
        
        # Update indices
        self._by_model[model_family].append(experiment_id)
        self._by_topic[topic].append(experiment_id)
        for agent_id in agent_sigs:
            role = self._normalize_role(agent_id)
            self._by_role[role].append(experiment_id)
        
        return record
    
    def _extract_model_family(self, model_name: str) -> str:
        """Extract model family from full model name."""
        name_lower = model_name.lower()
        
        if "qwen" in name_lower:
            return "Qwen"
        elif "llama" in name_lower:
            return "Llama"
        elif "mistral" in name_lower:
            return "Mistral"
        elif "gpt" in name_lower:
            return "GPT"
        elif "gemma" in name_lower:
            return "Gemma"
        elif "phi" in name_lower:
            return "Phi"
        elif "deepseek" in name_lower:
            return "DeepSeek"
        else:
            # Use first part of model name
            return model_name.split("/")[-1].split("-")[0]
    
    def _normalize_role(self, agent_id: str) -> str:
        """Normalize agent ID to role name."""
        role_map = {
            "judge": "Judge",
            "plaintiff_counsel": "Plaintiff",
            "defense_counsel": "Defense",
            "jury_foreperson": "Jury",
            "proposer": "Proposer",
            "critic": "Critic",
            "moderator": "Moderator",
        }
        return role_map.get(agent_id.lower(), agent_id)
    
    def get_all_signatures(
        self,
        filter_model: Optional[str] = None,
        filter_role: Optional[str] = None,
        filter_topic: Optional[str] = None,
        filter_injected: Optional[bool] = None,
    ) -> List[AgentSignature]:
        """
        Get all agent signatures, optionally filtered.
        
        Args:
            filter_model: Filter by model family (e.g., "Qwen")
            filter_role: Filter by role (e.g., "Judge")
            filter_topic: Filter by topic
            filter_injected: Filter by injection status
        """
        signatures = []
        
        for exp_id, exp in self.experiments.items():
            for agent_id, sig in exp.signatures.items():
                # Apply filters
                if filter_model and sig.model_family != filter_model:
                    continue
                if filter_role and sig.agent_role != filter_role:
                    continue
                if filter_topic and sig.topic != filter_topic:
                    continue
                if filter_injected is not None and sig.is_injected != filter_injected:
                    continue
                
                signatures.append(sig)
        
        return signatures
    
    def get_comparison_groups(
        self,
        group_by: str = "model",  # "model", "role", "topic"
    ) -> Dict[str, List[AgentSignature]]:
        """Group signatures for comparison."""
        groups = defaultdict(list)
        
        for sig in self.get_all_signatures():
            if group_by == "model":
                key = sig.model_family
            elif group_by == "role":
                key = sig.agent_role
            elif group_by == "topic":
                key = sig.topic
            else:
                key = "all"
            
            groups[key].append(sig)
        
        return dict(groups)
    
    def save(self, path: Optional[str] = None):
        """Save aggregator to disk."""
        path = path or self.save_path
        with open(path, 'wb') as f:
            pickle.dump({
                'experiments': self.experiments,
                '_by_model': dict(self._by_model),
                '_by_role': dict(self._by_role),
                '_by_topic': dict(self._by_topic),
            }, f)
        print(f"Saved {len(self.experiments)} experiments to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load aggregator from disk."""
        path = path or self.save_path
        if not Path(path).exists():
            print(f"No saved data at {path}")
            return
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.experiments = data['experiments']
        self._by_model = defaultdict(list, data.get('_by_model', {}))
        self._by_role = defaultdict(list, data.get('_by_role', {}))
        self._by_topic = defaultdict(list, data.get('_by_topic', {}))
        
        print(f"Loaded {len(self.experiments)} experiments from {path}")
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "total_experiments": len(self.experiments),
            "models": list(self._by_model.keys()),
            "roles": list(self._by_role.keys()),
            "topics": list(self._by_topic.keys()),
            "experiments_per_model": {k: len(v) for k, v in self._by_model.items()},
        }


# =============================================================================
# CROSS-MODEL GALAXY VISUALIZATION
# =============================================================================

class CrossModelGalaxy:
    """
    Creates galaxy visualizations comparing agents across models and experiments.
    """
    
    # Color schemes for different groupings
    MODEL_COLORS = {
        "Qwen": "#3b82f6",    # Blue
        "Llama": "#ef4444",   # Red
        "Mistral": "#8b5cf6", # Purple
        "GPT": "#22c55e",     # Green
        "Gemma": "#f59e0b",   # Amber
        "Phi": "#06b6d4",     # Cyan
        "DeepSeek": "#ec4899", # Pink
    }
    
    ROLE_COLORS = {
        "Judge": "#d4af37",      # Gold
        "Plaintiff": "#3b82f6",  # Blue
        "Defense": "#ef4444",    # Red
        "Jury": "#22c55e",       # Green
        "Proposer": "#8b5cf6",   # Purple
        "Critic": "#f59e0b",     # Amber
        "Moderator": "#06b6d4",  # Cyan
    }
    
    ROLE_MARKERS = {
        "Judge": "s",      # Square
        "Plaintiff": "^",  # Triangle up
        "Defense": "v",    # Triangle down
        "Jury": "o",       # Circle
        "Proposer": "D",   # Diamond
        "Critic": "p",     # Pentagon
        "Moderator": "*",  # Star
    }
    
    def __init__(self, aggregator: ExperimentAggregator):
        self.aggregator = aggregator
    
    def build_galaxy(
        self,
        color_by: str = "model",  # "model" or "role"
        marker_by: str = "role",  # "model" or "role"
        filter_model: Optional[str] = None,
        filter_role: Optional[str] = None,
        filter_topic: Optional[str] = None,
        method: str = "pca",  # "pca" or "tsne"
        show_ellipses: bool = True,
        title: str = "Cross-Model Behavioral Galaxy",
    ) -> Optional[str]:
        """
        Build cross-model galaxy visualization.
        
        Args:
            color_by: What to use for point colors ("model" or "role")
            marker_by: What to use for point markers ("model" or "role")
            filter_model: Only show this model family
            filter_role: Only show this role
            filter_topic: Only show this topic
            method: Dimensionality reduction method
            show_ellipses: Whether to show cluster ellipses
            title: Plot title
        
        Returns:
            Base64 encoded PNG image
        """
        if not HAS_PLOTTING:
            return None
        
        # Get signatures
        signatures = self.aggregator.get_all_signatures(
            filter_model=filter_model,
            filter_role=filter_role,
            filter_topic=filter_topic,
        )
        
        if len(signatures) < 2:
            print(f"Not enough signatures for galaxy: {len(signatures)}")
            return None
        
        # Build feature matrix
        vectors = []
        for sig in signatures:
            vec = sig.to_vector(normalize=False)
            # Pad to consistent size
            if len(vec) < 20:
                vec = np.pad(vec, (0, 20 - len(vec)))
            vectors.append(vec[:20])
        
        vectors = np.array(vectors)
        
        # Standardize
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors)
        
        # Dimensionality reduction
        if method == "tsne" and len(signatures) > 5:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(signatures) - 1))
        else:
            reducer = PCA(n_components=2)
        
        coords = reducer.fit_transform(vectors_scaled)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
        ax.set_facecolor('white')
        
        # Add background scatter
        np.random.seed(42)
        bg_x = np.random.randn(100) * 2.5
        bg_y = np.random.randn(100) * 2.0
        ax.scatter(bg_x, bg_y, c='#e0e0e0', s=15, alpha=0.3, zorder=1)
        
        # Draw ellipses for clusters
        if show_ellipses:
            self._draw_cluster_ellipses(ax, coords, signatures, color_by)
        
        # Plot points
        for i, sig in enumerate(signatures):
            x, y = coords[i]
            
            # Determine color and marker
            if color_by == "model":
                color = self.MODEL_COLORS.get(sig.model_family, "#6366f1")
            else:
                color = self.ROLE_COLORS.get(sig.agent_role, "#6366f1")
            
            if marker_by == "role":
                marker = self.ROLE_MARKERS.get(sig.agent_role, "o")
            else:
                marker = "o"
            
            # Size based on number of rounds
            size = 100 + sig.num_rounds * 30
            
            # Plot point
            if sig.is_injected:
                ax.scatter(x, y, c=color, s=size, marker=marker,
                          edgecolors='#ff0000', linewidths=2.5, zorder=10, alpha=0.9)
            else:
                ax.scatter(x, y, c=color, s=size, marker=marker,
                          edgecolors='#333', linewidths=1, zorder=10, alpha=0.8)
            
            # Label
            label = f"{sig.model_family[:3]}-{sig.agent_role[:3]}"
            if sig.is_injected:
                label += "⚡"
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=7, color='#333', alpha=0.8)
        
        # Create legends
        self._add_legends(ax, signatures, color_by, marker_by)
        
        # Style
        ax.set_title(title, fontsize=14, color='#333', fontweight='bold', pad=15)
        
        if hasattr(reducer, 'explained_variance_ratio_'):
            var1, var2 = reducer.explained_variance_ratio_[:2] * 100
            ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", color='#666', fontsize=10)
            ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", color='#666', fontsize=10)
        else:
            ax.set_xlabel("Dimension 1", color='#666', fontsize=10)
            ax.set_ylabel("Dimension 2", color='#666', fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ccc')
        ax.spines['bottom'].set_color('#ccc')
        ax.tick_params(colors='#999')
        ax.grid(False)
        
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='white', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return img_str
    
    def _draw_cluster_ellipses(
        self,
        ax,
        coords: np.ndarray,
        signatures: List[AgentSignature],
        group_by: str,
    ):
        """Draw ellipses around clusters."""
        # Group points
        groups = defaultdict(list)
        for i, sig in enumerate(signatures):
            if group_by == "model":
                key = sig.model_family
            else:
                key = sig.agent_role
            groups[key].append(coords[i])
        
        # Ellipse colors
        cluster_colors = [
            '#90EE90', '#87CEEB', '#DDA0DD', '#F0E68C',
            '#98FB98', '#ADD8E6', '#FFB6C1', '#E0FFFF',
        ]
        
        for idx, (group_name, points) in enumerate(groups.items()):
            if len(points) < 2:
                continue
            
            points = np.array(points)
            center = np.mean(points, axis=0)
            
            try:
                if len(points) >= 3:
                    cov = np.cov(points.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    eigenvalues = np.maximum(eigenvalues, 0.01)
                    
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[order]
                    eigenvectors = eigenvectors[:, order]
                    
                    width = 2 * np.sqrt(eigenvalues[0]) * 2.5
                    height = 2 * np.sqrt(eigenvalues[1]) * 2.5
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                else:
                    width, height, angle = 2.0, 1.5, 0
                
                width = max(width, 1.5)
                height = max(height, 1.0)
                
                ellipse = Ellipse(
                    center, width, height, angle=angle,
                    facecolor=cluster_colors[idx % len(cluster_colors)],
                    edgecolor='none', alpha=0.25, zorder=2
                )
                ax.add_patch(ellipse)
            except Exception:
                pass
    
    def _add_legends(
        self,
        ax,
        signatures: List[AgentSignature],
        color_by: str,
        marker_by: str,
    ):
        """Add legends for color and marker meanings."""
        # Color legend
        if color_by == "model":
            models = set(sig.model_family for sig in signatures)
            handles = [plt.scatter([], [], c=self.MODEL_COLORS.get(m, '#6366f1'), 
                                  s=80, label=m) for m in sorted(models)]
            legend1 = ax.legend(handles=handles, title="Model", loc='upper left',
                              framealpha=0.9, fontsize=8)
            ax.add_artist(legend1)
        else:
            roles = set(sig.agent_role for sig in signatures)
            handles = [plt.scatter([], [], c=self.ROLE_COLORS.get(r, '#6366f1'),
                                  s=80, label=r) for r in sorted(roles)]
            legend1 = ax.legend(handles=handles, title="Role", loc='upper left',
                              framealpha=0.9, fontsize=8)
            ax.add_artist(legend1)
        
        # Marker legend (if different from color)
        if marker_by != color_by:
            if marker_by == "role":
                roles = set(sig.agent_role for sig in signatures)
                handles = [plt.scatter([], [], c='gray', s=80, 
                                      marker=self.ROLE_MARKERS.get(r, 'o'),
                                      label=r) for r in sorted(roles)]
                ax.legend(handles=handles, title="Role (Shape)", loc='upper right',
                         framealpha=0.9, fontsize=8)
    
    def build_role_comparison(
        self,
        role: str,
        title: Optional[str] = None,
    ) -> Optional[str]:
        """
        Build galaxy comparing different models for a specific role.
        
        Args:
            role: Role to compare (e.g., "Judge")
            title: Optional custom title
        """
        return self.build_galaxy(
            color_by="model",
            marker_by="model",
            filter_role=role,
            title=title or f"Model Comparison: {role} Role",
        )
    
    def build_model_comparison(
        self,
        model: str,
        title: Optional[str] = None,
    ) -> Optional[str]:
        """
        Build galaxy comparing different roles for a specific model.
        
        Args:
            model: Model family to analyze (e.g., "Qwen")
            title: Optional custom title
        """
        return self.build_galaxy(
            color_by="role",
            marker_by="role",
            filter_model=model,
            title=title or f"Role Comparison: {model}",
        )


# =============================================================================
# INTEGRATION WITH DASHBOARD
# =============================================================================

def create_aggregator_endpoints(app, aggregator: ExperimentAggregator):
    """
    Add FastAPI endpoints for the experiment aggregator.
    
    Usage:
        from cross_model_galaxy import ExperimentAggregator, create_aggregator_endpoints
        
        aggregator = ExperimentAggregator()
        create_aggregator_endpoints(app, aggregator)
    """
    from fastapi import HTTPException
    from pydantic import BaseModel
    
    class AddExperimentRequest(BaseModel):
        experiment_id: str
        model_name: str
        signatures: Dict[str, Dict]
        topic: str = ""
        injection_target: str = ""
        injection_strength: float = 0.0
        probe_name: str = ""
        metadata: Dict = {}
    
    @app.post("/api/aggregator/add")
    async def add_experiment(req: AddExperimentRequest):
        """Add current experiment to aggregator."""
        try:
            record = aggregator.add_experiment(
                experiment_id=req.experiment_id,
                model_name=req.model_name,
                signatures=req.signatures,
                topic=req.topic,
                injection_target=req.injection_target,
                injection_strength=req.injection_strength,
                probe_name=req.probe_name,
                metadata=req.metadata,
            )
            return {"status": "added", "experiment_id": req.experiment_id}
        except Exception as e:
            raise HTTPException(500, str(e))
    
    @app.get("/api/aggregator/summary")
    async def get_summary():
        """Get aggregator summary."""
        return aggregator.summary()
    
    @app.get("/api/aggregator/galaxy")
    async def get_cross_model_galaxy(
        color_by: str = "model",
        marker_by: str = "role",
        filter_model: Optional[str] = None,
        filter_role: Optional[str] = None,
    ):
        """Generate cross-model galaxy visualization."""
        galaxy = CrossModelGalaxy(aggregator)
        img = galaxy.build_galaxy(
            color_by=color_by,
            marker_by=marker_by,
            filter_model=filter_model,
            filter_role=filter_role,
        )
        
        if img:
            return {"image_base64": img}
        else:
            raise HTTPException(400, "Not enough data for galaxy")
    
    @app.post("/api/aggregator/save")
    async def save_aggregator():
        """Save aggregator to disk."""
        aggregator.save()
        return {"status": "saved", "path": aggregator.save_path}
    
    @app.post("/api/aggregator/load")
    async def load_aggregator():
        """Load aggregator from disk."""
        aggregator.load()
        return {"status": "loaded", "experiments": len(aggregator.experiments)}


# =============================================================================
# CLI / DEMO
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    print("Cross-Model Galaxy Demo")
    print("=" * 50)
    
    # Create aggregator
    agg = ExperimentAggregator()
    
    # Add some mock experiments
    mock_experiments = [
        {
            "experiment_id": "exp_001",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "topic": "AI Regulation",
            "signatures": {
                "judge": {"mean_score": 0.1, "std_score": 0.2, "scores": [0.1, 0.15, 0.05], "is_injected": False},
                "plaintiff_counsel": {"mean_score": 1.5, "std_score": 0.3, "scores": [1.2, 1.5, 1.8], "is_injected": True},
                "defense_counsel": {"mean_score": -0.5, "std_score": 0.2, "scores": [-0.3, -0.5, -0.7], "is_injected": False},
            },
            "injection_target": "plaintiff_counsel",
            "injection_strength": 1.0,
        },
        {
            "experiment_id": "exp_002", 
            "model_name": "meta-llama/Llama-2-7b",
            "topic": "AI Regulation",
            "signatures": {
                "judge": {"mean_score": 0.2, "std_score": 0.15, "scores": [0.15, 0.2, 0.25], "is_injected": False},
                "plaintiff_counsel": {"mean_score": 2.0, "std_score": 0.4, "scores": [1.8, 2.0, 2.2], "is_injected": True},
                "defense_counsel": {"mean_score": -0.8, "std_score": 0.25, "scores": [-0.6, -0.8, -1.0], "is_injected": False},
            },
            "injection_target": "plaintiff_counsel",
            "injection_strength": 1.0,
        },
        {
            "experiment_id": "exp_003",
            "model_name": "mistralai/Mistral-7B-Instruct",
            "topic": "AI Regulation",
            "signatures": {
                "judge": {"mean_score": -0.1, "std_score": 0.1, "scores": [-0.05, -0.1, -0.15], "is_injected": False},
                "plaintiff_counsel": {"mean_score": 1.8, "std_score": 0.35, "scores": [1.5, 1.8, 2.1], "is_injected": True},
                "defense_counsel": {"mean_score": -0.3, "std_score": 0.15, "scores": [-0.2, -0.3, -0.4], "is_injected": False},
            },
            "injection_target": "plaintiff_counsel",
            "injection_strength": 1.0,
        },
    ]
    
    for exp in mock_experiments:
        agg.add_experiment(**exp)
    
    print(f"\nAggregator Summary:")
    print(json.dumps(agg.summary(), indent=2))
    
    # Generate galaxy
    if HAS_PLOTTING:
        galaxy = CrossModelGalaxy(agg)
        
        # Full galaxy
        img = galaxy.build_galaxy(
            color_by="model",
            marker_by="role",
            title="Cross-Model Court Behavioral Galaxy"
        )
        
        if img:
            # Save to file
            import base64
            with open("cross_model_galaxy.png", "wb") as f:
                f.write(base64.b64decode(img))
            print("\nSaved: cross_model_galaxy.png")
        
        # Role-specific comparison
        img_judge = galaxy.build_role_comparison("Judge")
        if img_judge:
            with open("judge_comparison.png", "wb") as f:
                f.write(base64.b64decode(img_judge))
            print("Saved: judge_comparison.png")
    
    print("\nDone!")