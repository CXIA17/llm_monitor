#!/usr/bin/env python3
"""
PAS DNA Galaxy & Injection Predictor
=====================================

Enhanced DNA analysis featuring:
1. DNA Galaxy - 2D scatter visualization of behavioral signatures
2. Injection Predictor - ML-based detection of injected agents
3. Fine-grained behavioral breakdown
4. Comparative analysis tools

Integrates with the PAS dashboard.
"""

import numpy as np
import json
import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# Optional imports
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    import matplotlib.patheffects as path_effects
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# DNA SIGNATURE (Enhanced)
# =============================================================================

@dataclass
class EnhancedDNASignature:
    """Enhanced DNA signature with fine-grained behavioral breakdown."""
    name: str
    agent_role: str
    is_injected: bool
    injection_strength: float
    
    # Core vector (32D)
    vector: np.ndarray
    
    # Fine-grained behavioral components
    confidence_profile: Dict[str, float] = field(default_factory=dict)
    temporal_dynamics: Dict[str, float] = field(default_factory=dict)
    linguistic_markers: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    num_rounds: int = 0
    mean_score: float = 0.0
    score_trajectory: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "agent_role": self.agent_role,
            "is_injected": self.is_injected,
            "injection_strength": self.injection_strength,
            "vector": self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector,
            "confidence_profile": self.confidence_profile,
            "temporal_dynamics": self.temporal_dynamics,
            "linguistic_markers": self.linguistic_markers,
            "num_rounds": self.num_rounds,
            "mean_score": self.mean_score,
            "score_trajectory": self.score_trajectory,
        }


# =============================================================================
# DNA GALAXY BUILDER
# =============================================================================

class DNAGalaxyBuilder:
    """
    Build 2D "Galaxy" visualizations of DNA signatures.
    
    Uses dimensionality reduction (PCA, t-SNE) to project
    high-dimensional DNA vectors into 2D space for visualization.
    """
    
    # Color scheme
    COLORS = {
        'injected': '#ef4444',       # Red
        'baseline': '#22c55e',        # Green
        'judge': '#f59e0b',           # Amber
        'prosecutor': '#ef4444',      # Red
        'prosecution': '#ef4444',     # Red
        'defense': '#3b82f6',         # Blue
        'defense_attorney': '#3b82f6',
        'witness': '#8b5cf6',         # Purple
        'background': '#0f172a',      # Dark blue
        'grid': '#334155',            # Slate
        'text': '#f1f5f9',            # Light
        'card': '#1e293b',
    }
    
    def __init__(self, method: str = "pca"):
        self.method = method
        
    def build_galaxy(
        self,
        signatures: Dict[str, EnhancedDNASignature],
        title: str = "DNA Galaxy",
        highlight_injected: bool = True,
    ) -> Optional[str]:
        """Build galaxy visualization and return as base64 PNG."""
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            return None
        
        if len(signatures) < 2:
            return None
        
        # Extract data
        names = list(signatures.keys())
        vectors = np.array([sig.vector for sig in signatures.values()])
        
        # Standardize
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors)
        
        # Dimensionality reduction
        if self.method == "tsne" and len(names) > 3:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(names)-1))
            coords = reducer.fit_transform(vectors_scaled)
        else:
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(vectors_scaled)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), facecolor=self.COLORS['background'])
        ax.set_facecolor(self.COLORS['background'])
        
        # Add grid
        ax.grid(True, color=self.COLORS['grid'], alpha=0.3, linestyle='--')
        
        # Plot each point
        for i, (name, sig) in enumerate(signatures.items()):
            x, y = coords[i]
            
            # Color by role
            role_lower = sig.agent_role.lower()
            color = self.COLORS.get(role_lower, self.COLORS['baseline'])
            
            # Injected style
            if sig.is_injected:
                marker = '*'
                size = 400
                edgewidth = 2
            else:
                marker = 'o'
                size = 250
                edgewidth = 1.5
            
            # Glow effect for injected
            if sig.is_injected:
                ax.scatter(x, y, c=color, s=size*2, alpha=0.2, marker=marker)
            
            # Main point
            ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.9,
                      edgecolors='white', linewidths=edgewidth, zorder=10)
            
            # Label
            label = sig.agent_role.replace('_', ' ').title()
            if sig.is_injected:
                label += ' ⚡'
            
            text = ax.annotate(
                label, (x, y),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, color=self.COLORS['text'],
                fontweight='bold' if sig.is_injected else 'normal',
            )
            text.set_path_effects([
                path_effects.withStroke(linewidth=3, foreground=self.COLORS['background'])
            ])
            
            # Score
            ax.annotate(
                f"Score: {sig.mean_score:.3f}", (x, y),
                xytext=(10, -8), textcoords='offset points',
                fontsize=9, color=self.COLORS['text'], alpha=0.7,
            )
        
        # Draw connections between injected and baseline
        role_groups = defaultdict(list)
        for i, sig in enumerate(signatures.values()):
            role_groups[sig.agent_role].append((i, sig.is_injected))
        
        for role, members in role_groups.items():
            injected = [m[0] for m in members if m[1]]
            baseline = [m[0] for m in members if not m[1]]
            for inj in injected:
                for base in baseline:
                    ax.plot(
                        [coords[inj][0], coords[base][0]],
                        [coords[inj][1], coords[base][1]],
                        color='white', alpha=0.3, linestyle='--', linewidth=1.5, zorder=1
                    )
        
        ax.set_title(title, fontsize=16, fontweight='bold', color=self.COLORS['text'], pad=20)
        ax.set_xlabel("Behavioral Dimension 1", fontsize=11, color=self.COLORS['text'])
        ax.set_ylabel("Behavioral Dimension 2", fontsize=11, color=self.COLORS['text'])
        
        for spine in ax.spines.values():
            spine.set_color(self.COLORS['grid'])
        ax.tick_params(colors=self.COLORS['text'])
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLORS['baseline'],
                  markersize=12, label='Baseline'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=self.COLORS['injected'],
                  markersize=15, label='Injected ⚡'),
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                 facecolor=self.COLORS['card'], edgecolor=self.COLORS['grid'],
                 labelcolor=self.COLORS['text'])
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor=self.COLORS['background'])
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64


# =============================================================================
# INJECTION PREDICTOR
# =============================================================================

class InjectionPredictor:
    """
    ML-based predictor to detect if an agent has been behaviorally injected.
    """
    
    FEATURE_NAMES = [
        "mean_score", "score_std", "score_range", "drift",
        "token_mean", "token_std", "token_p25", "token_p75",
        "temporal_change", "prop_increasing", "max_delta", "min_delta",
        "mean_inj_strength", "is_injected_flag",
        "num_rounds", "signal_to_noise",
    ]
    
    def __init__(self, method: str = "logistic"):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.is_trained = False
    
    def train(self, signatures: Dict[str, EnhancedDNASignature]) -> Dict[str, Any]:
        """Train the injection predictor."""
        if not HAS_SKLEARN:
            return {"error": "sklearn not available"}
        
        if len(signatures) < 4:
            return {"error": "Need at least 4 signatures"}
        
        X = np.array([sig.vector for sig in signatures.values()])
        y = np.array([1 if sig.is_injected else 0 for sig in signatures.values()])
        
        n_injected = np.sum(y)
        n_baseline = len(y) - n_injected
        
        if n_injected == 0 or n_baseline == 0:
            return {"error": "Need both classes"}
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == "random_forest":
            self.model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        else:
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        accuracy = accuracy_score(y, y_pred)
        try:
            auc = roc_auc_score(y, y_prob)
        except:
            auc = 0.5
        
        if self.method == "random_forest":
            self.feature_importance = self.model.feature_importances_
        else:
            self.feature_importance = np.abs(self.model.coef_[0])
        
        self.feature_importance = self.feature_importance / (self.feature_importance.sum() + 1e-8)
        
        return {
            "accuracy": float(accuracy),
            "auc_roc": float(auc),
            "n_samples": len(y),
            "n_injected": int(n_injected),
            "n_baseline": int(n_baseline),
            "top_features": self._get_top_features(5),
        }
    
    def predict(self, signature: EnhancedDNASignature) -> Dict[str, Any]:
        """Predict if signature is from injected agent."""
        if not self.is_trained:
            return {"error": "Not trained"}
        
        X = signature.vector.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        pred = self.model.predict(X_scaled)[0]
        prob = self.model.predict_proba(X_scaled)[0]
        
        return {
            "predicted_injected": bool(pred),
            "probability_injected": float(prob[1]),
            "confidence": float(max(prob)),
            "actual_injected": signature.is_injected,
            "correct": bool(pred) == signature.is_injected,
        }
    
    def _get_top_features(self, k: int = 5) -> List[Dict]:
        if self.feature_importance is None:
            return []
        
        indices = np.argsort(self.feature_importance)[::-1][:k]
        return [
            {
                "feature": self.FEATURE_NAMES[i] if i < len(self.FEATURE_NAMES) else f"dim_{i}",
                "importance": float(self.feature_importance[i]),
            }
            for i in indices
        ]
    
    def visualize_importance(self, title: str = "Feature Importance") -> Optional[str]:
        """Visualize feature importance."""
        if not HAS_MATPLOTLIB or self.feature_importance is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        
        n_features = min(10, len(self.feature_importance))
        indices = np.argsort(self.feature_importance)[::-1][:n_features]
        
        names = [self.FEATURE_NAMES[i] if i < len(self.FEATURE_NAMES) else f"dim_{i}" for i in indices]
        values = [self.feature_importance[i] for i in indices]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_features))
        
        ax.barh(range(n_features), values, color=colors)
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(names, color='#f1f5f9')
        ax.invert_yaxis()
        
        ax.set_xlabel("Importance", color='#f1f5f9')
        ax.set_title(title, fontsize=14, fontweight='bold', color='#f1f5f9', pad=15)
        
        ax.tick_params(colors='#f1f5f9')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64


# =============================================================================
# FINE-GRAINED ANALYZER
# =============================================================================

class FineGrainedAnalyzer:
    """Detailed DNA analysis."""
    
    DIMENSIONS = {
        "confidence": {"indices": [0, 1, 2, 3], "desc": "Certainty levels"},
        "stability": {"indices": [4, 5, 6, 7], "desc": "Response consistency"},
        "dynamics": {"indices": [8, 9, 10, 11], "desc": "Temporal changes"},
        "injection_sensitivity": {"indices": [12, 13], "desc": "Injection effects"},
    }
    
    def analyze(self, signature: EnhancedDNASignature) -> Dict[str, Any]:
        """Analyze a single signature."""
        vector = signature.vector
        
        result = {
            "agent": signature.agent_role,
            "is_injected": signature.is_injected,
            "mean_score": float(signature.mean_score),
            "dimensions": {},
        }
        
        for dim_name, info in self.DIMENSIONS.items():
            indices = [i for i in info["indices"] if i < len(vector)]
            if indices:
                vals = vector[indices]
                result["dimensions"][dim_name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "description": info["desc"],
                }
        
        return result
    
    def find_injection_markers(self, signatures: Dict[str, EnhancedDNASignature]) -> Dict[str, Any]:
        """Find which dimensions change most with injection."""
        injected = [s for s in signatures.values() if s.is_injected]
        baseline = [s for s in signatures.values() if not s.is_injected]
        
        if not injected or not baseline:
            return {"error": "Need both classes"}
        
        inj_mean = np.mean([s.vector for s in injected], axis=0)
        base_mean = np.mean([s.vector for s in baseline], axis=0)
        diff = inj_mean - base_mean
        
        markers = {"overall_shift": float(np.linalg.norm(diff)), "dimension_shifts": {}}
        
        for dim_name, info in self.DIMENSIONS.items():
            indices = [i for i in info["indices"] if i < len(diff)]
            if indices:
                dim_diff = diff[indices]
                markers["dimension_shifts"][dim_name] = {
                    "mean_shift": float(np.mean(dim_diff)),
                    "abs_shift": float(np.mean(np.abs(dim_diff))),
                    "direction": "increases" if np.mean(dim_diff) > 0 else "decreases",
                }
        
        ranked = sorted(markers["dimension_shifts"].items(), key=lambda x: -x[1]["abs_shift"])
        markers["most_affected"] = [r[0] for r in ranked]
        
        return markers


# =============================================================================
# INTEGRATION
# =============================================================================

class DNAGalaxyIntegration:
    """Integrate DNA Galaxy into dashboard."""
    
    def __init__(self):
        self.galaxy = DNAGalaxyBuilder()
        self.predictor = InjectionPredictor()
        self.analyzer = FineGrainedAnalyzer()
        self.signatures: Dict[str, EnhancedDNASignature] = {}
    
    def extract_signatures(self, round_results: List[Dict], model_name: str = "model") -> Dict[str, EnhancedDNASignature]:
        """Extract enhanced DNA signatures from experiment results."""
        self.signatures = {}
        
        if not round_results:
            return self.signatures
        
        agent_data = defaultdict(lambda: {
            "scores": [], "mean_scores": [], "is_injected": False, "injection_strengths": []
        })
        
        for round_data in round_results:
            for agent_name, data in round_data.get("agents", {}).items():
                agent_data[agent_name]["mean_scores"].append(data.get("mean_score", 0.0))
                agent_data[agent_name]["scores"].extend(data.get("scores", []))
                agent_data[agent_name]["is_injected"] = data.get("is_injected", False)
                agent_data[agent_name]["injection_strengths"].append(data.get("injection_strength", 0.0))
        
        for agent_name, data in agent_data.items():
            if not data["mean_scores"]:
                continue
            
            vector = self._build_vector(data)
            
            sig = EnhancedDNASignature(
                name=f"{model_name}_{agent_name}",
                agent_role=agent_name,
                is_injected=data["is_injected"],
                injection_strength=float(np.mean(data["injection_strengths"])),
                vector=vector,
                num_rounds=len(data["mean_scores"]),
                mean_score=float(np.mean(data["mean_scores"])),
                score_trajectory=data["mean_scores"],
            )
            
            self.signatures[sig.name] = sig
        
        return self.signatures
    
    def _build_vector(self, data: Dict) -> np.ndarray:
        """Build 32D DNA vector."""
        features = []
        
        mean_scores = np.array(data["mean_scores"])
        all_scores = np.array(data["scores"]) if data["scores"] else np.array([0.0])
        inj_strengths = np.array(data["injection_strengths"])
        
        # Score stats (4)
        features.extend([
            float(np.mean(mean_scores)),
            float(np.std(mean_scores)) if len(mean_scores) > 1 else 0.0,
            float(np.max(mean_scores) - np.min(mean_scores)),
            float(mean_scores[-1] - mean_scores[0]) if len(mean_scores) > 1 else 0.0,
        ])
        
        # Token stats (4)
        features.extend([
            float(np.mean(all_scores)),
            float(np.std(all_scores)) if len(all_scores) > 1 else 0.0,
            float(np.percentile(all_scores, 25)) if len(all_scores) > 0 else 0.0,
            float(np.percentile(all_scores, 75)) if len(all_scores) > 0 else 0.0,
        ])
        
        # Temporal (4)
        if len(mean_scores) > 2:
            diff = np.diff(mean_scores)
            features.extend([
                float(np.mean(np.abs(diff))),
                float(np.sum(diff > 0) / len(diff)),
                float(np.max(diff)),
                float(np.min(diff)),
            ])
        else:
            features.extend([0.0, 0.5, 0.0, 0.0])
        
        # Injection (2)
        features.extend([
            float(np.mean(inj_strengths)),
            float(1.0 if data["is_injected"] else 0.0),
        ])
        
        # Meta (2)
        features.extend([
            float(len(mean_scores)),
            float(np.mean(mean_scores) / (np.std(mean_scores) + 1e-6)),
        ])
        
        # Pad to 32
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32], dtype=np.float32)
    
    def generate_analysis(self) -> Dict[str, Any]:
        """Generate complete analysis."""
        results = {
            "n_signatures": len(self.signatures),
            "galaxy_image": None,
            "importance_image": None,
            "predictor_metrics": None,
            "predictions": {},
            "injection_markers": None,
            "agent_analysis": {},
        }
        
        if len(self.signatures) < 2:
            return results
        
        # Galaxy
        results["galaxy_image"] = self.galaxy.build_galaxy(
            self.signatures, title="🌌 DNA Galaxy: Agent Behavioral Clustering"
        )
        
        # Predictor
        results["predictor_metrics"] = self.predictor.train(self.signatures)
        
        if self.predictor.is_trained:
            results["importance_image"] = self.predictor.visualize_importance(
                "🎯 Injection Detection: Feature Importance"
            )
            
            for name, sig in self.signatures.items():
                results["predictions"][name] = self.predictor.predict(sig)
        
        # Markers
        results["injection_markers"] = self.analyzer.find_injection_markers(self.signatures)
        
        # Per-agent
        for name, sig in self.signatures.items():
            results["agent_analysis"][name] = self.analyzer.analyze(sig)
        
        return results


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'EnhancedDNASignature',
    'DNAGalaxyBuilder',
    'InjectionPredictor',
    'FineGrainedAnalyzer',
    'DNAGalaxyIntegration',
]


if __name__ == "__main__":
    print("DNA Galaxy & Injection Predictor")
    print("=" * 40)
    
    # Test
    np.random.seed(42)
    
    test_sigs = {}
    for agent in ["judge", "prosecutor", "defense"]:
        base_vec = np.random.randn(32).astype(np.float32) * 0.5
        test_sigs[f"{agent}_baseline"] = EnhancedDNASignature(
            name=f"test_{agent}_baseline",
            agent_role=agent,
            is_injected=False,
            injection_strength=0.0,
            vector=base_vec,
            mean_score=float(np.random.uniform(0.3, 0.5)),
        )
        
        inj_vec = base_vec + np.random.randn(32).astype(np.float32) * 0.3 + 0.5
        test_sigs[f"{agent}_injected"] = EnhancedDNASignature(
            name=f"test_{agent}_injected",
            agent_role=agent,
            is_injected=True,
            injection_strength=3.0,
            vector=inj_vec,
            mean_score=float(np.random.uniform(0.6, 0.9)),
        )
    
    integration = DNAGalaxyIntegration()
    integration.signatures = test_sigs
    
    results = integration.generate_analysis()
    
    print(f"\nSignatures: {results['n_signatures']}")
    print(f"Galaxy: {'✓' if results['galaxy_image'] else '✗'}")
    print(f"Predictor: {results.get('predictor_metrics', {})}")
    
    if results['injection_markers']:
        print(f"Most affected: {results['injection_markers'].get('most_affected', [])}")