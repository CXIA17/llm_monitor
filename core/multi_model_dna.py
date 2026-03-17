#!/usr/bin/env python3
"""
Multi-Model DNA Integration System
===================================

This module enables cross-model phylogenetic analysis by:
1. Normalizing probe scores into comparable "Steerability Ratios"
2. Aligning feature vectors across different model experiments
3. Building combined phylogenetic trees from multiple models

The key insight: raw probe scores are NOT comparable across models.
A "5-unit shift" means something very different for a stable model vs a chaotic one.
We use Z-score normalization to convert everything to "sigmas" (standard deviations).

Architecture:
                                                                    
    Model A Experiments          Model B Experiments                 
    ┌─────────────────┐          ┌─────────────────┐                 
    │ baseline.json   │          │ baseline.json   │                 
    │ gated_3.0.json  │          │ gated_3.0.json  │                 
    │ gated_4.0.json  │          │ gated_4.0.json  │                 
    └────────┬────────┘          └────────┬────────┘                 
             │                            │                          
             ▼                            ▼                          
    ┌─────────────────┐          ┌─────────────────┐                 
    │ Z-Score Norm    │          │ Z-Score Norm    │                 
    │ shift / σ_base  │          │ shift / σ_base  │                 
    └────────┬────────┘          └────────┬────────┘                 
             │                            │                          
             └──────────┬─────────────────┘                          
                        ▼                                            
              ┌─────────────────────┐                                
              │   Feature Alignment │                                
              │   (Same dimensions) │                                
              └──────────┬──────────┘                                
                         │                                           
                         ▼                                           
              ┌─────────────────────┐                                
              │  Distance Matrix    │                                
              │  (Cosine/Euclidean) │                                
              └──────────┬──────────┘                                
                         │                                           
                         ▼                                           
              ┌─────────────────────┐                                
              │  Phylogenetic Tree  │                                
              │  (Hierarchical)     │                                
              └─────────────────────┘                                

Usage:
    from core.multi_model_dna import MultiModelDNAIntegrator
    
    integrator = MultiModelDNAIntegrator()
    
    # Add experiments from different models
    integrator.add_model_experiments(
        model_name="Llama-3-8B",
        experiment_dir="results/llama3/",
        probe_categories=["overconfidence", "sycophancy", "toxicity"]
    )
    integrator.add_model_experiments(
        model_name="Qwen-2.5-7B", 
        experiment_dir="results/qwen/",
        probe_categories=["overconfidence", "sycophancy", "toxicity"]
    )
    
    # Generate combined tree
    integrator.build_phylogenetic_tree("combined_tree.png")
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict

# Optional imports
try:
    from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Tree generation will be limited.")

try:
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ModelDNA:
    """
    DNA signature for a single model configuration.
    
    The DNA vector contains normalized "steerability scores" for each probe category.
    Each score represents: "How many standard deviations did the model shift 
    when we applied intervention?"
    """
    model_id: str                              # e.g., "Llama-3-8B"
    config_id: str                             # e.g., "baseline", "gated_3.0"
    features: Dict[str, float] = field(default_factory=dict)
    
    # Metadata for analysis
    raw_scores: Dict[str, float] = field(default_factory=dict)      # Before normalization
    baseline_stds: Dict[str, float] = field(default_factory=dict)   # σ for each feature
    
    @property
    def full_id(self) -> str:
        """Unique identifier for tree labels."""
        return f"{self.model_id}_{self.config_id}"
    
    def to_vector(self, feature_order: List[str]) -> np.ndarray:
        """Convert to numpy vector with consistent feature ordering."""
        return np.array([self.features.get(f, 0.0) for f in feature_order])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "config_id": self.config_id,
            "features": self.features,
            "raw_scores": self.raw_scores,
            "baseline_stds": self.baseline_stds,
        }


@dataclass
class ExperimentMetrics:
    """Extracted metrics from a single experiment run."""
    model_id: str
    config_id: str                             # e.g., "baseline", "gated_3.0"
    probe_category: str
    
    # Core metrics
    mean_score: float                          # Average probe score across agents
    score_std: float                           # Standard deviation of token-level scores
    score_trajectory: List[float]              # Score progression over rounds
    
    # Per-agent breakdown
    agent_scores: Dict[str, float] = field(default_factory=dict)
    
    # Shadow/Ghost data (if available)
    ghost_positive_score: Optional[float] = None
    ghost_negative_score: Optional[float] = None


# =============================================================================
# METRIC EXTRACTION
# =============================================================================

def extract_metrics_from_json(
    json_path: str,
    model_id: str,
    config_id: str,
    probe_category: str = "default"
) -> ExperimentMetrics:
    """
    Extract standardized metrics from an experiment JSON file.
    
    Handles both single-experiment format and comparison format
    (with "baseline" and "injected" keys).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle comparison format
    if "baseline" in data and "injected" in data:
        # Return metrics for the injected run (the one we're analyzing)
        data = data["injected"]
    
    # Extract per-agent scores
    agent_scores = {}
    all_token_scores = []
    score_trajectories = []
    
    for agent_id, metrics in data.get("agent_metrics", {}).items():
        agent_scores[agent_id] = metrics.get("mean_score", 0.0)
        
        # Collect all token-level scores for std calculation
        for round_scores in metrics.get("token_scores", []):
            if isinstance(round_scores, list):
                all_token_scores.extend(round_scores)
        
        # Collect trajectory
        score_trajectories.append(metrics.get("probe_scores", []))
    
    # Calculate aggregate metrics
    mean_score = np.mean(list(agent_scores.values())) if agent_scores else 0.0
    score_std = np.std(all_token_scores) if all_token_scores else 1.0
    
    # Flatten trajectories to single list (average across agents per round)
    if score_trajectories:
        max_rounds = max(len(t) for t in score_trajectories)
        trajectory = []
        for r in range(max_rounds):
            round_scores = [t[r] for t in score_trajectories if r < len(t)]
            trajectory.append(np.mean(round_scores) if round_scores else 0.0)
    else:
        trajectory = []
    
    # Extract shadow/ghost data if available
    ghost_pos = None
    ghost_neg = None
    shadow_logs = data.get("shadow_logs", {})
    if shadow_logs:
        # Average ghost scores across all rounds and agents
        pos_scores = []
        neg_scores = []
        for round_data in shadow_logs.values():
            if isinstance(round_data, dict):
                for agent_shadow in round_data.values():
                    if isinstance(agent_shadow, dict):
                        if "ghost_positive_score" in agent_shadow:
                            pos_scores.append(agent_shadow["ghost_positive_score"])
                        if "ghost_negative_score" in agent_shadow:
                            neg_scores.append(agent_shadow["ghost_negative_score"])
        
        ghost_pos = np.mean(pos_scores) if pos_scores else None
        ghost_neg = np.mean(neg_scores) if neg_scores else None
    
    return ExperimentMetrics(
        model_id=model_id,
        config_id=config_id,
        probe_category=probe_category,
        mean_score=mean_score,
        score_std=score_std,
        score_trajectory=trajectory,
        agent_scores=agent_scores,
        ghost_positive_score=ghost_pos,
        ghost_negative_score=ghost_neg,
    )


# =============================================================================
# NORMALIZATION STRATEGIES
# =============================================================================

class NormalizationStrategy:
    """Base class for score normalization strategies."""
    
    def normalize(
        self,
        baseline_metrics: ExperimentMetrics,
        injected_metrics: ExperimentMetrics
    ) -> float:
        """
        Convert raw score shift to normalized DNA score.
        
        Returns a value that is comparable across different models.
        """
        raise NotImplementedError


class ZScoreNormalization(NormalizationStrategy):
    """
    Z-Score normalization: shift / baseline_std
    
    Converts the raw shift into "number of standard deviations."
    This is the most robust method for cross-model comparison because:
    - A stable model (low σ) will have large Z-scores for small movements
    - A chaotic model (high σ) will have small Z-scores for the same movement
    """
    
    def normalize(
        self,
        baseline_metrics: ExperimentMetrics,
        injected_metrics: ExperimentMetrics
    ) -> float:
        raw_shift = injected_metrics.mean_score - baseline_metrics.mean_score
        sigma = baseline_metrics.score_std
        
        # Avoid division by zero
        if sigma < 1e-6:
            sigma = 1.0
        
        return raw_shift / sigma


class GhostBoundaryNormalization(NormalizationStrategy):
    """
    Ghost boundary normalization: shift / (ghost_pos - ghost_neg)
    
    Normalizes by the model's "steering range" - the maximum possible
    behavioral span as measured by ghost responses.
    
    This is theoretically ideal but requires ghost scores in the data.
    """
    
    def normalize(
        self,
        baseline_metrics: ExperimentMetrics,
        injected_metrics: ExperimentMetrics
    ) -> float:
        raw_shift = injected_metrics.mean_score - baseline_metrics.mean_score
        
        # Try to use ghost boundaries
        ghost_pos = injected_metrics.ghost_positive_score
        ghost_neg = injected_metrics.ghost_negative_score
        
        if ghost_pos is not None and ghost_neg is not None:
            ghost_range = abs(ghost_pos - ghost_neg)
            if ghost_range > 1e-6:
                return raw_shift / ghost_range
        
        # Fallback to Z-score if ghost data unavailable
        return ZScoreNormalization().normalize(baseline_metrics, injected_metrics)


class MinMaxNormalization(NormalizationStrategy):
    """
    Min-Max normalization to [0, 1] range.
    
    Uses observed min/max scores as boundaries.
    Less robust than Z-score but useful for visualization.
    """
    
    def __init__(self, global_min: float = -10.0, global_max: float = 10.0):
        self.global_min = global_min
        self.global_max = global_max
    
    def normalize(
        self,
        baseline_metrics: ExperimentMetrics,
        injected_metrics: ExperimentMetrics
    ) -> float:
        raw_shift = injected_metrics.mean_score - baseline_metrics.mean_score
        
        # Normalize to [0, 1] based on global expected range
        normalized = (raw_shift - self.global_min) / (self.global_max - self.global_min)
        return np.clip(normalized, 0.0, 1.0)


# =============================================================================
# MULTI-MODEL DNA INTEGRATOR
# =============================================================================

class MultiModelDNAIntegrator:
    """
    Main class for integrating DNA from multiple LLM models.
    
    This class handles:
    1. Loading experiments from multiple models
    2. Normalizing scores for cross-model comparison
    3. Aligning feature vectors
    4. Building combined phylogenetic trees
    
    Example:
        integrator = MultiModelDNAIntegrator()
        
        # Add model experiments
        integrator.add_model_experiments("Llama-3-8B", "results/llama3/")
        integrator.add_model_experiments("Qwen-2.5-7B", "results/qwen/")
        integrator.add_model_experiments("Gemma-7B", "results/gemma/")
        
        # Build tree
        integrator.build_phylogenetic_tree("llm_family_tree.png")
    """
    
    def __init__(
        self,
        normalization: str = "zscore",  # "zscore", "ghost", "minmax"
        distance_metric: str = "cosine",  # "cosine", "euclidean", "manhattan"
    ):
        self.dna_registry: List[ModelDNA] = []
        self.all_features: set = set()
        
        # Set normalization strategy
        if normalization == "zscore":
            self.normalizer = ZScoreNormalization()
        elif normalization == "ghost":
            self.normalizer = GhostBoundaryNormalization()
        elif normalization == "minmax":
            self.normalizer = MinMaxNormalization()
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        self.distance_metric = distance_metric
        
        # Store raw experiments for analysis
        self._experiments: Dict[str, Dict[str, ExperimentMetrics]] = defaultdict(dict)
    
    def add_single_experiment(
        self,
        model_id: str,
        baseline_path: str,
        injected_path: str,
        probe_category: str,
        config_id: str = "default"
    ):
        """
        Add a single experiment (baseline + injected pair) for one probe category.
        
        This is the lowest-level method for adding data.
        """
        # Extract metrics
        baseline_metrics = extract_metrics_from_json(
            baseline_path, model_id, "baseline", probe_category
        )
        injected_metrics = extract_metrics_from_json(
            injected_path, model_id, config_id, probe_category
        )
        
        # Normalize
        normalized_score = self.normalizer.normalize(baseline_metrics, injected_metrics)
        
        # Track feature
        self.all_features.add(probe_category)
        
        # Find or create DNA entry
        full_id = f"{model_id}_{config_id}"
        existing = next((d for d in self.dna_registry if d.full_id == full_id), None)
        
        if existing:
            existing.features[probe_category] = normalized_score
            existing.raw_scores[probe_category] = injected_metrics.mean_score - baseline_metrics.mean_score
            existing.baseline_stds[probe_category] = baseline_metrics.score_std
        else:
            dna = ModelDNA(
                model_id=model_id,
                config_id=config_id,
                features={probe_category: normalized_score},
                raw_scores={probe_category: injected_metrics.mean_score - baseline_metrics.mean_score},
                baseline_stds={probe_category: baseline_metrics.score_std},
            )
            self.dna_registry.append(dna)
        
        # Store for analysis
        self._experiments[full_id][probe_category] = {
            "baseline": baseline_metrics,
            "injected": injected_metrics,
            "normalized": normalized_score,
        }
        
        print(f"[{model_id}/{config_id}] {probe_category}: "
              f"raw_shift={injected_metrics.mean_score - baseline_metrics.mean_score:.4f}, "
              f"σ={baseline_metrics.score_std:.4f}, "
              f"DNA={normalized_score:.4f}")
    
    def add_model_experiments(
        self,
        model_id: str,
        experiment_dir: str,
        probe_categories: Optional[List[str]] = None,
        injection_strengths: List[float] = [3.0, 4.0],
    ):
        """
        Add all experiments from a model's output directory.
        
        Expected directory structure:
            experiment_dir/
            ├── results/
            │   ├── {model_safe}_baseline.json
            │   ├── {model_safe}_gated_3.0.json
            │   └── {model_safe}_gated_4.0.json
            └── probes/
                └── {model_safe}_probes.pkl
        
        Args:
            model_id: Human-readable model name (e.g., "Llama-3-8B")
            experiment_dir: Path to experiment output directory
            probe_categories: List of probe categories to include (auto-detect if None)
            injection_strengths: Which injection strengths to process
        """
        exp_dir = Path(experiment_dir)
        results_dir = exp_dir / "results"
        
        if not results_dir.exists():
            # Maybe results are directly in exp_dir
            results_dir = exp_dir
        
        # Find baseline file
        baseline_files = list(results_dir.glob("*baseline*.json"))
        if not baseline_files:
            print(f"Warning: No baseline file found in {results_dir}")
            return
        
        baseline_path = baseline_files[0]
        
        # Find injection files
        for strength in injection_strengths:
            # Try different naming patterns
            patterns = [
                f"*gated_{strength}*.json",
                f"*gated_{strength:.1f}*.json",
                f"*injection_{strength}*.json",
                f"*strength_{strength}*.json",
            ]
            
            injection_file = None
            for pattern in patterns:
                matches = list(results_dir.glob(pattern))
                if matches:
                    injection_file = matches[0]
                    break
            
            if injection_file is None:
                print(f"Warning: No injection file found for strength {strength}")
                continue
            
            # Detect probe categories from the files
            if probe_categories is None:
                with open(baseline_path, 'r') as f:
                    data = json.load(f)
                # Try to infer from comparison structure
                if "comparison" in data:
                    detected = [data["comparison"].get("probe_category", "default")]
                else:
                    detected = ["default"]
                categories = detected
            else:
                categories = probe_categories
            
            # Add experiment for each category
            for category in categories:
                config_id = f"gated_{strength}"
                self.add_single_experiment(
                    model_id=model_id,
                    baseline_path=str(baseline_path),
                    injected_path=str(injection_file),
                    probe_category=category,
                    config_id=config_id,
                )
    
    def add_from_comparison_json(
        self,
        model_id: str,
        comparison_path: str,
        probe_category: str = "default"
    ):
        """
        Add experiment from a comparison JSON file (contains both baseline and injected).
        
        This is the format produced by run_comparison() in the orchestrator.
        """
        with open(comparison_path, 'r') as f:
            data = json.load(f)
        
        if "baseline" not in data or "injected" not in data:
            raise ValueError(f"File {comparison_path} is not a comparison format")
        
        # Extract config from comparison metadata
        comparison_meta = data.get("comparison", {})
        strength = comparison_meta.get("strength", 1.0)
        injection_type = comparison_meta.get("type", "gated")
        config_id = f"{injection_type}_{strength}"
        
        # Extract metrics manually
        baseline_data = data["baseline"]
        injected_data = data["injected"]
        
        # Calculate metrics for baseline
        baseline_scores = []
        baseline_token_scores = []
        for agent_metrics in baseline_data.get("agent_metrics", {}).values():
            baseline_scores.append(agent_metrics.get("mean_score", 0.0))
            for round_scores in agent_metrics.get("token_scores", []):
                if isinstance(round_scores, list):
                    baseline_token_scores.extend(round_scores)
        
        baseline_mean = np.mean(baseline_scores) if baseline_scores else 0.0
        baseline_std = np.std(baseline_token_scores) if baseline_token_scores else 1.0
        
        # Calculate metrics for injected
        injected_scores = []
        for agent_metrics in injected_data.get("agent_metrics", {}).values():
            injected_scores.append(agent_metrics.get("mean_score", 0.0))
        
        injected_mean = np.mean(injected_scores) if injected_scores else 0.0
        
        # Calculate normalized score
        raw_shift = injected_mean - baseline_mean
        sigma = baseline_std if baseline_std > 1e-6 else 1.0
        normalized_score = raw_shift / sigma
        
        # Track feature
        self.all_features.add(probe_category)
        
        # Find or create DNA entry
        full_id = f"{model_id}_{config_id}"
        existing = next((d for d in self.dna_registry if d.full_id == full_id), None)
        
        if existing:
            existing.features[probe_category] = normalized_score
            existing.raw_scores[probe_category] = raw_shift
            existing.baseline_stds[probe_category] = sigma
        else:
            dna = ModelDNA(
                model_id=model_id,
                config_id=config_id,
                features={probe_category: normalized_score},
                raw_scores={probe_category: raw_shift},
                baseline_stds={probe_category: sigma},
            )
            self.dna_registry.append(dna)
        
        print(f"[{model_id}/{config_id}] {probe_category}: "
              f"raw_shift={raw_shift:.4f}, σ={sigma:.4f}, DNA={normalized_score:.4f}")
    
    def compute_distance_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise distance matrix between all DNA signatures.
        
        Returns:
            distances: NxN distance matrix
            labels: List of model labels in same order
        """
        if not self.dna_registry:
            raise ValueError("No DNA signatures loaded!")
        
        # Get consistent feature ordering
        feature_order = sorted(list(self.all_features))
        
        # Build matrix
        labels = []
        vectors = []
        
        for dna in self.dna_registry:
            labels.append(dna.full_id)
            vectors.append(dna.to_vector(feature_order))
        
        X = np.array(vectors)
        
        # Compute distances
        if self.distance_metric == "cosine":
            if HAS_SKLEARN:
                distances = cosine_distances(X)
            else:
                # Manual cosine distance
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                X_normalized = X / norms
                similarities = X_normalized @ X_normalized.T
                distances = 1 - similarities
        elif self.distance_metric == "euclidean":
            if HAS_SKLEARN:
                distances = euclidean_distances(X)
            else:
                distances = squareform(pdist(X, metric='euclidean'))
        elif self.distance_metric == "manhattan":
            distances = squareform(pdist(X, metric='cityblock'))
        else:
            distances = squareform(pdist(X, metric=self.distance_metric))
        
        return distances, labels
    
    def build_phylogenetic_tree(
        self,
        output_path: str = "phylogenetic_tree.png",
        method: str = "ward",  # "ward", "average", "complete", "single"
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        show_distances: bool = True,
    ):
        """
        Build and save a phylogenetic tree from all loaded DNA signatures.
        
        Args:
            output_path: Where to save the tree image
            method: Linkage method for hierarchical clustering
            title: Custom title for the plot
            figsize: Figure size (width, height)
            show_distances: Whether to show distance labels on branches
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required for tree generation")
        
        if not self.dna_registry:
            raise ValueError("No DNA signatures loaded!")
        
        # Get consistent feature ordering
        feature_order = sorted(list(self.all_features))
        
        # Build matrix
        labels = []
        vectors = []
        
        for dna in self.dna_registry:
            labels.append(dna.full_id)
            vectors.append(dna.to_vector(feature_order))
        
        X = np.array(vectors)
        
        # Perform hierarchical clustering
        if method == "ward":
            # Ward's method requires euclidean distance
            Z = linkage(X, method='ward')
        else:
            # Compute custom distance matrix first
            distances, _ = self.compute_distance_matrix()
            # Convert to condensed form for linkage
            condensed = squareform(distances)
            Z = linkage(condensed, method=method)
        
        # Create figure with two subplots
        fig = plt.figure(figsize=figsize)
        
        # Main dendrogram
        ax1 = fig.add_subplot(121)
        
        dendro = dendrogram(
            Z,
            labels=labels,
            orientation='left',
            leaf_font_size=10,
            ax=ax1,
        )
        
        if title:
            ax1.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax1.set_title(
                f"LLM Behavioral Phylogeny\n({len(feature_order)} markers, {len(labels)} configurations)",
                fontsize=14,
                fontweight='bold'
            )
        
        ax1.set_xlabel(f"Behavioral Distance ({method.capitalize()} Linkage)")
        
        # Feature importance heatmap
        ax2 = fig.add_subplot(122)
        
        # Reorder vectors according to dendrogram
        order = dendro['leaves']
        ordered_labels = [labels[i] for i in order]
        ordered_vectors = X[order]
        
        im = ax2.imshow(ordered_vectors, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
        ax2.set_yticks(range(len(ordered_labels)))
        ax2.set_yticklabels(ordered_labels, fontsize=9)
        ax2.set_xticks(range(len(feature_order)))
        ax2.set_xticklabels(feature_order, rotation=45, ha='right', fontsize=9)
        ax2.set_title("DNA Feature Matrix (Z-scores)", fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.6)
        cbar.set_label("Steerability (σ)", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Phylogenetic tree saved to: {output_path}")
        
        return Z, labels
    
    def build_per_category_trees(
        self,
        output_dir: str,
        method: str = "ward",
        figsize: Tuple[int, int] = (12, 8),
    ) -> Dict[str, str]:
        """
        Build SEPARATE phylogenetic trees for each probe category.
        
        This is the recommended approach because different probe categories
        measure fundamentally different behavioral dimensions. A model might
        be highly steerable on "overconfidence" but resistant on "toxicity".
        
        Args:
            output_dir: Directory to save tree images
            method: Linkage method for hierarchical clustering
            figsize: Figure size for each tree
            
        Returns:
            Dict mapping category names to output file paths
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required for tree generation")
        
        if not self.dna_registry:
            raise ValueError("No DNA signatures loaded!")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        print(f"\nGenerating per-category phylogenetic trees...")
        
        for category in sorted(self.all_features):
            # Collect all models that have this category
            labels = []
            scores = []
            
            for dna in self.dna_registry:
                if category in dna.features:
                    labels.append(dna.full_id)
                    scores.append(dna.features[category])
            
            if len(labels) < 2:
                print(f"  Skipping {category}: need at least 2 models (found {len(labels)})")
                continue
            
            # Build tree for this single dimension
            X = np.array(scores).reshape(-1, 1)
            
            # Perform hierarchical clustering
            Z = linkage(X, method=method)
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=figsize, 
                                     gridspec_kw={'width_ratios': [2, 1]})
            
            # Dendrogram
            ax1 = axes[0]
            dendro = dendrogram(
                Z,
                labels=labels,
                orientation='left',
                leaf_font_size=10,
                ax=ax1,
            )
            
            ax1.set_title(
                f"{category.upper()} Steerability Phylogeny\n({len(labels)} configurations)",
                fontsize=14,
                fontweight='bold'
            )
            ax1.set_xlabel(f"Distance ({method.capitalize()} Linkage)")
            
            # Bar chart of scores (ordered by dendrogram)
            ax2 = axes[1]
            order = dendro['leaves']
            ordered_labels = [labels[i] for i in order]
            ordered_scores = [scores[i] for i in order]
            
            # Color by sign and magnitude
            colors = []
            for s in ordered_scores:
                if s >= 2.0:
                    colors.append('#d62728')  # Strong positive (red)
                elif s >= 1.0:
                    colors.append('#ff7f0e')  # Moderate positive (orange)
                elif s >= 0:
                    colors.append('#2ca02c')  # Weak positive (green)
                elif s >= -1.0:
                    colors.append('#17becf')  # Weak negative (cyan)
                else:
                    colors.append('#1f77b4')  # Strong negative (blue)
            
            y_pos = range(len(ordered_labels))
            ax2.barh(y_pos, ordered_scores, color=colors, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(ordered_labels, fontsize=9)
            ax2.set_xlabel("Steerability (σ)")
            ax2.set_title("Z-Score Values", fontsize=12)
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax2.axvline(x=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax2.axvline(x=-1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax2.grid(axis='x', alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d62728', alpha=0.7, label='Strong + (≥2σ)'),
                Patch(facecolor='#ff7f0e', alpha=0.7, label='Moderate + (1-2σ)'),
                Patch(facecolor='#2ca02c', alpha=0.7, label='Weak + (0-1σ)'),
                Patch(facecolor='#17becf', alpha=0.7, label='Weak - (0 to -1σ)'),
                Patch(facecolor='#1f77b4', alpha=0.7, label='Strong - (<-1σ)'),
            ]
            ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)
            
            plt.tight_layout()
            
            # Save
            output_path = output_dir / f"tree_{category}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            output_files[category] = str(output_path)
            print(f"  ✓ {category} tree saved to: {output_path}")
        
        return output_files
    
    def build_category_comparison_matrix(
        self,
        output_path: str,
        figsize: Tuple[int, int] = (14, 10),
    ):
        """
        Build a comparison matrix showing all models × all categories.
        
        This provides a bird's-eye view of which models are steerable on which
        dimensions, using a heatmap visualization.
        """
        if not self.dna_registry:
            raise ValueError("No DNA signatures loaded!")
        
        # Get consistent ordering
        feature_order = sorted(list(self.all_features))
        model_order = sorted([dna.full_id for dna in self.dna_registry])
        
        # Build matrix
        matrix = np.zeros((len(model_order), len(feature_order)))
        
        for dna in self.dna_registry:
            row_idx = model_order.index(dna.full_id)
            for col_idx, feature in enumerate(feature_order):
                matrix[row_idx, col_idx] = dna.features.get(feature, 0.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Heatmap
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
        
        # Labels
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(model_order, fontsize=9)
        ax.set_xticks(range(len(feature_order)))
        ax.set_xticklabels([f.upper() for f in feature_order], rotation=45, ha='right', fontsize=10)
        
        # Add text annotations
        for i in range(len(model_order)):
            for j in range(len(feature_order)):
                value = matrix[i, j]
                text_color = 'white' if abs(value) > 1.5 else 'black'
                ax.text(j, i, f'{value:.1f}σ', ha='center', va='center', 
                       fontsize=8, color=text_color)
        
        # Title and colorbar
        ax.set_title("Model × Category Steerability Matrix\n(Z-scores: how many σ did injection shift behavior?)",
                    fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Steerability (σ)", fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Category comparison matrix saved to: {output_path}")
    
    def export_distance_matrix(self, output_path: str):
        """Export the distance matrix to a JSON file."""
        distances, labels = self.compute_distance_matrix()
        
        output = {
            "labels": labels,
            "distances": distances.tolist(),
            "metric": self.distance_metric,
            "features": sorted(list(self.all_features)),
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Distance matrix saved to: {output_path}")
    
    def export_dna_registry(self, output_path: str):
        """Export all DNA signatures to a JSON file."""
        output = {
            "num_models": len(self.dna_registry),
            "features": sorted(list(self.all_features)),
            "normalization": type(self.normalizer).__name__,
            "signatures": [dna.to_dict() for dna in self.dna_registry],
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"DNA registry saved to: {output_path}")
    
    def generate_report(self, output_dir: str):
        """Generate a comprehensive analysis report with per-category trees."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING MULTI-MODEL DNA ANALYSIS REPORT")
        print("="*60)
        
        # 1. Per-category phylogenetic trees (PRIMARY OUTPUT)
        trees_dir = output_dir / "trees_per_category"
        trees_dir.mkdir(exist_ok=True)
        tree_files = self.build_per_category_trees(str(trees_dir))
        
        # 2. Combined phylogenetic tree (for overall view)
        print(f"\nGenerating combined tree (all categories)...")
        self.build_phylogenetic_tree(str(output_dir / "tree_combined.png"))
        
        # 3. Category comparison matrix (heatmap)
        print(f"\nGenerating category comparison matrix...")
        self.build_category_comparison_matrix(str(output_dir / "category_matrix.png"))
        
        # 4. Distance matrices (per category)
        matrices_dir = output_dir / "distance_matrices"
        matrices_dir.mkdir(exist_ok=True)
        self._export_per_category_distances(str(matrices_dir))
        
        # 5. Combined distance matrix
        self.export_distance_matrix(str(output_dir / "distance_matrix_combined.json"))
        
        # 6. DNA registry
        self.export_dna_registry(str(output_dir / "dna_signatures.json"))
        
        # 7. Feature comparison plot
        self._plot_feature_comparison(str(output_dir / "feature_comparison.png"))
        
        # 8. Text summary
        self._write_summary(str(output_dir / "analysis_summary.txt"))
        
        # Print summary
        print("\n" + "="*60)
        print("REPORT COMPLETE")
        print("="*60)
        print(f"\nOutput directory: {output_dir}")
        print(f"\nPer-category trees ({len(tree_files)} categories):")
        for cat, path in tree_files.items():
            print(f"  • {cat}: {Path(path).name}")
        print(f"\nOther outputs:")
        print(f"  • tree_combined.png (all categories)")
        print(f"  • category_matrix.png (model × category heatmap)")
        print(f"  • dna_signatures.json")
        print(f"  • analysis_summary.txt")
    
    def _export_per_category_distances(self, output_dir: str):
        """Export distance matrices for each category separately."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for category in sorted(self.all_features):
            # Collect models with this category
            labels = []
            scores = []
            
            for dna in self.dna_registry:
                if category in dna.features:
                    labels.append(dna.full_id)
                    scores.append(dna.features[category])
            
            if len(labels) < 2:
                continue
            
            # Compute pairwise distances (using absolute difference for 1D)
            n = len(labels)
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    distances[i, j] = abs(scores[i] - scores[j])
            
            output = {
                "category": category,
                "labels": labels,
                "scores": scores,
                "distances": distances.tolist(),
            }
            
            with open(output_dir / f"distances_{category}.json", 'w') as f:
                json.dump(output, f, indent=2)
    
    def _plot_feature_comparison(self, output_path: str):
        """Plot feature-by-feature comparison across models."""
        if not self.dna_registry:
            return
        
        feature_order = sorted(list(self.all_features))
        n_features = len(feature_order)
        n_models = len(self.dna_registry)
        
        fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 6))
        if n_features == 1:
            axes = [axes]
        
        for idx, feature in enumerate(feature_order):
            ax = axes[idx]
            
            model_names = []
            scores = []
            
            for dna in self.dna_registry:
                model_names.append(dna.full_id)
                scores.append(dna.features.get(feature, 0.0))
            
            # Sort by score
            sorted_indices = np.argsort(scores)
            model_names = [model_names[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            
            # Color by sign
            colors = ['green' if s >= 0 else 'red' for s in scores]
            
            bars = ax.barh(range(len(model_names)), scores, color=colors, alpha=0.7)
            ax.set_yticks(range(len(model_names)))
            ax.set_yticklabels(model_names, fontsize=8)
            ax.set_xlabel("Steerability (σ)")
            ax.set_title(feature.capitalize())
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle("Feature-by-Feature Steerability Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _write_summary(self, output_path: str):
        """Write a text summary of the analysis."""
        lines = [
            "=" * 70,
            "MULTI-MODEL DNA ANALYSIS SUMMARY",
            "=" * 70,
            "",
            f"Total configurations analyzed: {len(self.dna_registry)}",
            f"Features (probe categories): {sorted(list(self.all_features))}",
            f"Normalization method: {type(self.normalizer).__name__}",
            f"Distance metric: {self.distance_metric}",
            "",
            "-" * 70,
            "DNA SIGNATURES (Z-Score Steerability)",
            "-" * 70,
            "",
        ]
        
        # Group by model
        model_groups = defaultdict(list)
        for dna in self.dna_registry:
            model_groups[dna.model_id].append(dna)
        
        for model_id, dnas in sorted(model_groups.items()):
            lines.append(f"Model: {model_id}")
            lines.append("-" * 40)
            
            for dna in sorted(dnas, key=lambda d: d.config_id):
                lines.append(f"  Config: {dna.config_id}")
                for feature in sorted(dna.features.keys()):
                    score = dna.features[feature]
                    raw = dna.raw_scores.get(feature, 0)
                    sigma = dna.baseline_stds.get(feature, 1)
                    lines.append(f"    {feature:20s}: DNA={score:+.3f}σ (raw={raw:+.3f}, σ={sigma:.3f})")
            
            lines.append("")
        
        # Most/least steerable
        lines.extend([
            "-" * 70,
            "EXTREME VALUES",
            "-" * 70,
            "",
        ])
        
        for feature in sorted(self.all_features):
            scores = [(dna.full_id, dna.features.get(feature, 0)) for dna in self.dna_registry]
            scores.sort(key=lambda x: x[1])
            
            if scores:
                lines.append(f"{feature.capitalize()}:")
                lines.append(f"  Most steerable:  {scores[-1][0]} ({scores[-1][1]:+.3f}σ)")
                lines.append(f"  Least steerable: {scores[0][0]} ({scores[0][1]:+.3f}σ)")
                lines.append("")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))


# =============================================================================
# SAE FINGERPRINT INTEGRATION
# =============================================================================

# Import SAE fingerprinting (optional)
try:
    from core.sae_fingerprint import (
        SAEFeatureExtractor,
        SAEModelFingerprint,
        SAEFingerprintAnalyzer,
        FingerprintDiff,
        build_model_fingerprint,
        build_fingerprints_from_precomputed,
    )
    HAS_SAE = True
except ImportError:
    HAS_SAE = False


class SAEMultiModelIntegrator:
    """
    Extends multi-model DNA analysis with SAE-based functional fingerprinting.

    While MultiModelDNAIntegrator works with probe-based steerability scores,
    this class adds SAE-based model fingerprints that capture semantic
    behavioral patterns across all responses.

    The key difference:
    - Probe-based DNA: "How much did the model shift under injection?" (steerability)
    - SAE fingerprint: "What semantic behaviors characterize this model?" (identity)

    Usage:
        integrator = SAEMultiModelIntegrator(
            reader_model_name="meta-llama/Llama-3.3-70B",
            sae_path="/path/to/sae.pt",
        )

        # Add model responses
        integrator.add_model_responses("GPT-5", gpt5_responses)
        integrator.add_model_responses("Grok-4", grok4_responses)

        # Build fingerprints and analyze
        integrator.build_fingerprints()
        diff = integrator.diff_models("GPT-5", "Grok-4")
        print(diff.get_report())

        # Build phylogenetic tree from SAE fingerprints
        integrator.build_phylogenetic_tree("sae_tree.png")
    """

    def __init__(
        self,
        reader_model_name: Optional[str] = None,
        sae_path: Optional[str] = None,
        layer_idx: int = 24,
        feature_labels: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize SAE multi-model integrator.

        Args:
            reader_model_name: HuggingFace model name for the reader LLM
            sae_path: Path to pretrained SAE checkpoint
            layer_idx: Reader model layer to hook for activations
            feature_labels: Optional mapping of SAE latent index → label
        """
        if not HAS_SAE:
            raise ImportError(
                "SAE fingerprinting module not available. "
                "Ensure core/sae_fingerprint.py is accessible."
            )

        self.reader_model_name = reader_model_name
        self.sae_path = sae_path
        self.layer_idx = layer_idx
        self.feature_labels = feature_labels or {}

        # Storage
        self._model_responses: Dict[str, List[str]] = {}
        self._analyzer = SAEFingerprintAnalyzer()

        # Optional: link to probe-based integrator for combined analysis
        self._probe_integrator: Optional[MultiModelDNAIntegrator] = None

    def add_model_responses(
        self,
        model_id: str,
        responses: List[str],
    ):
        """
        Add text responses from a target model.

        Args:
            model_id: Identifier for the target LLM
            responses: List of text responses from the model
        """
        self._model_responses[model_id] = responses

    def add_precomputed_fingerprint(
        self,
        fingerprint: 'SAEModelFingerprint',
    ):
        """Add a pre-built SAE fingerprint directly."""
        self._analyzer.add_fingerprint(fingerprint)

    def load_fingerprint(self, filepath: str):
        """Load a fingerprint from JSON file."""
        fp = SAEModelFingerprint.load(filepath)
        self._analyzer.add_fingerprint(fp)

    def set_probe_integrator(self, integrator: MultiModelDNAIntegrator):
        """Link a probe-based integrator for combined analysis."""
        self._probe_integrator = integrator

    def build_fingerprints(self):
        """
        Build SAE fingerprints for all added model responses.

        This runs Steps 2-4 of the SAE pipeline:
        - Feed responses through reader LLM + SAE
        - Max-pool + binarize per response
        - Aggregate into model fingerprints
        """
        if not self._model_responses:
            print("No model responses to process.")
            return

        if not self.reader_model_name or not self.sae_path:
            raise ValueError(
                "reader_model_name and sae_path are required for live "
                "extraction. Use add_precomputed_fingerprint() instead."
            )

        for model_id, responses in self._model_responses.items():
            print(f"Building SAE fingerprint for {model_id} "
                  f"({len(responses)} responses)...")

            fingerprint = build_model_fingerprint(
                model_id=model_id,
                responses=responses,
                reader_model_name=self.reader_model_name,
                sae_path=self.sae_path,
                layer_idx=self.layer_idx,
                feature_labels=self.feature_labels,
            )
            self._analyzer.add_fingerprint(fingerprint)
            print(f"  Done. Active features: "
                  f"{len(fingerprint.get_active_features())}")

    def build_fingerprints_from_precomputed(
        self,
        model_activations: Dict[str, np.ndarray],
        d_sae: int = 65536,
    ):
        """
        Build fingerprints from pre-extracted SAE activation arrays.

        Args:
            model_activations: Dict of model_id → activations array
                Each array: (n_responses, n_tokens, d_sae)
            d_sae: SAE dictionary size
        """
        analyzer = build_fingerprints_from_precomputed(
            model_responses=model_activations,
            d_sae=d_sae,
            feature_labels=self.feature_labels,
        )
        # Merge into our analyzer
        for model_id, fp in analyzer.fingerprints.items():
            self._analyzer.add_fingerprint(fp)

    # -----------------------------------------------------------------
    # Analysis: Step 5 - Dataset Diffing
    # -----------------------------------------------------------------

    def diff_models(
        self,
        model_a: str,
        model_b: str,
        top_n: int = 20,
    ) -> 'FingerprintDiff':
        """
        Diff two model fingerprints to find distinguishing semantic features.

        Args:
            model_a: Reference model ID
            model_b: Target model ID
            top_n: Number of top features to include

        Returns:
            FingerprintDiff with ranked distinguishing features
        """
        return self._analyzer.diff_fingerprints(model_a, model_b, top_n=top_n)

    def one_vs_rest(
        self,
        target_model: str,
        top_n: int = 20,
    ) -> 'FingerprintDiff':
        """Compare one model against all others."""
        return self._analyzer.one_vs_rest(target_model, top_n=top_n)

    def compute_distance_matrix(
        self,
        metric: str = "cosine",
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute pairwise distance matrix from SAE fingerprints."""
        return self._analyzer.compute_distance_matrix(metric)

    def build_phylogenetic_tree(
        self,
        output_path: str = "sae_phylogenetic_tree.png",
        metric: str = "cosine",
        method: str = "average",
        title: Optional[str] = None,
    ):
        """Build phylogenetic tree from SAE fingerprints."""
        self._analyzer.build_phylogenetic_tree(
            output_path=output_path,
            metric=metric,
            method=method,
            title=title,
        )

    # -----------------------------------------------------------------
    # Combined Analysis (SAE + Probe-based)
    # -----------------------------------------------------------------

    def generate_combined_report(
        self,
        output_dir: str,
    ):
        """
        Generate a comprehensive report combining SAE fingerprints
        with probe-based steerability analysis.

        Args:
            output_dir: Directory to save analysis outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("SAE FUNCTIONAL FINGERPRINT ANALYSIS")
        print("=" * 60)

        model_ids = self._analyzer.list_models()
        print(f"\nModels analyzed: {len(model_ids)}")
        for mid in model_ids:
            fp = self._analyzer.fingerprints[mid]
            active = len(fp.get_active_features())
            print(f"  - {mid}: {fp.n_responses} responses, "
                  f"{active} active features")

        # 1. SAE phylogenetic tree
        print("\nGenerating SAE phylogenetic tree...")
        self.build_phylogenetic_tree(
            str(output_dir / "sae_phylogenetic_tree.png"),
            title="LLM Functional DNA - SAE Fingerprint Phylogeny",
        )

        # 2. Pairwise diffs
        print("\nComputing pairwise fingerprint diffs...")
        diffs = self._analyzer.all_pairwise_diffs(top_n=10)

        diff_summaries = []
        for (mid_a, mid_b), diff in diffs.items():
            diff_summaries.append({
                "model_a": mid_a,
                "model_b": mid_b,
                "mean_abs_diff": diff.mean_absolute_diff,
                "n_significant": diff.n_significant_diffs,
                "top_feature": diff.top_positive[0]["label"] if diff.top_positive else "N/A",
                "top_diff_pct": diff.top_positive[0]["diff_pct"] if diff.top_positive else 0,
            })

        with open(output_dir / "pairwise_diffs.json", 'w') as f:
            json.dump(diff_summaries, f, indent=2)

        # 3. One-vs-rest for each model
        print("\nComputing one-vs-rest analyses...")
        unique_features = {}
        for mid in model_ids:
            if len(model_ids) < 2:
                continue
            ovr = self.one_vs_rest(mid, top_n=10)
            unique_features[mid] = {
                "top_positive": ovr.top_positive[:5],
                "top_negative": ovr.top_negative[:5],
                "n_significant": ovr.n_significant_diffs,
            }

        with open(output_dir / "unique_features.json", 'w') as f:
            json.dump(unique_features, f, indent=2, default=str)

        # 4. Diff charts for top pairs
        if len(model_ids) >= 2:
            print("\nGenerating diff charts...")
            charts_dir = output_dir / "diff_charts"
            charts_dir.mkdir(exist_ok=True)
            for mid_a, mid_b in list(diffs.keys())[:5]:
                chart = self._analyzer.generate_diff_chart(mid_a, mid_b)
                if chart:
                    import base64
                    chart_path = charts_dir / f"diff_{mid_a}_vs_{mid_b}.png"
                    with open(chart_path, 'wb') as f:
                        f.write(base64.b64decode(chart))

        # 5. Save all fingerprints
        self._analyzer.save_all(output_dir / "fingerprints")

        # 6. Text summary
        self._write_sae_summary(
            str(output_dir / "sae_analysis_summary.txt"),
            unique_features,
            diff_summaries,
        )

        # 7. If probe integrator is linked, generate combined tree
        if self._probe_integrator and self._probe_integrator.dna_registry:
            print("\nGenerating combined probe + SAE analysis...")
            self._probe_integrator.generate_report(
                str(output_dir / "probe_analysis")
            )

        print(f"\nReport saved to: {output_dir}")

    def _write_sae_summary(
        self,
        output_path: str,
        unique_features: Dict[str, Any],
        diff_summaries: List[Dict[str, Any]],
    ):
        """Write a text summary of SAE fingerprint analysis."""
        lines = [
            "=" * 70,
            "SAE FUNCTIONAL MODEL FINGERPRINT ANALYSIS",
            "=" * 70,
            "",
            "This analysis uses Sparse Autoencoder (SAE) features to create",
            "interpretable, functional fingerprints of each LLM's behavioral",
            "patterns. Each feature represents a learned semantic concept.",
            "",
        ]

        # Per-model unique features
        lines.append("-" * 70)
        lines.append("UNIQUE IDENTIFYING FEATURES (One-vs-Rest)")
        lines.append("-" * 70)
        lines.append("")

        for mid, analysis in unique_features.items():
            lines.append(f"Model: {mid}")
            lines.append(f"  Significant distinguishing features: {analysis['n_significant']}")
            lines.append(f"  Top features (higher than group average):")
            for feat in analysis.get("top_positive", [])[:3]:
                lines.append(f"    + {feat['label']}: +{feat['diff_pct']:.1f}%")
            lines.append(f"  Features lower than group average:")
            for feat in analysis.get("top_negative", [])[:3]:
                lines.append(f"    - {feat['label']}: {feat['diff_pct']:.1f}%")
            lines.append("")

        # Pairwise diffs
        lines.append("-" * 70)
        lines.append("PAIRWISE MODEL COMPARISONS")
        lines.append("-" * 70)
        lines.append("")

        for diff in sorted(diff_summaries, key=lambda d: d["mean_abs_diff"], reverse=True):
            lines.append(
                f"  {diff['model_a']} vs {diff['model_b']}: "
                f"mean|diff|={diff['mean_abs_diff']:.4f}, "
                f"significant={diff['n_significant']}, "
                f"top: {diff['top_feature']} ({diff['top_diff_pct']:+.1f}%)"
            )

        lines.append("")

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_multi_model_tree(
    model_experiments: Dict[str, str],  # {model_id: experiment_dir}
    probe_categories: List[str],
    output_dir: str = "combined_analysis",
    normalization: str = "zscore",
    distance_metric: str = "cosine",
) -> MultiModelDNAIntegrator:
    """
    Convenience function to build a combined tree from multiple models.
    
    Args:
        model_experiments: Dict mapping model IDs to experiment directories
        probe_categories: List of probe categories to analyze
        output_dir: Where to save the analysis
        normalization: Normalization strategy ("zscore", "ghost", "minmax")
        distance_metric: Distance metric ("cosine", "euclidean", "manhattan")
    
    Returns:
        The integrator object for further analysis
    
    Example:
        build_multi_model_tree(
            model_experiments={
                "Llama-3-8B": "experiments/llama3/",
                "Qwen-2.5-7B": "experiments/qwen/",
                "Gemma-7B": "experiments/gemma/",
            },
            probe_categories=["overconfidence", "sycophancy", "toxicity"],
            output_dir="combined_tree/"
        )
    """
    integrator = MultiModelDNAIntegrator(
        normalization=normalization,
        distance_metric=distance_metric,
    )
    
    for model_id, exp_dir in model_experiments.items():
        print(f"\nProcessing {model_id}...")
        integrator.add_model_experiments(
            model_id=model_id,
            experiment_dir=exp_dir,
            probe_categories=probe_categories,
        )
    
    integrator.generate_report(output_dir)
    
    return integrator


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build phylogenetic tree from multiple LLM experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add experiments from directories
  python multi_model_dna.py \\
      --model Llama-3:results/llama3 \\
      --model Qwen-2.5:results/qwen \\
      --categories overconfidence,sycophancy,toxicity \\
      --output combined_tree/

  # Add from comparison JSON files directly
  python multi_model_dna.py \\
      --comparison Llama-3:results/llama3_comparison.json \\
      --comparison Qwen-2.5:results/qwen_comparison.json \\
      --output combined_tree/
        """
    )
    
    parser.add_argument("--model", "-m", action="append", default=[],
                        help="Model experiment in format 'ModelID:experiment_dir'")
    parser.add_argument("--comparison", "-c", action="append", default=[],
                        help="Comparison JSON in format 'ModelID:path.json'")
    parser.add_argument("--categories", default="overconfidence,sycophancy,toxicity,hedging",
                        help="Comma-separated probe categories")
    parser.add_argument("--output", "-o", default="combined_analysis",
                        help="Output directory")
    parser.add_argument("--normalization", "-n", default="zscore",
                        choices=["zscore", "ghost", "minmax"])
    parser.add_argument("--distance", "-d", default="cosine",
                        choices=["cosine", "euclidean", "manhattan"])
    
    args = parser.parse_args()
    
    # Parse categories
    categories = [c.strip() for c in args.categories.split(",")]
    
    # Create integrator
    integrator = MultiModelDNAIntegrator(
        normalization=args.normalization,
        distance_metric=args.distance,
    )
    
    # Add from directories
    for model_spec in args.model:
        if ":" not in model_spec:
            print(f"Error: Invalid model spec '{model_spec}'. Use 'ModelID:path'")
            continue
        
        model_id, exp_dir = model_spec.split(":", 1)
        integrator.add_model_experiments(
            model_id=model_id,
            experiment_dir=exp_dir,
            probe_categories=categories,
        )
    
    # Add from comparison files
    for comp_spec in args.comparison:
        if ":" not in comp_spec:
            print(f"Error: Invalid comparison spec '{comp_spec}'. Use 'ModelID:path.json'")
            continue
        
        model_id, json_path = comp_spec.split(":", 1)
        for category in categories:
            try:
                integrator.add_from_comparison_json(
                    model_id=model_id,
                    comparison_path=json_path,
                    probe_category=category,
                )
            except Exception as e:
                print(f"Warning: Could not process {json_path} for {category}: {e}")
    
    # Generate report
    if integrator.dna_registry:
        integrator.generate_report(args.output)
    else:
        print("No DNA signatures loaded!")


if __name__ == "__main__":
    main()