#!/usr/bin/env python3
"""
LLM-DNA Utilities for Multi-Agent PAS
======================================

Extracted and adapted from the LLM-DNA project for phylogenetic analysis
of multi-agent experiment outputs.

Components:
- DNASignature: Core signature dataclass
- DNA Extractors: Text, Embedding, LMHead methods
- Distance Metrics: Euclidean, Cosine, Nei distance
- Phylogenetic Tree: UPGMA/NJ tree building with visualization

Original: https://github.com/[your-repo]/llm-dna
"""

import json
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from pathlib import Path
from enum import Enum

# Optional imports
try:
    from scipy.spatial.distance import pdist, squareform, cdist
    from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# DNA SIGNATURE
# =============================================================================

@dataclass
class DNASignature:
    """
    Core DNA signature representation.
    
    Compatible with LLM-DNA project format.
    """
    model_name: str
    signature: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields
    extraction_method: str = "unknown"
    reduction_method: str = "none"
    dna_dim: int = 0
    timestamp: str = ""
    
    def __post_init__(self):
        if isinstance(self.signature, list):
            self.signature = np.array(self.signature)
        if not self.dna_dim:
            self.dna_dim = len(self.signature) if self.signature is not None else 0
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "model_name": self.model_name,
            "signature": self.signature.tolist() if self.signature is not None else [],
            "metadata": self.metadata,
            "extraction_method": self.extraction_method,
            "reduction_method": self.reduction_method,
            "dna_dim": self.dna_dim,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNASignature":
        """Create from dict."""
        sig = data.get("signature", [])
        return cls(
            model_name=data["model_name"],
            signature=np.array(sig) if sig else np.array([]),
            metadata=data.get("metadata", {}),
            extraction_method=data.get("extraction_method", "unknown"),
            reduction_method=data.get("reduction_method", "none"),
            dna_dim=data.get("dna_dim", len(sig)),
            timestamp=data.get("timestamp", ""),
        )
    
    def save(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "DNASignature":
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def distance_to(self, other: "DNASignature", metric: str = "euclidean") -> float:
        """Compute distance to another signature."""
        return compute_distance(self.signature, other.signature, metric)


# =============================================================================
# DISTANCE METRICS
# =============================================================================

class DistanceMetric(Enum):
    """Available distance metrics."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    CORRELATION = "correlation"
    NEI = "nei"  # PhyloLM's Nei distance


def compute_distance(
    sig1: np.ndarray, 
    sig2: np.ndarray, 
    metric: str = "euclidean"
) -> float:
    """
    Compute distance between two signatures.
    
    Args:
        sig1, sig2: Signature vectors
        metric: Distance metric (euclidean, cosine, manhattan, correlation, nei)
    """
    if metric == "euclidean":
        return float(np.linalg.norm(sig1 - sig2))
    
    elif metric == "cosine":
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return float(1 - np.dot(sig1, sig2) / (norm1 * norm2))
    
    elif metric == "manhattan":
        return float(np.sum(np.abs(sig1 - sig2)))
    
    elif metric == "correlation":
        if np.std(sig1) == 0 or np.std(sig2) == 0:
            return 1.0
        return float(1 - np.corrcoef(sig1, sig2)[0, 1])
    
    elif metric == "nei":
        # PhyloLM's Nei distance: D = -log(S) where S = sum(P1*P2) / sqrt(sum(P1^2) * sum(P2^2))
        # For probability distributions
        p1 = np.abs(sig1) + 1e-10
        p2 = np.abs(sig2) + 1e-10
        p1 = p1 / np.sum(p1)
        p2 = p2 / np.sum(p2)
        
        numerator = np.sum(p1 * p2)
        denominator = np.sqrt(np.sum(p1**2) * np.sum(p2**2))
        
        if denominator == 0:
            return float('inf')
        
        similarity = numerator / denominator
        if similarity <= 0:
            return float('inf')
        
        return float(-np.log(similarity))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_distance_matrix(
    signatures: List[DNASignature],
    metric: str = "euclidean"
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise distance matrix.
    
    Returns:
        (distance_matrix, labels)
    """
    n = len(signatures)
    matrix = np.zeros((n, n))
    labels = [s.model_name for s in signatures]
    
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_distance(signatures[i].signature, signatures[j].signature, metric)
            matrix[i, j] = d
            matrix[j, i] = d
    
    return matrix, labels


def compute_distance_matrix_from_array(
    matrix: np.ndarray,
    metric: str = "euclidean"
) -> np.ndarray:
    """Compute distance matrix from signature array."""
    if not HAS_SCIPY:
        n = matrix.shape[0]
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = compute_distance(matrix[i], matrix[j], metric)
                dist[i, j] = d
                dist[j, i] = d
        return dist
    
    if metric == "nei":
        # Custom handling for Nei distance
        n = matrix.shape[0]
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = compute_distance(matrix[i], matrix[j], "nei")
                dist[i, j] = d
                dist[j, i] = d
        return dist
    
    return squareform(pdist(matrix, metric=metric))


# =============================================================================
# DIMENSIONALITY REDUCTION
# =============================================================================

class DimensionalityReducer:
    """
    Reduce high-dimensional embeddings to fixed DNA dimension.
    
    Methods:
    - random_projection: Gaussian random projection (fast, preserves distances)
    - pca: Principal Component Analysis
    - truncate: Simple truncation
    """
    
    def __init__(
        self, 
        method: str = "random_projection",
        target_dim: int = 64,
        random_state: int = 42
    ):
        self.method = method
        self.target_dim = target_dim
        self.random_state = random_state
        self._reducer = None
        self._projection_matrix = None
    
    def fit(self, X: np.ndarray) -> "DimensionalityReducer":
        """Fit the reducer to data."""
        if self.method == "random_projection":
            if HAS_SKLEARN:
                self._reducer = GaussianRandomProjection(
                    n_components=self.target_dim,
                    random_state=self.random_state
                )
                self._reducer.fit(X)
            else:
                np.random.seed(self.random_state)
                input_dim = X.shape[1]
                self._projection_matrix = np.random.randn(
                    self.target_dim, input_dim
                ) / np.sqrt(self.target_dim)
        
        elif self.method == "pca":
            if not HAS_SKLEARN:
                raise ImportError("sklearn required for PCA")
            self._reducer = PCA(n_components=self.target_dim, random_state=self.random_state)
            self._reducer.fit(X)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to lower dimension."""
        if self.method == "truncate":
            return X[:, :self.target_dim] if X.ndim > 1 else X[:self.target_dim]
        
        if self.method == "random_projection":
            if self._reducer is not None:
                return self._reducer.transform(X)
            elif self._projection_matrix is not None:
                if X.ndim == 1:
                    return self._projection_matrix @ X
                return (self._projection_matrix @ X.T).T
        
        if self._reducer is not None:
            return self._reducer.transform(X)
        
        return X[:, :self.target_dim] if X.ndim > 1 else X[:self.target_dim]
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


# =============================================================================
# PHYLOGENETIC TREE
# =============================================================================

class TreeMethod(Enum):
    """Tree construction methods."""
    UPGMA = "upgma"  # Unweighted Pair Group Method with Arithmetic Mean
    NJ = "nj"        # Neighbor-Joining (approximated)
    WARD = "ward"    # Ward's minimum variance


class PhylogeneticTree:
    """
    Phylogenetic tree builder and visualizer.
    
    Supports:
    - Multiple tree construction methods (UPGMA, NJ, Ward)
    - Multiple distance metrics
    - Newick export for iTOL
    - Dendrogram visualization
    - Distance matrix heatmaps
    """
    
    def __init__(
        self, 
        method: str = "upgma",
        metric: str = "euclidean"
    ):
        if not HAS_SCIPY:
            raise ImportError("scipy required for phylogenetic tree building")
        
        self.method = method.lower()
        self.metric = metric.lower()
        
        self.linkage_matrix = None
        self.distance_matrix = None
        self.labels: List[str] = []
        self.signatures: List[DNASignature] = []
    
    def fit(
        self, 
        signatures: Union[List[DNASignature], np.ndarray],
        labels: Optional[List[str]] = None
    ) -> "PhylogeneticTree":
        """
        Build tree from signatures.
        
        Args:
            signatures: List of DNASignature objects or numpy array
            labels: Optional labels (required if signatures is array)
        """
        if isinstance(signatures, list) and len(signatures) > 0:
            if isinstance(signatures[0], DNASignature):
                self.signatures = signatures
                self.labels = [s.model_name for s in signatures]
                matrix = np.vstack([s.signature for s in signatures])
            else:
                matrix = np.array(signatures)
                self.labels = labels or [f"sig_{i}" for i in range(len(matrix))]
        else:
            matrix = np.array(signatures)
            self.labels = labels or [f"sig_{i}" for i in range(len(matrix))]
        
        # Compute distance matrix
        self.distance_matrix = compute_distance_matrix_from_array(matrix, self.metric)
        
        # Build linkage
        linkage_method_map = {
            "upgma": "average",
            "nj": "average",  # NJ approximation
            "ward": "ward",
        }
        linkage_method = linkage_method_map.get(self.method, "average")
        
        # Convert to condensed form for linkage
        condensed = squareform(self.distance_matrix)
        self.linkage_matrix = linkage(condensed, method=linkage_method)
        
        return self
    
    def get_newick(self) -> str:
        """
        Export tree in Newick format.
        
        Can be uploaded to iTOL (https://itol.embl.de/) for visualization.
        """
        if self.linkage_matrix is None:
            raise ValueError("Tree not built. Call fit() first.")
        
        tree = to_tree(self.linkage_matrix)
        
        def _to_newick(node, labels):
            if node.is_leaf():
                return labels[node.id]
            else:
                left = _to_newick(node.left, labels)
                right = _to_newick(node.right, labels)
                return f"({left}:{node.left.dist:.6f},{right}:{node.right.dist:.6f})"
        
        return _to_newick(tree, self.labels) + ";"
    
    def get_leaf_order(self) -> List[str]:
        """Get leaf order from dendrogram."""
        if self.linkage_matrix is None:
            return self.labels
        
        dend = dendrogram(self.linkage_matrix, no_plot=True)
        return [self.labels[i] for i in dend['leaves']]
    
    def plot_dendrogram(
        self,
        figsize: Tuple[int, int] = (14, 10),
        title: str = "Behavioral Phylogeny",
        save_path: Optional[str] = None,
        color_threshold: Optional[float] = None,
        leaf_font_size: int = 9,
        orientation: str = "right",
        color_map: Optional[Dict[str, str]] = None
    ):
        """
        Plot dendrogram visualization.
        
        Args:
            figsize: Figure size
            title: Plot title
            save_path: Path to save figure
            color_threshold: Threshold for coloring clusters
            leaf_font_size: Font size for leaf labels
            orientation: "right", "left", "top", "bottom"
            color_map: Optional mapping of label patterns to colors
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
        if self.linkage_matrix is None:
            raise ValueError("Tree not built. Call fit() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create dendrogram
        dend = dendrogram(
            self.linkage_matrix,
            labels=self.labels,
            orientation=orientation,
            leaf_font_size=leaf_font_size,
            color_threshold=color_threshold,
            ax=ax
        )
        
        # Color leaf labels if color_map provided
        if color_map and orientation in ["right", "left"]:
            ylocs = ax.get_yticks()
            for idx, label in enumerate(ax.get_yticklabels()):
                label_text = label.get_text()
                for pattern, color in color_map.items():
                    if pattern in label_text:
                        label.set_color(color)
                        break
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if orientation in ["right", "left"]:
            ax.set_xlabel(f"Distance ({self.metric})", fontsize=11)
        else:
            ax.set_ylabel(f"Distance ({self.metric})", fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved dendrogram: {save_path}")
        
        return fig, ax
    
    def plot_distance_matrix(
        self,
        figsize: Tuple[int, int] = (12, 10),
        title: str = "Signature Distance Matrix",
        save_path: Optional[str] = None,
        cmap: str = "viridis",
        annotate: bool = False
    ):
        """Plot distance matrix heatmap."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
        if self.distance_matrix is None:
            raise ValueError("Tree not built. Call fit() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Reorder by dendrogram
        order = dendrogram(self.linkage_matrix, no_plot=True)['leaves']
        reordered = self.distance_matrix[order][:, order]
        reordered_labels = [self.labels[i] for i in order]
        
        im = ax.imshow(reordered, cmap=cmap, aspect='auto')
        
        ax.set_xticks(range(len(reordered_labels)))
        ax.set_yticks(range(len(reordered_labels)))
        ax.set_xticklabels(reordered_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(reordered_labels, fontsize=8)
        
        if annotate and len(self.labels) <= 15:
            for i in range(len(reordered_labels)):
                for j in range(len(reordered_labels)):
                    ax.text(j, i, f"{reordered[i,j]:.2f}", 
                           ha='center', va='center', fontsize=7)
        
        plt.colorbar(im, ax=ax, label=f"Distance ({self.metric})")
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved distance matrix: {save_path}")
        
        return fig, ax
    
    def plot_combined(
        self,
        figsize: Tuple[int, int] = (18, 8),
        title: str = "Phylogenetic Analysis",
        save_path: Optional[str] = None
    ):
        """Plot dendrogram and distance matrix side by side."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Dendrogram
        dendrogram(
            self.linkage_matrix,
            labels=self.labels,
            orientation='right',
            leaf_font_size=9,
            ax=ax1
        )
        ax1.set_title("Dendrogram", fontsize=12)
        ax1.set_xlabel(f"Distance ({self.metric})")
        
        # Distance matrix
        order = dendrogram(self.linkage_matrix, no_plot=True)['leaves']
        reordered = self.distance_matrix[order][:, order]
        reordered_labels = [self.labels[i] for i in order]
        
        im = ax2.imshow(reordered, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(reordered_labels)))
        ax2.set_yticks(range(len(reordered_labels)))
        ax2.set_xticklabels(reordered_labels, rotation=90, fontsize=8)
        ax2.set_yticklabels(reordered_labels, fontsize=8)
        plt.colorbar(im, ax=ax2, label=f"Distance ({self.metric})")
        ax2.set_title("Distance Matrix", fontsize=12)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved combined plot: {save_path}")
        
        return fig, (ax1, ax2)
    
    def save(self, output_dir: str, prefix: str = "tree"):
        """
        Save tree in multiple formats.
        
        Creates:
        - {prefix}.newick: Newick format for iTOL
        - {prefix}_distances.json: Distance matrix with labels
        - {prefix}_metadata.json: Tree metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Newick
        newick_path = output_dir / f"{prefix}.newick"
        with open(newick_path, 'w') as f:
            f.write(self.get_newick())
        
        # Distance matrix
        dist_path = output_dir / f"{prefix}_distances.json"
        with open(dist_path, 'w') as f:
            json.dump({
                "labels": self.labels,
                "distances": self.distance_matrix.tolist(),
                "metric": self.metric,
            }, f, indent=2)
        
        # Metadata
        meta_path = output_dir / f"{prefix}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                "num_leaves": len(self.labels),
                "labels": self.labels,
                "method": self.method,
                "metric": self.metric,
                "leaf_order": self.get_leaf_order(),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        
        print(f"Saved tree files to {output_dir}/")
        return output_dir
    
    @classmethod
    def load(cls, newick_path: str) -> str:
        """Load Newick string from file."""
        with open(newick_path, 'r') as f:
            return f.read().strip()


# =============================================================================
# iTOL ANNOTATION HELPERS
# =============================================================================

def generate_itol_colorstrip(
    labels: List[str],
    color_rules: Dict[str, str],
    output_path: str,
    dataset_label: str = "Categories"
):
    """
    Generate iTOL color strip annotation file.
    
    Args:
        labels: List of leaf labels
        color_rules: Dict mapping patterns to hex colors
        output_path: Output file path
        dataset_label: Label for the color strip
    """
    lines = [
        "DATASET_COLORSTRIP",
        "SEPARATOR TAB",
        f"DATASET_LABEL\t{dataset_label}",
        "COLOR\t#000000",
        "",
        "DATA",
    ]
    
    for label in labels:
        color = "#808080"  # Default gray
        for pattern, c in color_rules.items():
            if pattern in label:
                color = c
                break
        lines.append(f"{label}\t{color}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved iTOL color strip: {output_path}")


def generate_itol_labels(
    labels: List[str],
    label_map: Optional[Dict[str, str]] = None,
    output_path: str = "itol_labels.txt"
):
    """
    Generate iTOL label annotation file for renaming leaves.
    
    Args:
        labels: Original labels
        label_map: Optional mapping to new labels
        output_path: Output file path
    """
    lines = [
        "LABELS",
        "SEPARATOR TAB",
        "",
        "DATA",
    ]
    
    for label in labels:
        new_label = label_map.get(label, label) if label_map else label
        # Clean up label for display
        display = new_label.replace("__", " | ").replace("_", " ")
        lines.append(f"{label}\t{display}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved iTOL labels: {output_path}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_tree_from_signatures(
    signatures: List[DNASignature],
    method: str = "upgma",
    metric: str = "euclidean",
    output_dir: Optional[str] = None,
    plot: bool = True
) -> PhylogeneticTree:
    """
    Quick function to build and optionally visualize a tree.
    
    Args:
        signatures: List of DNA signatures
        method: Tree method (upgma, nj, ward)
        metric: Distance metric
        output_dir: Optional output directory
        plot: Whether to generate plots
    """
    tree = PhylogeneticTree(method=method, metric=metric)
    tree.fit(signatures)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tree.save(str(output_dir))
        
        if plot and HAS_MATPLOTLIB:
            tree.plot_dendrogram(save_path=str(output_dir / "dendrogram.png"))
            tree.plot_distance_matrix(save_path=str(output_dir / "distance_matrix.png"))
    
    return tree


def load_signatures_from_directory(dna_dir: str) -> List[DNASignature]:
    """Load all DNA signatures from a directory."""
    signatures = []
    dna_path = Path(dna_dir)
    
    # Try all_signatures.json first
    all_sigs = dna_path / "all_signatures.json"
    if all_sigs.exists():
        with open(all_sigs, 'r') as f:
            data = json.load(f)
        for item in data:
            signatures.append(DNASignature.from_dict(item))
        return signatures
    
    # Otherwise load individual files
    for json_file in dna_path.rglob("*_dna.json"):
        try:
            signatures.append(DNASignature.load(str(json_file)))
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return signatures