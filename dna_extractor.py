"""
Behavioral DNA Extractor for Federal Court Dashboard
=====================================================

Integrates with the Federal Court simulation to extract fine-grained
behavioral DNA signatures in real-time.

Provides:
- Real-time feature extraction during simulation
- Post-session analysis and visualization
- Comparison between agents
- Injection effect analysis
- Galaxy visualization with fine-grained dimensions
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import base64
import io
from datetime import datetime

# Import behavioral DNA classes
from behavioral_dna import (
    AgentBehavioralDNA,
    AgentBehavioralDNACollection,
    AgentBehavioralDNAMetadata,
    AgentRole,
    CourtPhase,
    TrajectoryArchetype,
    SAEFeatures,
    extract_behavioral_dna_from_session,
    TOTAL_DNA_DIM,
    SAE_SUMMARY_DIM,
    SAE_ENRICHED_DNA_DIM,
)

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

# Optional visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import FancyBboxPatch
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


# =============================================================================
# FEATURE LABELS FOR VISUALIZATION
# =============================================================================

FEATURE_LABELS = {
    # Token-level (0-11)
    0: "Score Mean",
    1: "Score Std",
    2: "Score Range",
    3: "Score Entropy",
    4: "Score Skewness",
    5: "Score Kurtosis",
    6: "Mean Gradient",
    7: "Max Gradient",
    8: "Gradient Sign Changes",
    9: "Argmax Position",
    10: "Argmin Position",
    11: "First-Token Bias",
    
    # Temporal (12-21)
    12: "Drift Velocity",
    13: "Drift Direction",
    14: "Total Drift",
    15: "Oscillation Frequency",
    16: "Oscillation Amplitude",
    17: "Momentum",
    18: "Acceleration",
    19: "Convergence Rate",
    20: "Final Trajectory Slope",
    21: "Variance Trend",
    
    # Cross-Agent (22-29)
    22: "Mean Reactivity",
    23: "Max Reactivity",
    24: "Mean Mirroring",
    25: "Dominance Index",
    26: "Opposition Strength",
    27: "Polarization Contribution",
    28: "Judge Alignment",
    29: "Injection Contagion",
    
    # Linguistic (30-39)
    30: "Hedging Index",
    31: "Assertiveness Score",
    32: "Certainty Ratio",
    33: "Citation Density",
    34: "Emotional Valence",
    35: "Emotional Intensity",
    36: "Sentence Complexity",
    37: "Question Ratio",
    38: "First Person Ratio",
    39: "Conditional Density",
    
    # Probe Interaction (40-45)
    40: "Confidence-Persuasiveness",
    41: "Logic-Emotion Balance",
    42: "Calibration Score",
    43: "Dominant Probe Score",
    44: "Max Probe Volatility",
    45: "Probe Count (norm)",
    
    # Injection (46-53)
    46: "Is Injected",
    47: "Injection Strength",
    48: "Absorption Rate",
    49: "Peak Effect",
    50: "Decay Half-Life",
    51: "Amplification Factor",
    52: "Cross Contamination",
    53: "Resistance Score",
    
    # Role Compliance (54-59)
    54: "Role Compliance",
    55: "Out-of-Role (norm)",
    56: "Role Confusion",
    57: "Role Metric 1",
    58: "Role Metric 2",
    59: "Role Metric 3",
    
    # Composite (60-67)
    60: "Advocacy Effectiveness",
    61: "Judicial Quality",
    62: "Deliberation Quality",
    63: "Calibration Index",
    64: "Manipulation Susceptibility",
    65: "Behavioral Stability",
    66: "Engagement Index",
    67: "Polarization Index",

    # SAE Summary Features (128-143, only present when SAE enrichment is enabled)
    128: "SAE Top Feature 1 Freq",
    129: "SAE Top Feature 2 Freq",
    130: "SAE Top Feature 3 Freq",
    131: "SAE Top Feature 4 Freq",
    132: "SAE Top Feature 5 Freq",
    133: "SAE Top Feature 6 Freq",
    134: "SAE Top Feature 7 Freq",
    135: "SAE Top Feature 8 Freq",
    136: "SAE Mean Activation Freq",
    137: "SAE Activation Sparsity",
    138: "SAE Frequency Entropy",
    139: "SAE Frequency Std",
    140: "SAE Frequency Skewness",
    141: "SAE Frequency Kurtosis",
    142: "SAE Gini Coefficient",
    143: "SAE Population Overlap",
}


DIMENSION_GROUPS = {
    "Token Dynamics": list(range(0, 12)),
    "Temporal Patterns": list(range(12, 22)),
    "Cross-Agent": list(range(22, 30)),
    "Linguistic": list(range(30, 40)),
    "Probe Interaction": list(range(40, 46)),
    "Injection Response": list(range(46, 54)),
    "Role Compliance": list(range(54, 60)),
    "Composite Indices": list(range(60, 68)),
    "SAE Fingerprint": list(range(128, 144)),
}


# =============================================================================
# FINE-GRAINED DNA ANALYZER
# =============================================================================

class FineGrainedDNAAnalyzer:
    """
    Analyzes specific parameter changes within DNA embeddings rather than
    relying solely on overall vector distance. Provides per-dimension
    profiling, discriminative fingerprinting, and parameter-pattern matching.
    """

    def __init__(self, collection: AgentBehavioralDNACollection):
        self.collection = collection
        self._vectors: Dict[str, np.ndarray] = {}
        self._matrix: Optional[np.ndarray] = None
        self._pop_mean: Optional[np.ndarray] = None
        self._pop_std: Optional[np.ndarray] = None
        self._z_scores: Dict[str, np.ndarray] = {}
        self._build_cache()

    def _build_cache(self):
        """Pre-compute vectors and population statistics."""
        agents = list(self.collection.signatures.keys())
        if not agents:
            return

        for aid in agents:
            self._vectors[aid] = self.collection[aid].to_vector()

        self._matrix = np.array([self._vectors[aid] for aid in agents])
        self._pop_mean = np.mean(self._matrix, axis=0)
        self._pop_std = np.std(self._matrix, axis=0)
        # Avoid division by zero — dims with zero std are non-discriminative
        self._pop_std = np.where(self._pop_std < 1e-10, 1.0, self._pop_std)

        for aid in agents:
            self._z_scores[aid] = (self._vectors[aid] - self._pop_mean) / self._pop_std

    def _dim_group(self, idx: int) -> str:
        """Get the group name for a dimension index."""
        for group_name, indices in DIMENSION_GROUPS.items():
            if idx in indices:
                return group_name
        return "Unknown"

    # -----------------------------------------------------------------
    # (a) Per-dimension shift profiling
    # -----------------------------------------------------------------

    def compute_parameter_profiles(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        For each agent, compute z-scores per dimension relative to the
        population and classify as elevated / suppressed / normal.

        Returns:
            {agent_id: {dim_index: {value, z_score, classification, label, group}}}
        """
        profiles: Dict[str, Dict[int, Dict[str, Any]]] = {}

        for aid, z_vec in self._z_scores.items():
            raw_vec = self._vectors[aid]
            profile: Dict[int, Dict[str, Any]] = {}

            for idx in range(len(z_vec)):
                z = float(z_vec[idx])
                if z > 1.0:
                    classification = "elevated"
                elif z < -1.0:
                    classification = "suppressed"
                else:
                    classification = "normal"

                profile[idx] = {
                    "value": float(raw_vec[idx]),
                    "z_score": z,
                    "classification": classification,
                    "label": FEATURE_LABELS.get(idx, f"Feature {idx}"),
                    "group": self._dim_group(idx),
                    "pop_mean": float(self._pop_mean[idx]),
                    "pop_std": float(self._pop_std[idx]),
                }

            profiles[aid] = profile

        return profiles

    # -----------------------------------------------------------------
    # (b) Discriminative dimension identification
    # -----------------------------------------------------------------

    def get_discriminative_dimensions(
        self,
        agent_id: str,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find the dimensions where this agent is most distinct from the
        population. These form the agent's 'parameter fingerprint'.

        Returns:
            List of {index, label, group, z_score, value, direction} sorted
            by |z_score| descending.
        """
        if agent_id not in self._z_scores:
            return []

        z_vec = self._z_scores[agent_id]
        raw_vec = self._vectors[agent_id]

        ranked = np.argsort(np.abs(z_vec))[::-1][:top_n]

        result = []
        for idx in ranked:
            idx = int(idx)
            z = float(z_vec[idx])
            result.append({
                "index": idx,
                "label": FEATURE_LABELS.get(idx, f"Feature {idx}"),
                "group": self._dim_group(idx),
                "z_score": z,
                "value": float(raw_vec[idx]),
                "direction": "elevated" if z > 0 else "suppressed",
            })

        return result

    def get_all_fingerprints(self, top_n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Get discriminative dimension fingerprints for all agents."""
        return {
            aid: self.get_discriminative_dimensions(aid, top_n)
            for aid in self._z_scores
        }

    # -----------------------------------------------------------------
    # (c) Parameter shift analysis between two agents
    # -----------------------------------------------------------------

    def analyze_parameter_shifts(
        self,
        agent_a: str,
        agent_b: str,
    ) -> Dict[str, Any]:
        """
        Enhanced pairwise comparison: per-dimension signed shifts, effect
        sizes, group-level summaries, and shift correlation.

        Returns dict with:
            - per_dimension: list of per-dim shift details
            - group_summary: {group: {mean_shift, mean_effect_size, top_dim}}
            - shift_correlation: dims that shift in correlated ways
            - overall: euclidean, cosine, significant_count
        """
        if agent_a not in self._vectors or agent_b not in self._vectors:
            return {"error": "Agent not found"}

        vec_a = self._vectors[agent_a]
        vec_b = self._vectors[agent_b]
        diff = vec_b - vec_a  # signed shift

        # Per-dimension analysis
        per_dim = []
        for idx in range(len(diff)):
            raw_shift = float(diff[idx])
            pop_std_val = float(self._pop_std[idx])
            effect_size = raw_shift / pop_std_val if pop_std_val > 1e-10 else 0.0

            significant = abs(effect_size) > 0.8  # Cohen's d threshold

            per_dim.append({
                "index": idx,
                "label": FEATURE_LABELS.get(idx, f"Feature {idx}"),
                "group": self._dim_group(idx),
                "value_a": float(vec_a[idx]),
                "value_b": float(vec_b[idx]),
                "shift": raw_shift,
                "effect_size": effect_size,
                "direction": "increase" if raw_shift > 0 else "decrease" if raw_shift < 0 else "unchanged",
                "significant": significant,
            })

        # Sort by absolute effect size for convenience
        per_dim_sorted = sorted(per_dim, key=lambda d: abs(d["effect_size"]), reverse=True)

        # Group-level summary
        group_summary = {}
        for group_name, indices in DIMENSION_GROUPS.items():
            valid = [per_dim[i] for i in indices if i < len(diff)]
            if not valid:
                continue
            shifts = [d["shift"] for d in valid]
            effects = [d["effect_size"] for d in valid]
            top_dim = max(valid, key=lambda d: abs(d["effect_size"]))

            group_summary[group_name] = {
                "mean_shift": float(np.mean(shifts)),
                "mean_abs_shift": float(np.mean(np.abs(shifts))),
                "mean_effect_size": float(np.mean(np.abs(effects))),
                "max_effect_size": float(np.max(np.abs(effects))),
                "significant_count": sum(1 for d in valid if d["significant"]),
                "dominant_direction": "increase" if np.mean(shifts) > 0 else "decrease",
                "top_dimension": {
                    "label": top_dim["label"],
                    "effect_size": top_dim["effect_size"],
                },
            }

        # Overall metrics
        euclidean_dist = float(np.linalg.norm(diff))
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        cosine_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
        significant_count = sum(1 for d in per_dim if d["significant"])

        # Shift correlation: which dimensions shift together?
        # Use the diff vector and find pairs of dims with same sign & large magnitude
        shift_clusters = self._find_correlated_shifts(diff)

        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "per_dimension": per_dim_sorted,
            "group_summary": group_summary,
            "shift_clusters": shift_clusters,
            "overall": {
                "euclidean_distance": euclidean_dist,
                "cosine_similarity": cosine_sim,
                "total_dimensions": len(diff),
                "significant_shifts": significant_count,
                "significant_ratio": significant_count / len(diff) if len(diff) > 0 else 0.0,
            },
        }

    def _find_correlated_shifts(self, diff: np.ndarray) -> List[Dict[str, Any]]:
        """Find clusters of dimensions that shift together across the population."""
        if self._matrix is None or len(self._matrix) < 3:
            return []

        # Compute correlation matrix across agents for all dimensions
        # Each column is a dimension, each row is an agent
        try:
            corr = np.corrcoef(self._matrix.T)
        except Exception:
            return []

        # Find clusters: dims with |correlation| > 0.7 that also shift significantly
        significant_dims = [i for i in range(len(diff)) if abs(diff[i] / (self._pop_std[i] + 1e-10)) > 0.5]

        clusters = []
        visited = set()

        for i in significant_dims:
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)

            for j in significant_dims:
                if j in visited or j == i:
                    continue
                if abs(corr[i, j]) > 0.7:
                    cluster.append(j)
                    visited.add(j)

            if len(cluster) >= 2:
                clusters.append({
                    "dimensions": [
                        {"index": idx, "label": FEATURE_LABELS.get(idx, f"Feature {idx}")}
                        for idx in cluster
                    ],
                    "shift_direction": "increase" if np.mean(diff[cluster]) > 0 else "decrease",
                    "mean_shift": float(np.mean(diff[cluster])),
                })

        return clusters

    # -----------------------------------------------------------------
    # (d) Agent matching by parameter pattern
    # -----------------------------------------------------------------

    def match_agent_by_parameters(
        self,
        output_vector: np.ndarray,
        top_n_dims: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Identify which agent an output DNA vector most resembles using
        parameter-level profile matching rather than raw distance.

        Scoring:
        1. Compute the candidate's z-score profile
        2. Find its discriminative dims (elevated/suppressed)
        3. For each known agent, score by:
           - Jaccard similarity of discriminative dimension sets
           - Weighted cosine similarity on discriminative dims only
           - Combined score = 0.4 * jaccard + 0.6 * weighted_cosine

        Returns:
            Ranked list of {agent_id, combined_score, jaccard, weighted_cosine,
                            aligned_dims, misaligned_dims}
        """
        if self._pop_mean is None:
            return []

        # Candidate z-scores
        candidate_z = (output_vector - self._pop_mean) / self._pop_std
        candidate_disc = set(int(i) for i in np.argsort(np.abs(candidate_z))[::-1][:top_n_dims])
        candidate_elevated = set(i for i in candidate_disc if candidate_z[i] > 0)
        candidate_suppressed = set(i for i in candidate_disc if candidate_z[i] <= 0)

        results = []
        for aid, z_vec in self._z_scores.items():
            agent_disc = set(int(i) for i in np.argsort(np.abs(z_vec))[::-1][:top_n_dims])
            agent_elevated = set(i for i in agent_disc if z_vec[i] > 0)
            agent_suppressed = set(i for i in agent_disc if z_vec[i] <= 0)

            # Jaccard: overlap of discriminative dimension sets
            union = candidate_disc | agent_disc
            intersection = candidate_disc & agent_disc
            jaccard = len(intersection) / len(union) if union else 0.0

            # Direction-aware Jaccard: dims must also agree on direction
            direction_match = (candidate_elevated & agent_elevated) | (candidate_suppressed & agent_suppressed)
            direction_jaccard = len(direction_match) / len(union) if union else 0.0

            # Weighted cosine on discriminative dims only
            disc_union = list(union)
            if disc_union:
                c_sub = candidate_z[disc_union]
                a_sub = z_vec[disc_union]
                dot = np.dot(c_sub, a_sub)
                norm_c = np.linalg.norm(c_sub)
                norm_a = np.linalg.norm(a_sub)
                weighted_cosine = float(dot / (norm_c * norm_a)) if norm_c > 0 and norm_a > 0 else 0.0
            else:
                weighted_cosine = 0.0

            combined = 0.3 * direction_jaccard + 0.7 * max(0.0, weighted_cosine)

            # Alignment detail
            aligned = []
            misaligned = []
            for idx in intersection:
                idx = int(idx)
                same_dir = (candidate_z[idx] > 0) == (z_vec[idx] > 0)
                entry = {
                    "index": idx,
                    "label": FEATURE_LABELS.get(idx, f"Feature {idx}"),
                    "candidate_z": float(candidate_z[idx]),
                    "agent_z": float(z_vec[idx]),
                }
                if same_dir:
                    aligned.append(entry)
                else:
                    misaligned.append(entry)

            results.append({
                "agent_id": aid,
                "combined_score": float(combined),
                "direction_jaccard": float(direction_jaccard),
                "jaccard": float(jaccard),
                "weighted_cosine": float(weighted_cosine),
                "aligned_dims": sorted(aligned, key=lambda d: abs(d["candidate_z"]), reverse=True),
                "misaligned_dims": sorted(misaligned, key=lambda d: abs(d["candidate_z"]), reverse=True),
                "shared_discriminative_count": len(intersection),
            })

        results.sort(key=lambda r: r["combined_score"], reverse=True)
        return results

    # -----------------------------------------------------------------
    # (e) Population-level parameter statistics
    # -----------------------------------------------------------------

    def compute_population_statistics(self) -> Dict[str, Any]:
        """
        Per-dimension statistics across all agents. Identifies stable vs
        discriminative dimensions and correlated dimension clusters.
        """
        if self._matrix is None:
            return {}

        n_dims = self._matrix.shape[1]

        per_dim = []
        for idx in range(n_dims):
            col = self._matrix[:, idx]
            mean_val = float(np.mean(col))
            std_val = float(np.std(col))
            cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else 0.0

            per_dim.append({
                "index": idx,
                "label": FEATURE_LABELS.get(idx, f"Feature {idx}"),
                "group": self._dim_group(idx),
                "mean": mean_val,
                "std": std_val,
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "range": float(np.max(col) - np.min(col)),
                "coefficient_of_variation": float(cv),
                "is_discriminative": std_val > 0.1,  # non-trivial variance
            })

        # Rank by discriminative power (coefficient of variation)
        per_dim_ranked = sorted(per_dim, key=lambda d: d["coefficient_of_variation"], reverse=True)

        discriminative_dims = [d for d in per_dim if d["is_discriminative"]]
        stable_dims = [d for d in per_dim if not d["is_discriminative"]]

        # Group-level statistics
        group_stats = {}
        for group_name, indices in DIMENSION_GROUPS.items():
            valid = [per_dim[i] for i in indices if i < n_dims]
            if valid:
                group_stats[group_name] = {
                    "mean_cv": float(np.mean([d["coefficient_of_variation"] for d in valid])),
                    "discriminative_count": sum(1 for d in valid if d["is_discriminative"]),
                    "total_count": len(valid),
                    "most_variable": max(valid, key=lambda d: d["coefficient_of_variation"])["label"],
                }

        # Correlation clusters
        try:
            corr = np.corrcoef(self._matrix.T)
            high_corr_pairs = []
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    if abs(corr[i, j]) > 0.7:
                        high_corr_pairs.append({
                            "dim_a": {"index": i, "label": FEATURE_LABELS.get(i, f"Feature {i}")},
                            "dim_b": {"index": j, "label": FEATURE_LABELS.get(j, f"Feature {j}")},
                            "correlation": float(corr[i, j]),
                        })
            high_corr_pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        except Exception:
            high_corr_pairs = []

        return {
            "per_dimension": per_dim,
            "per_dimension_ranked": per_dim_ranked[:20],
            "discriminative_count": len(discriminative_dims),
            "stable_count": len(stable_dims),
            "group_statistics": group_stats,
            "correlated_pairs": high_corr_pairs[:20],
            "total_agents": len(self._vectors),
            "total_dimensions": n_dims,
        }

    # -----------------------------------------------------------------
    # Visualization: parameter heatmap
    # -----------------------------------------------------------------

    def generate_parameter_heatmap(self) -> Optional[str]:
        """
        Heatmap of z-scores: rows=agents, columns=dimensions.
        Blue=suppressed, red=elevated, white=normal.
        Returns base64-encoded PNG or None if visualization unavailable.
        """
        if not HAS_VIZ or not self._z_scores:
            return None

        agents = list(self._z_scores.keys())
        z_matrix = np.array([self._z_scores[a] for a in agents])
        n_agents, n_dims = z_matrix.shape

        # Use only labeled dimensions
        max_dim = max(FEATURE_LABELS.keys()) + 1
        use_dims = min(n_dims, max_dim)
        z_matrix = z_matrix[:, :use_dims]

        col_labels = [FEATURE_LABELS.get(i, f"F{i}") for i in range(use_dims)]

        fig_width = max(14, use_dims * 0.22)
        fig_height = max(4, n_agents * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')

        # Clamp z-scores for color range
        vmax = min(3.0, np.max(np.abs(z_matrix)))
        im = ax.imshow(z_matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

        # Annotate highly discriminative cells
        for i in range(n_agents):
            for j in range(use_dims):
                if abs(z_matrix[i, j]) > 1.5:
                    ax.text(j, i, f"{z_matrix[i, j]:.1f}", ha='center', va='center',
                            fontsize=6, color='white' if abs(z_matrix[i, j]) > 2.0 else '#e2e8f0')

        # Group dividers
        prev_end = 0
        for group_name, indices in DIMENSION_GROUPS.items():
            valid = [idx for idx in indices if idx < use_dims]
            if valid:
                end = max(valid)
                if prev_end > 0:
                    ax.axvline(x=prev_end - 0.5, color='#f8fafc', linewidth=1.0, alpha=0.6)
                prev_end = end + 1

        ax.set_yticks(range(n_agents))
        ax.set_yticklabels(agents, color='white', fontsize=9)
        ax.set_xticks(range(use_dims))
        ax.set_xticklabels(col_labels, rotation=90, color='#94a3b8', fontsize=6)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Z-Score', color='white', fontsize=10)
        cbar.ax.tick_params(colors='#94a3b8')

        ax.set_title("Parameter Profile Heatmap (z-scores)",
                      fontsize=14, color='white', fontweight='bold', pad=15)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_str

    # -----------------------------------------------------------------
    # Visualization: parameter waterfall chart
    # -----------------------------------------------------------------

    def generate_parameter_waterfall(
        self,
        agent_a: str,
        agent_b: str,
        top_n: int = 25,
    ) -> Optional[str]:
        """
        Bar chart of signed per-dimension shifts between two agents,
        grouped by dimension category, highlighting significant shifts.
        Returns base64-encoded PNG or None.
        """
        if not HAS_VIZ:
            return None

        analysis = self.analyze_parameter_shifts(agent_a, agent_b)
        if "error" in analysis:
            return None

        # Take top N by effect size
        dims = analysis["per_dimension"][:top_n]

        labels = [d["label"] for d in dims]
        effects = [d["effect_size"] for d in dims]
        significants = [d["significant"] for d in dims]

        fig, ax = plt.subplots(figsize=(10, max(6, len(dims) * 0.3)), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')

        y_pos = np.arange(len(labels))
        colors = []
        for e, sig in zip(effects, significants):
            if not sig:
                colors.append('#64748b')
            elif e > 0:
                colors.append('#ef4444')
            else:
                colors.append('#3b82f6')

        bars = ax.barh(y_pos, effects, color=colors, alpha=0.85, height=0.7)

        # Significance markers
        for i, (bar, sig) in enumerate(zip(bars, significants)):
            if sig:
                w = bar.get_width()
                offset = 0.05 if w >= 0 else -0.05
                ax.text(w + offset, i, '*', color='#fbbf24', fontsize=12,
                        ha='left' if w >= 0 else 'right', va='center', fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color='white', fontsize=8)
        ax.axvline(x=0, color='white', linewidth=0.5)

        ax.set_xlabel("Effect Size (Cohen's d)", color='#94a3b8', fontsize=10)
        ax.set_title(
            f"Parameter Shifts: {agent_a} -> {agent_b}",
            fontsize=13, color='white', fontweight='bold', pad=15,
        )

        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, axis='x', color='#334155', alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ef4444', alpha=0.85, label='Significant increase'),
            Patch(facecolor='#3b82f6', alpha=0.85, label='Significant decrease'),
            Patch(facecolor='#64748b', alpha=0.85, label='Not significant'),
        ]
        ax.legend(handles=legend_elements, loc='lower right',
                  frameon=True, facecolor='#1e293b', edgecolor='#334155',
                  labelcolor='white', fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_str


# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================

class BehavioralDNAExtractor:
    """
    Extracts and analyzes fine-grained behavioral DNA from court sessions.
    """
    
    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.collection: Optional[AgentBehavioralDNACollection] = None
        self.session_data: Dict[str, Any] = {}
        
        # Fine-grained analyzer (created after extraction)
        self._analyzer: Optional[FineGrainedDNAAnalyzer] = None

        # Cached visualizations
        self._galaxy_cache: Optional[str] = None
        self._radar_cache: Dict[str, str] = {}
        self._feature_importance_cache: Optional[str] = None
    
    def extract_from_session(
        self,
        round_results: List[Dict],
        agents_config: Dict[str, Any],
        injection_target: str = "",
        injection_strength: float = 0.0,
        probe: str = "overconfidence",
        case_id: str = "unknown",
        trial_type: str = "jury",
    ) -> AgentBehavioralDNACollection:
        """
        Extract behavioral DNA from a completed session.
        
        Args:
            round_results: List of round data from dashboard
            agents_config: Agent configuration dict
            injection_target: Which agent was injected
            injection_strength: Injection strength
            probe: Probe used
            case_id: Case identifier
            trial_type: Type of trial
            
        Returns:
            Collection of behavioral DNA signatures
        """
        # Build session data
        self.session_data = {
            "round_results": round_results,
            "agents": agents_config,
            "injection_target": injection_target,
            "injection_strength": injection_strength,
            "probe": probe,
            "probes": [probe],
            "case_id": case_id,
            "trial_type": trial_type,
            "session_id": f"court_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }
        
        # Extract DNA
        self.collection = extract_behavioral_dna_from_session(
            self.session_data,
            model_name=self.model_name
        )
        
        # Clear caches
        self._galaxy_cache = None
        self._radar_cache = {}
        self._feature_importance_cache = None

        # Build fine-grained analyzer
        self._analyzer = FineGrainedDNAAnalyzer(self.collection)

        return self.collection
    
    def get_agent_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get summary for a specific agent."""
        if not self.collection or agent_id not in self.collection.signatures:
            return {}
        
        dna = self.collection[agent_id]
        return dna.get_statistics()
    
    def get_all_summaries(self) -> Dict[str, Any]:
        """Get summaries for all agents."""
        if not self.collection:
            return {}
        
        return {
            "agents": {
                aid: dna.get_statistics()
                for aid, dna in self.collection.signatures.items()
            },
            "polarization_index": self.collection.compute_polarization_index(),
            "total_agents": len(self.collection),
            "injected_agents": [
                aid for aid, dna in self.collection.signatures.items()
                if dna.injection_features.is_injected
            ],
        }
    
    def get_fingerprints(self) -> Dict[str, str]:
        """Get behavioral fingerprints for all agents."""
        if not self.collection:
            return {}
        
        return {
            aid: dna.get_behavioral_fingerprint()
            for aid, dna in self.collection.signatures.items()
        }
    
    def compare_agents(
        self,
        agent_a: str,
        agent_b: str,
    ) -> Dict[str, Any]:
        """Compare two agents' behavioral DNA."""
        if not self.collection:
            return {}
        
        if agent_a not in self.collection.signatures or agent_b not in self.collection.signatures:
            return {"error": "Agent not found"}
        
        dna_a = self.collection[agent_a]
        dna_b = self.collection[agent_b]
        
        vec_a = dna_a.to_vector()
        vec_b = dna_b.to_vector()
        
        # Compute distances
        euclidean_dist = float(np.linalg.norm(vec_a - vec_b))
        cosine_sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
        
        # Find most different dimensions
        diffs = np.abs(vec_a - vec_b)
        top_diff_indices = np.argsort(diffs)[-10:][::-1]
        
        top_differences = []
        for idx in top_diff_indices:
            label = FEATURE_LABELS.get(idx, f"Feature {idx}")
            top_differences.append({
                "index": int(idx),
                "label": label,
                "agent_a_value": float(vec_a[idx]),
                "agent_b_value": float(vec_b[idx]),
                "difference": float(diffs[idx]),
            })
        
        # Group differences
        group_diffs = {}
        for group_name, indices in DIMENSION_GROUPS.items():
            valid_indices = [i for i in indices if i < len(vec_a)]
            if valid_indices:
                group_diff = float(np.mean(np.abs(vec_a[valid_indices] - vec_b[valid_indices])))
                group_diffs[group_name] = group_diff
        
        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "euclidean_distance": euclidean_dist,
            "cosine_similarity": cosine_sim,
            "top_differences": top_differences,
            "group_differences": group_diffs,
            "trajectory_a": dna_a.temporal_features.trajectory_archetype.value,
            "trajectory_b": dna_b.temporal_features.trajectory_archetype.value,
        }
    
    def get_injection_analysis(self) -> Dict[str, Any]:
        """Analyze injection effects across agents."""
        if not self.collection:
            return {}
        
        injected = []
        baseline = []
        
        for aid, dna in self.collection.signatures.items():
            stats = {
                "agent_id": aid,
                "role": dna.agent_role.value,
                "mean_score": dna.token_features.score_mean,
                "drift": dna.temporal_features.total_drift,
                "stability": dna.composite_indices.behavioral_stability_index,
                "trajectory": dna.temporal_features.trajectory_archetype.value,
            }
            
            if dna.injection_features.is_injected:
                stats.update({
                    "injection_strength": dna.injection_features.injection_strength,
                    "absorption_rate": dna.injection_features.absorption_rate,
                    "peak_effect": dna.injection_features.peak_effect,
                    "amplification": dna.injection_features.amplification_factor,
                })
                injected.append(stats)
            else:
                stats.update({
                    "resistance": dna.injection_features.resistance_score,
                    "cross_contamination": dna.injection_features.cross_contamination,
                })
                baseline.append(stats)
        
        # Compute effect sizes
        effect_analysis = {}
        if injected and baseline:
            inj_drifts = [a["drift"] for a in injected]
            base_drifts = [a["drift"] for a in baseline]
            
            effect_analysis = {
                "mean_drift_injected": float(np.mean(inj_drifts)),
                "mean_drift_baseline": float(np.mean(base_drifts)),
                "drift_effect_size": float(np.mean(inj_drifts) - np.mean(base_drifts)),
            }
        
        return {
            "injected_agents": injected,
            "baseline_agents": baseline,
            "effect_analysis": effect_analysis,
        }
    
    def get_dimension_breakdown(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed dimension breakdown for an agent."""
        if not self.collection or agent_id not in self.collection.signatures:
            return {}
        
        dna = self.collection[agent_id]
        vec = dna.to_vector()
        
        breakdown = {}
        for group_name, indices in DIMENSION_GROUPS.items():
            group_data = {
                "values": [],
                "labels": [],
                "mean": 0.0,
                "std": 0.0,
            }
            
            for idx in indices:
                if idx < len(vec):
                    group_data["values"].append(float(vec[idx]))
                    group_data["labels"].append(FEATURE_LABELS.get(idx, f"Feature {idx}"))
            
            if group_data["values"]:
                group_data["mean"] = float(np.mean(group_data["values"]))
                group_data["std"] = float(np.std(group_data["values"]))
            
            breakdown[group_name] = group_data
        
        return breakdown
    
    # =========================================================================
    # FINE-GRAINED PARAMETER ANALYSIS (delegates to FineGrainedDNAAnalyzer)
    # =========================================================================

    def _ensure_analyzer(self) -> Optional[FineGrainedDNAAnalyzer]:
        """Lazily create or return the fine-grained analyzer."""
        if self._analyzer is None and self.collection and len(self.collection) >= 2:
            self._analyzer = FineGrainedDNAAnalyzer(self.collection)
        return self._analyzer

    def get_parameter_profiles(self) -> Dict[str, Any]:
        """Per-dimension z-score profiles for all agents."""
        analyzer = self._ensure_analyzer()
        if not analyzer:
            return {}
        return analyzer.compute_parameter_profiles()

    def get_discriminative_dimensions(
        self, agent_id: str, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get the dimensions that uniquely characterize an agent."""
        analyzer = self._ensure_analyzer()
        if not analyzer:
            return []
        return analyzer.get_discriminative_dimensions(agent_id, top_n)

    def get_parameter_shift_analysis(
        self, agent_a: str, agent_b: str
    ) -> Dict[str, Any]:
        """
        Enhanced pairwise comparison with per-dimension signed shifts,
        effect sizes, group summaries, and correlated shift clusters.
        """
        analyzer = self._ensure_analyzer()
        if not analyzer:
            return {}
        return analyzer.analyze_parameter_shifts(agent_a, agent_b)

    def match_output_to_agent(
        self,
        output_vector: np.ndarray,
        top_n_dims: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Identify which agent an output DNA vector most resembles using
        parameter-level profile matching (discriminative dims + direction
        agreement) rather than raw vector distance.
        """
        analyzer = self._ensure_analyzer()
        if not analyzer:
            return []
        return analyzer.match_agent_by_parameters(output_vector, top_n_dims)

    def get_population_statistics(self) -> Dict[str, Any]:
        """Population-level per-dimension statistics and correlations."""
        analyzer = self._ensure_analyzer()
        if not analyzer:
            return {}
        return analyzer.compute_population_statistics()

    def generate_parameter_heatmap(self) -> Optional[str]:
        """Generate z-score heatmap across agents and dimensions."""
        analyzer = self._ensure_analyzer()
        if not analyzer:
            return None
        return analyzer.generate_parameter_heatmap()

    def generate_parameter_waterfall(
        self, agent_a: str, agent_b: str, top_n: int = 25
    ) -> Optional[str]:
        """Generate waterfall chart of parameter shifts between two agents."""
        analyzer = self._ensure_analyzer()
        if not analyzer:
            return None
        return analyzer.generate_parameter_waterfall(agent_a, agent_b, top_n)

    # =========================================================================
    # SAE FINGERPRINTING INTEGRATION
    # =========================================================================

    def enrich_with_sae(
        self,
        reader_model_name: Optional[str] = None,
        sae_path: Optional[str] = None,
        layer_idx: int = 24,
        precomputed_fingerprints: Optional[Dict[str, np.ndarray]] = None,
        feature_labels: Optional[Dict[int, str]] = None,
    ):
        """
        Enrich all agents' DNA with SAE-based features.

        This runs the SAE fingerprinting pipeline (Steps 2-4) on each
        agent's responses and adds SAE summary features to their DNA vectors.

        Two modes:
        1. Live mode: Provide reader_model_name + sae_path to extract features
           in real-time using the reader LLM + SAE.
        2. Precomputed mode: Provide precomputed_fingerprints dict mapping
           agent_id → SAE frequency vector (shape d_sae).

        Args:
            reader_model_name: HuggingFace model name for reader LLM
            sae_path: Path to pretrained SAE checkpoint
            layer_idx: Reader model layer to hook for activations
            precomputed_fingerprints: Dict of agent_id → frequency vectors
            feature_labels: Optional SAE latent label mapping
        """
        if not self.collection:
            return

        if precomputed_fingerprints is not None:
            # Precomputed mode: directly apply frequency vectors
            all_freqs = list(precomputed_fingerprints.values())
            population_avg = np.mean(all_freqs, axis=0) if all_freqs else None

            for agent_id, freq_vec in precomputed_fingerprints.items():
                if agent_id in self.collection.signatures:
                    self.collection[agent_id].enrich_with_sae(
                        sae_frequencies=freq_vec,
                        feature_labels=feature_labels,
                        population_frequencies=population_avg,
                    )

        elif reader_model_name and sae_path:
            if not HAS_SAE:
                raise ImportError(
                    "SAE fingerprinting module not available. "
                    "Ensure core/sae_fingerprint.py is accessible."
                )

            # Live mode: extract SAE features from agent responses
            extractor = SAEFeatureExtractor(
                reader_model_name=reader_model_name,
                sae_path=sae_path,
                layer_idx=layer_idx,
                feature_labels=feature_labels,
            )

            try:
                # Step 2-3: Extract binary vectors per agent
                agent_fingerprints = {}
                for agent_id, dna in self.collection.signatures.items():
                    if dna._statements:
                        binary_vectors = extractor.extract_binary_vectors(
                            dna._statements
                        )
                        # Step 4: Aggregate into per-agent fingerprint
                        fp = SAEModelFingerprint.from_binary_vectors(
                            model_id=agent_id,
                            binary_vectors=binary_vectors,
                            feature_labels=feature_labels,
                        )
                        agent_fingerprints[agent_id] = fp.frequencies

                # Compute population average
                all_freqs = list(agent_fingerprints.values())
                population_avg = np.mean(all_freqs, axis=0) if all_freqs else None

                # Apply to each agent
                for agent_id, frequencies in agent_fingerprints.items():
                    self.collection[agent_id].enrich_with_sae(
                        sae_frequencies=frequencies,
                        feature_labels=feature_labels,
                        population_frequencies=population_avg,
                    )
            finally:
                extractor.cleanup()
        else:
            raise ValueError(
                "Provide either (reader_model_name + sae_path) for live "
                "extraction, or precomputed_fingerprints for offline mode."
            )

        # Rebuild analyzer to include SAE dimensions
        if self.collection and len(self.collection) >= 2:
            self._analyzer = FineGrainedDNAAnalyzer(self.collection)

    def get_sae_fingerprint_analysis(
        self,
        agent_id: str,
    ) -> Dict[str, Any]:
        """
        Get SAE fingerprint analysis for a specific agent.

        Returns SAE feature summary including top activated features,
        sparsity metrics, and distributional properties.
        """
        if not self.collection or agent_id not in self.collection.signatures:
            return {}

        dna = self.collection[agent_id]
        if not dna.sae_features.is_populated:
            return {"error": "SAE features not populated. Call enrich_with_sae() first."}

        sae = dna.sae_features
        return {
            "agent_id": agent_id,
            "n_active_features": sae.n_active_features,
            "activation_sparsity": sae.activation_sparsity,
            "mean_activation_frequency": sae.mean_activation_frequency,
            "frequency_entropy": sae.frequency_entropy,
            "frequency_gini": sae.frequency_gini,
            "n_unique_features": sae.n_unique_features,
            "feature_overlap_with_population": sae.feature_overlap_with_population,
            "top_features": [
                {"label": label, "frequency": freq, "index": idx}
                for label, freq, idx in zip(
                    sae.top_feature_labels,
                    sae.top_feature_frequencies,
                    sae.top_feature_indices,
                )
            ],
        }

    def compare_agents_sae(
        self,
        agent_a: str,
        agent_b: str,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Compare two agents using their SAE fingerprints (dataset diffing).

        This performs Step 5 of the SAE pipeline at the agent level,
        identifying the semantic features that most distinguish the two agents.

        Args:
            agent_a: First agent ID
            agent_b: Second agent ID
            top_n: Number of top distinguishing features to return

        Returns:
            Dict with top distinguishing SAE features and similarity metrics
        """
        if not self.collection:
            return {}

        for aid in [agent_a, agent_b]:
            if aid not in self.collection.signatures:
                return {"error": f"Agent '{aid}' not found"}

        dna_a = self.collection[agent_a]
        dna_b = self.collection[agent_b]

        if not dna_a.sae_features.is_populated or not dna_b.sae_features.is_populated:
            return {"error": "SAE features not populated for both agents. Call enrich_with_sae() first."}

        freq_a = dna_a.sae_features._full_frequencies
        freq_b = dna_b.sae_features._full_frequencies

        if freq_a is None or freq_b is None:
            return {"error": "Full SAE frequency vectors not available"}

        # Compute diff
        diff = freq_b - freq_a
        sorted_indices = np.argsort(np.abs(diff))[::-1]

        # Merge feature labels
        labels_a = dict(zip(dna_a.sae_features.top_feature_indices,
                           dna_a.sae_features.top_feature_labels))
        labels_b = dict(zip(dna_b.sae_features.top_feature_indices,
                           dna_b.sae_features.top_feature_labels))
        all_labels = {**labels_a, **labels_b}

        top_diffs = []
        for idx in sorted_indices[:top_n]:
            idx = int(idx)
            top_diffs.append({
                "index": idx,
                "label": all_labels.get(idx, f"SAE Latent #{idx}"),
                "freq_a": float(freq_a[idx]),
                "freq_b": float(freq_b[idx]),
                "diff": float(diff[idx]),
                "diff_pct": float(diff[idx]) * 100,
                "direction": "higher_in_b" if diff[idx] > 0 else "higher_in_a",
            })

        # Cosine similarity
        dot = np.dot(freq_a, freq_b)
        norm_a = np.linalg.norm(freq_a)
        norm_b = np.linalg.norm(freq_b)
        cosine_sim = float(dot / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0

        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "cosine_similarity": cosine_sim,
            "mean_absolute_diff": float(np.mean(np.abs(diff))),
            "top_distinguishing_features": top_diffs,
        }

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def generate_galaxy_visualization(
        self,
        method: str = "pca",
        highlight_dimensions: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Generate DNA Galaxy visualization.
        
        Args:
            method: Dimensionality reduction method ("pca", "tsne")
            highlight_dimensions: Dimension groups to highlight
            
        Returns:
            Base64-encoded PNG image
        """
        if not HAS_VIZ or not self.collection or len(self.collection) < 2:
            return None
        
        # Get vectors and metadata
        agents = list(self.collection.signatures.keys())
        vectors = np.array([self.collection[a].to_vector() for a in agents])
        
        # Standardize
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors)
        
        # Reduce dimensions
        if method == "tsne" and len(agents) > 3:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(agents) - 1))
        else:
            reducer = PCA(n_components=2)
        
        coords = reducer.fit_transform(vectors_scaled)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        
        # Role colors
        role_colors = {
            AgentRole.JUDGE: '#f59e0b',
            AgentRole.PLAINTIFF_COUNSEL: '#ef4444',
            AgentRole.DEFENSE_COUNSEL: '#3b82f6',
            AgentRole.JURY_FOREPERSON: '#8b5cf6',
            AgentRole.CLERK: '#6b7280',
            AgentRole.UNKNOWN: '#64748b',
        }
        
        # Plot each agent
        for i, agent_id in enumerate(agents):
            dna = self.collection[agent_id]
            x, y = coords[i]
            
            color = role_colors.get(dna.agent_role, '#64748b')
            is_injected = dna.injection_features.is_injected
            
            # Size based on activity/engagement
            size = 200 + dna.composite_indices.engagement_index * 300
            
            # Marker based on trajectory
            archetype = dna.temporal_features.trajectory_archetype
            marker = {
                TrajectoryArchetype.STEADY_STATE: 'o',
                TrajectoryArchetype.ESCALATING: '^',
                TrajectoryArchetype.DE_ESCALATING: 'v',
                TrajectoryArchetype.VOLATILE: 'd',
                TrajectoryArchetype.PHASE_SHIFT: 's',
            }.get(archetype, 'o')
            
            # Plot
            if is_injected:
                # Glow effect for injected
                ax.scatter(x, y, c=color, s=size * 2, alpha=0.2, marker=marker)
                ax.scatter(x, y, c=color, s=size * 1.5, alpha=0.3, marker=marker)
                ax.scatter(x, y, c=color, s=size, marker=marker, 
                          edgecolors='white', linewidths=3, zorder=10)
            else:
                ax.scatter(x, y, c=color, s=size, marker=marker,
                          edgecolors='white', linewidths=1.5, zorder=10)
            
            # Label
            label = dna.agent_role.value.replace('_', '\n')
            if is_injected:
                label += '\n⚡'
            
            txt = ax.annotate(
                label, (x, y),
                xytext=(12, 12), textcoords='offset points',
                fontsize=9, color='white',
                fontweight='bold' if is_injected else 'normal',
                ha='left', va='bottom',
            )
            txt.set_path_effects([
                path_effects.withStroke(linewidth=3, foreground='#0f172a')
            ])
        
        # Draw connection lines between opposing counsel
        plaintiff_idx = None
        defense_idx = None
        for i, aid in enumerate(agents):
            if self.collection[aid].agent_role == AgentRole.PLAINTIFF_COUNSEL:
                plaintiff_idx = i
            elif self.collection[aid].agent_role == AgentRole.DEFENSE_COUNSEL:
                defense_idx = i
        
        if plaintiff_idx is not None and defense_idx is not None:
            ax.plot(
                [coords[plaintiff_idx, 0], coords[defense_idx, 0]],
                [coords[plaintiff_idx, 1], coords[defense_idx, 1]],
                color='#64748b', linestyle='--', alpha=0.5, linewidth=1,
            )
        
        # Style
        ax.set_title(
            "🌌 Behavioral DNA Galaxy",
            fontsize=16, color='white', fontweight='bold', pad=20,
        )
        
        if hasattr(reducer, 'explained_variance_ratio_'):
            var1, var2 = reducer.explained_variance_ratio_[:2] * 100
            ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", color='#94a3b8', fontsize=10)
            ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", color='#94a3b8', fontsize=10)
        else:
            ax.set_xlabel("Dimension 1", color='#94a3b8')
            ax.set_ylabel("Dimension 2", color='#94a3b8')
        
        ax.grid(True, color='#334155', alpha=0.3, linestyle='--')
        
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.tick_params(colors='#94a3b8')
        
        # Legend
        legend_elements = []
        for role, color in role_colors.items():
            if role != AgentRole.UNKNOWN:
                legend_elements.append(
                    plt.scatter([], [], c=color, s=100, label=role.value.replace('_', ' ').title())
                )
        
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            frameon=True,
            facecolor='#1e293b',
            edgecolor='#334155',
            labelcolor='white',
            fontsize=8,
        )
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        self._galaxy_cache = img_str
        return img_str
    
    def generate_radar_chart(self, agent_id: str) -> Optional[str]:
        """Generate radar chart for agent's behavioral dimensions."""
        if not HAS_VIZ or not self.collection or agent_id not in self.collection.signatures:
            return None
        
        if agent_id in self._radar_cache:
            return self._radar_cache[agent_id]
        
        dna = self.collection[agent_id]
        
        # Get dimension group means
        breakdown = self.get_dimension_breakdown(agent_id)
        
        categories = list(DIMENSION_GROUPS.keys())
        values = [breakdown[cat]["mean"] for cat in categories]
        
        # Normalize to 0-1
        values = np.array(values)
        values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
        
        # Close the radar
        values = np.concatenate((values, [values[0]]))
        angles = np.linspace(0, 2 * np.pi, len(categories) + 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        
        # Plot
        color = {
            AgentRole.JUDGE: '#f59e0b',
            AgentRole.PLAINTIFF_COUNSEL: '#ef4444',
            AgentRole.DEFENSE_COUNSEL: '#3b82f6',
            AgentRole.JURY_FOREPERSON: '#8b5cf6',
        }.get(dna.agent_role, '#6366f1')
        
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='white', fontsize=9)
        
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['', '', ''], color='#94a3b8')
        
        ax.grid(True, color='#334155', alpha=0.5)
        
        title = f"{dna.agent_role.value.replace('_', ' ').title()}"
        if dna.injection_features.is_injected:
            title += " ⚡"
        
        ax.set_title(title, fontsize=14, color='white', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        self._radar_cache[agent_id] = img_str
        return img_str
    
    def generate_feature_importance_chart(self) -> Optional[str]:
        """Generate chart showing most discriminative features between injected and baseline."""
        if not HAS_VIZ or not self.collection:
            return None
        
        injected_vecs = []
        baseline_vecs = []
        
        for dna in self.collection:
            vec = dna.to_vector()
            if dna.injection_features.is_injected:
                injected_vecs.append(vec)
            else:
                baseline_vecs.append(vec)
        
        if not injected_vecs or not baseline_vecs:
            return None
        
        injected_mean = np.mean(injected_vecs, axis=0)
        baseline_mean = np.mean(baseline_vecs, axis=0)
        
        # Compute differences
        diffs = injected_mean - baseline_mean
        
        # Get top features
        top_indices = np.argsort(np.abs(diffs))[-15:][::-1]
        
        top_labels = [FEATURE_LABELS.get(i, f"Feature {i}") for i in top_indices]
        top_values = [diffs[i] for i in top_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        
        colors = ['#ef4444' if v > 0 else '#3b82f6' for v in top_values]
        
        y_pos = np.arange(len(top_labels))
        ax.barh(y_pos, top_values, color=colors, alpha=0.8, height=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_labels, color='white', fontsize=9)
        
        ax.axvline(x=0, color='white', linewidth=0.5)
        
        ax.set_xlabel("Difference (Injected - Baseline)", color='#94a3b8', fontsize=10)
        ax.set_title(
            "🎯 Feature Importance: Injection Effects",
            fontsize=14, color='white', fontweight='bold', pad=15,
        )
        
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, axis='x', color='#334155', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        self._feature_importance_cache = img_str
        return img_str
    
    def generate_trajectory_comparison(self) -> Optional[str]:
        """Generate trajectory comparison chart."""
        if not HAS_VIZ or not self.collection:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        
        role_colors = {
            AgentRole.JUDGE: '#f59e0b',
            AgentRole.PLAINTIFF_COUNSEL: '#ef4444',
            AgentRole.DEFENSE_COUNSEL: '#3b82f6',
            AgentRole.JURY_FOREPERSON: '#8b5cf6',
        }
        
        max_rounds = 0
        for dna in self.collection:
            scores = dna._round_scores
            if not scores:
                continue
            
            max_rounds = max(max_rounds, len(scores))
            x = np.arange(1, len(scores) + 1)
            
            color = role_colors.get(dna.agent_role, '#64748b')
            style = '--' if dna.injection_features.is_injected else '-'
            marker = 'D' if dna.injection_features.is_injected else 'o'
            
            label = dna.agent_role.value.replace('_', ' ').title()
            if dna.injection_features.is_injected:
                label += ' ⚡'
            
            ax.plot(x, scores, color=color, linestyle=style, marker=marker,
                   markersize=6, linewidth=2, label=label, alpha=0.8)
        
        ax.set_xlabel("Round", color='#94a3b8', fontsize=11)
        ax.set_ylabel("Probe Score", color='#94a3b8', fontsize=11)
        ax.set_title(
            "📈 Behavioral Trajectories",
            fontsize=14, color='white', fontweight='bold', pad=15,
        )
        
        ax.axhline(y=0, color='#64748b', linestyle='-', linewidth=0.5)
        
        ax.legend(
            loc='upper right',
            frameon=True,
            facecolor='#1e293b',
            edgecolor='#334155',
            labelcolor='white',
            fontsize=9,
        )
        
        ax.grid(True, color='#334155', alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.tick_params(colors='#94a3b8')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return img_str
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def get_full_analysis(self) -> Dict[str, Any]:
        """Get complete analysis results."""
        if not self.collection:
            return {"error": "No data extracted"}
        
        # Fine-grained parameter analysis
        agents = list(self.collection.signatures.keys())
        parameter_fingerprints = {
            aid: self.get_discriminative_dimensions(aid)
            for aid in agents
        }

        # SAE fingerprint analysis (if enriched)
        sae_analysis = {}
        has_sae = any(
            dna.sae_features.is_populated
            for dna in self.collection.signatures.values()
        )
        if has_sae:
            sae_analysis = {
                "per_agent": {
                    aid: self.get_sae_fingerprint_analysis(aid)
                    for aid in agents
                },
            }

        return {
            "summaries": self.get_all_summaries(),
            "fingerprints": self.get_fingerprints(),
            "injection_analysis": self.get_injection_analysis(),
            "parameter_analysis": {
                "population_statistics": self.get_population_statistics(),
                "parameter_fingerprints": parameter_fingerprints,
            },
            "sae_analysis": sae_analysis,
            "visualizations": {
                "galaxy": self.generate_galaxy_visualization(),
                "feature_importance": self.generate_feature_importance_chart(),
                "trajectories": self.generate_trajectory_comparison(),
                "parameter_heatmap": self.generate_parameter_heatmap(),
            },
            "per_agent_radars": {
                aid: self.generate_radar_chart(aid)
                for aid in agents
            },
            "dimension_labels": FEATURE_LABELS,
            "dimension_groups": DIMENSION_GROUPS,
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export all data as dictionary."""
        if not self.collection:
            return {}
        
        return {
            "metadata": {
                "model_name": self.model_name,
                "extraction_time": datetime.now().isoformat(),
                "num_agents": len(self.collection),
            },
            "agents": {
                aid: {
                    "statistics": dna.get_statistics(),
                    "vector": dna.to_vector().tolist(),
                    "fingerprint": dna.get_behavioral_fingerprint(),
                }
                for aid, dna in self.collection.signatures.items()
            },
            "analysis": {
                "injection": self.get_injection_analysis(),
                "polarization": self.collection.compute_polarization_index(),
            },
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BehavioralDNAExtractor',
    'FineGrainedDNAAnalyzer',
    'FEATURE_LABELS',
    'DIMENSION_GROUPS',
]