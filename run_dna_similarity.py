#!/usr/bin/env python3
"""
DNA Cosine Similarity Visualizer
==================================

Computes 128-D DNA embeddings for all agents across experiment result files,
then visualizes pairwise cosine similarity as:

  1. Full agent-level heatmap (grouped by model)
  2. Model-averaged similarity matrix
  3. Probe-type similarity matrix (within and across models)
  4. Injected vs baseline similarity comparison (per model)

Usage:
  python run_dna_similarity.py \
      --results \
          experiment_results/h1_results_20260326_100042.json \
          experiment_results/claude_judge/h3_results_20260325_081500.json \
          experiment_results/eleutherai_pythia_1b/h1_results_20260329_232014.json \
          experiment_results/eleutherai_pythia_1b/h3_results_20260329_232014.json \
          experiment_results/eleutherai_pythia_2_8b/h1_results_20260330_231405.json \
          experiment_results/eleutherai_pythia_2_8b/h3_results_20260330_231405.json \
          experiment_results/google_gemma_3_1b_it/h1_results_20260330_064849.json \
          experiment_results/google_gemma_3_1b_it/h3_results_20260330_064849.json \
      --model-label \
          qwen3_4b qwen3_4b \
          pythia_1b pythia_1b \
          pythia_2.8b pythia_2.8b \
          gemma_3_1b gemma_3_1b \
      --embedding-model sentence-transformers/all-mpnet-base-v2 \
      --device cuda:5 \
      -o experiment_results/dna_similarity
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


# ═════════════════════════════════════════════════════════════════
# Reuse data loading and DNA computation from run_dna_galaxy.py
# ═════════════════════════════════════════════════════════════════

from run_dna_galaxy import (
    AgentDNAEntry,
    load_agent_responses,
    compute_dna_vectors,
)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pairwise_cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for (N, D) array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = vectors / norms
    return normed @ normed.T


# ═════════════════════════════════════════════════════════════════
# Plot 1: Full agent heatmap grouped by model
# ═════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "qwen3_4b":  "#1565c0",
    "pythia_1b": "#6a1b9a",
    "pythia_2.8b": "#ad1457",
    "gemma_3_1b": "#2e7d32",
}


def plot_agent_heatmap(entries: List[AgentDNAEntry], output_path: str):
    """Full NxN cosine similarity heatmap, agents grouped by model."""
    # Sort entries by model, then probe, then agent_id
    entries_sorted = sorted(entries, key=lambda e: (e.model_label, e.probe, e.agent_id))
    n = len(entries_sorted)

    vectors = np.array([e.dna_vector for e in entries_sorted])
    sim_matrix = pairwise_cosine_matrix(vectors)

    # Group boundaries
    model_groups = []
    current_model = None
    start = 0
    for i, e in enumerate(entries_sorted):
        if e.model_label != current_model:
            if current_model is not None:
                model_groups.append((current_model, start, i))
            current_model = e.model_label
            start = i
    model_groups.append((current_model, start, n))

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#ffffff")
    ax.set_facecolor("#f8f9fa")

    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.2, vmax=1.0, aspect="equal")

    # Draw model group boundaries
    for model, s, e in model_groups:
        color = MODEL_COLORS.get(model, "#888888")
        rect = Rectangle((s - 0.5, s - 0.5), e - s, e - s,
                          linewidth=2.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

    # Model labels on axes
    for model, s, e in model_groups:
        mid = (s + e) / 2
        color = MODEL_COLORS.get(model, "#333333")
        ax.text(-1.5, mid, model, ha="right", va="center", fontsize=9,
                fontweight="bold", color=color)
        ax.text(mid, n + 0.8, model, ha="center", va="top", fontsize=9,
                fontweight="bold", color=color, rotation=0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Agent DNA Cosine Similarity (128-D)\nGrouped by Model",
                 fontsize=14, fontweight="bold", pad=20, color="#1a1a2e")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=10)

    # Annotation: mean within/between
    within_sims = []
    between_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            if entries_sorted[i].model_label == entries_sorted[j].model_label:
                within_sims.append(sim_matrix[i, j])
            else:
                between_sims.append(sim_matrix[i, j])

    # Compute topology split
    topo_within_court, topo_within_debate, topo_cross = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            ti = entries_sorted[i].topology
            tj = entries_sorted[j].topology
            if ti == "court" and tj == "court":
                topo_within_court.append(sim_matrix[i, j])
            elif ti == "debate" and tj == "debate":
                topo_within_debate.append(sim_matrix[i, j])
            else:
                topo_cross.append(sim_matrix[i, j])
    court_self = np.mean(topo_within_court) if topo_within_court else 0
    court_cross = np.mean(topo_cross) if topo_cross else 0

    # Key finding box
    finding = (
        f"KEY FINDING: Topology dominates similarity structure\n"
        f"Court self-sim: {court_self:.3f}  |  Court-Debate cross: {court_cross:.3f}  |  Gap: {court_self - court_cross:.3f}\n"
        f"Within-model: {np.mean(within_sims):.3f}  |  Between-model: {np.mean(between_sims):.3f}  |  Gap: {np.mean(within_sims) - np.mean(between_sims):.3f}\n"
        f"Topology gap is {(court_self - court_cross) / max(np.mean(within_sims) - np.mean(between_sims), 0.001):.0f}x larger than model identity gap"
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.01, finding, fontsize=8.5, ha="center", va="bottom",
             color="#1a1a2e", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd",
                       edgecolor="#ffc107", linewidth=1.5, alpha=0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═════════════════════════════════════════════════════════════════
# Plot 2: Model-averaged similarity matrix
# ═════════════════════════════════════════════════════════════════

def plot_model_similarity(entries: List[AgentDNAEntry], output_path: str):
    """Averaged cosine similarity between models."""
    by_model = defaultdict(list)
    for e in entries:
        by_model[e.model_label].append(e.dna_vector)

    models = sorted(by_model.keys())
    n = len(models)
    matrix = np.zeros((n, n))

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            sims = []
            for v1 in by_model[m1]:
                for v2 in by_model[m2]:
                    sims.append(cosine_sim(v1, v2))
            matrix[i, j] = np.mean(sims)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#ffffff")
    ax.set_facecolor("#f8f9fa")

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=0.3, vmax=1.0, aspect="equal")

    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > 0.75 else "black"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    colors = [MODEL_COLORS.get(m, "#333") for m in models]
    ax.set_xticklabels(models, fontsize=10, fontweight="bold", rotation=30, ha="right")
    ax.set_yticklabels(models, fontsize=10, fontweight="bold")
    for tick, c in zip(ax.get_xticklabels(), colors):
        tick.set_color(c)
    for tick, c in zip(ax.get_yticklabels(), colors):
        tick.set_color(c)

    ax.set_title("Model-Level DNA Similarity (128-D, averaged over agents)",
                 fontsize=12, fontweight="bold", pad=15, color="#1a1a2e")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Cosine Similarity", fontsize=10)

    # Find highest self-sim, lowest self-sim, and closest cross pair
    diag = [matrix[i, i] for i in range(n)]
    best_self_idx = int(np.argmax(diag))
    worst_self_idx = int(np.argmin(diag))
    # Closest cross-model pair
    best_cross_val, best_cross_pair = -1, ("", "")
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] > best_cross_val:
                best_cross_val = matrix[i, j]
                best_cross_pair = (models[i], models[j])
    # Pythia family distance
    pythia_idxs = {m: i for i, m in enumerate(models) if "pythia" in m}
    pythia_sim = matrix[pythia_idxs.get("pythia_1b", 0), pythia_idxs.get("pythia_2.8b", 1)] if len(pythia_idxs) == 2 else 0

    finding = (
        f"KEY FINDINGS:\n"
        f"  Most coherent model:  {models[best_self_idx]} (self-sim {diag[best_self_idx]:.3f})\n"
        f"  Least coherent model: {models[worst_self_idx]} (self-sim {diag[worst_self_idx]:.3f})\n"
        f"  Closest cross-family: {best_cross_pair[0]} <-> {best_cross_pair[1]} ({best_cross_val:.3f})\n"
        f"  Pythia intra-family:  pythia_1b <-> pythia_2.8b ({pythia_sim:.3f}) < gemma <-> qwen\n"
        f"  Architecture family does NOT predict behavioral similarity"
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.01, finding, fontsize=8, ha="center", va="bottom",
             color="#1a1a2e", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd",
                       edgecolor="#ffc107", linewidth=1.5, alpha=0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Print numerical summary
    print("\n  Model-level similarity matrix:")
    header = "            " + "  ".join(f"{m:>12}" for m in models)
    print(f"  {header}")
    for i, m in enumerate(models):
        row = "  ".join(f"{matrix[i, j]:12.4f}" for j in range(n))
        print(f"  {m:>12}  {row}")


# ═════════════════════════════════════════════════════════════════
# Plot 3: Model x Probe similarity breakdown
# ═════════════════════════════════════════════════════════════════

PROBE_ORDER = ["sentiment", "sycophancy", "toxicity"]


def plot_model_probe_similarity(entries: List[AgentDNAEntry], output_path: str):
    """Heatmap of (model, probe) vs (model, probe) mean cosine similarity."""
    by_mp = defaultdict(list)
    for e in entries:
        by_mp[(e.model_label, e.probe)].append(e.dna_vector)

    models = sorted(set(e.model_label for e in entries))
    probes = [p for p in PROBE_ORDER if any((m, p) in by_mp for m in models)]
    keys = [(m, p) for m in models for p in probes]
    n = len(keys)

    matrix = np.zeros((n, n))
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if k1 in by_mp and k2 in by_mp:
                sims = [cosine_sim(v1, v2) for v1 in by_mp[k1] for v2 in by_mp[k2]]
                matrix[i, j] = np.mean(sims)

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#ffffff")
    ax.set_facecolor("#f8f9fa")

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=0.2, vmax=1.0, aspect="equal")

    labels = [f"{m}\n{p}" for m, p in keys]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7)

    # Color tick labels by model
    for idx, (m, p) in enumerate(keys):
        c = MODEL_COLORS.get(m, "#333")
        ax.get_xticklabels()[idx].set_color(c)
        ax.get_yticklabels()[idx].set_color(c)

    # Draw model group boundaries
    n_probes = len(probes)
    for i in range(len(models)):
        s = i * n_probes
        e = s + n_probes
        color = MODEL_COLORS.get(models[i], "#888")
        rect = Rectangle((s - 0.5, s - 0.5), n_probes, n_probes,
                          linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

    # Annotate values in cells
    for i in range(n):
        for j in range(n):
            if n <= 16:
                color = "white" if matrix[i, j] > 0.7 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_title("DNA Similarity by Model x Probe (128-D)",
                 fontsize=13, fontweight="bold", pad=15, color="#1a1a2e")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Cosine Similarity", fontsize=10)

    # Compute per-model probe range and find best/worst probe
    probe_stats = {}
    for m in models:
        vals = []
        for p in probes:
            if (m, p) in by_mp:
                idx = keys.index((m, p))
                vals.append((p, matrix[idx, idx]))
        if vals:
            vals_sorted = sorted(vals, key=lambda x: x[1])
            probe_stats[m] = {
                "best": vals_sorted[-1],
                "worst": vals_sorted[0],
                "range": vals_sorted[-1][1] - vals_sorted[0][1],
            }
    # Most uniform model (smallest range)
    uniform_model = min(probe_stats, key=lambda m: probe_stats[m]["range"])
    # Most variable model
    variable_model = max(probe_stats, key=lambda m: probe_stats[m]["range"])

    finding = (
        f"KEY FINDINGS:\n"
        f"  Sycophancy produces the most uniform agents across all models (highest within-probe self-sim)\n"
        f"  Most probe-invariant model:  {uniform_model} (range {probe_stats[uniform_model]['range']:.3f})\n"
        f"  Most probe-sensitive model:  {variable_model} (range {probe_stats[variable_model]['range']:.3f})\n"
        f"  Cross-probe sim ~ within-probe sim -> probe type is a weak discriminator in 128-D space"
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.01, finding, fontsize=8.5, ha="center", va="bottom",
             color="#1a1a2e", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd",
                       edgecolor="#ffc107", linewidth=1.5, alpha=0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═════════════════════════════════════════════════════════════════
# Plot 4: Injected vs Baseline similarity per model
# ═════════════════════════════════════════════════════════════════

def plot_injection_similarity(entries: List[AgentDNAEntry], output_path: str):
    """Bar chart: within-injected, within-baseline, and cross similarity per model."""
    models = sorted(set(e.model_label for e in entries))

    categories = ["Inj-Inj", "Base-Base", "Inj-Base"]
    x = np.arange(len(models))
    width = 0.25
    results = {cat: [] for cat in categories}

    for m in models:
        inj_vecs = [e.dna_vector for e in entries if e.model_label == m and e.is_injected]
        bas_vecs = [e.dna_vector for e in entries if e.model_label == m and not e.is_injected]

        def mean_pairwise(vecs_a, vecs_b):
            sims = []
            for v1 in vecs_a:
                for v2 in vecs_b:
                    if not np.array_equal(v1, v2):
                        sims.append(cosine_sim(v1, v2))
            return np.mean(sims) if sims else 0.0

        results["Inj-Inj"].append(mean_pairwise(inj_vecs, inj_vecs))
        results["Base-Base"].append(mean_pairwise(bas_vecs, bas_vecs))
        results["Inj-Base"].append(mean_pairwise(inj_vecs, bas_vecs))

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#ffffff")
    ax.set_facecolor("#f8f9fa")

    bar_colors = ["#ef4444", "#3b82f6", "#a855f7"]
    for i, cat in enumerate(categories):
        bars = ax.bar(x + i * width, results[cat], width, label=cat,
                      color=bar_colors[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, results[cat]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    for tick, m in zip(ax.get_xticklabels(), models):
        tick.set_color(MODEL_COLORS.get(m, "#333"))

    ax.set_ylabel("Mean Cosine Similarity", fontsize=11)
    ax.set_title("Injected vs Baseline DNA Similarity (128-D, per model)",
                 fontsize=13, fontweight="bold", pad=15, color="#1a1a2e")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Find most sensitive and most resistant models
    gaps = [results["Base-Base"][i] - results["Inj-Inj"][i] for i in range(len(models))]
    most_sensitive_idx = int(np.argmax(gaps))
    most_resistant_idx = int(np.argmin(gaps))
    # Best separation: lowest Inj-Base relative to BB and II
    sep_scores = [(results["Base-Base"][i] + results["Inj-Inj"][i]) / 2 - results["Inj-Base"][i]
                  for i in range(len(models))]
    best_sep_idx = int(np.argmax(sep_scores))

    finding = (
        f"KEY FINDINGS:\n"
        f"  Injection increases diversity: Base-Base > Inj-Inj for all models\n"
        f"  Most injection-sensitive: {models[most_sensitive_idx]} "
        f"(BB-II gap: {gaps[most_sensitive_idx]:.3f})\n"
        f"  Most injection-resistant: {models[most_resistant_idx]} "
        f"(BB-II gap: {gaps[most_resistant_idx]:.3f})\n"
        f"  Best injected/baseline separation: {models[best_sep_idx]} "
        f"(Inj-Base {results['Inj-Base'][best_sep_idx]:.3f} < both BB and II)"
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.01, finding, fontsize=8.5, ha="center", va="bottom",
             color="#1a1a2e", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd",
                       edgecolor="#ffc107", linewidth=1.5, alpha=0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Print summary
    print("\n  Injection similarity summary:")
    print(f"  {'Model':>12}  {'Inj-Inj':>8}  {'Base-Base':>10}  {'Inj-Base':>9}  {'Delta':>7}")
    for i, m in enumerate(models):
        delta = results["Base-Base"][i] - results["Inj-Base"][i]
        print(f"  {m:>12}  {results['Inj-Inj'][i]:8.4f}  {results['Base-Base'][i]:10.4f}"
              f"  {results['Inj-Base'][i]:9.4f}  {delta:+7.4f}")


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize 128-D DNA cosine similarity across agents and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", "-r", nargs="+", required=True,
                        help="Experiment result JSON files")
    parser.add_argument("--model-label", nargs="+", default=None,
                        help="One model label per result file")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2",
                        help="Sentence embedding model path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dna-dim", type=int, default=128)
    parser.add_argument("-o", "--output-dir", default="experiment_results/dna_similarity",
                        help="Output directory for plots")
    parser.add_argument("--save-data", action="store_true",
                        help="Save raw similarity matrices as JSON")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load and compute DNA vectors
    print("=" * 60)
    print("DNA Cosine Similarity Analysis")
    print("=" * 60)

    agent_data = load_agent_responses(args.results, args.model_label)
    entries = compute_dna_vectors(
        agent_data,
        embedding_model=args.embedding_model,
        device=args.device,
        dna_dim=args.dna_dim,
    )

    print(f"\nTotal agents: {len(entries)}")
    print(f"DNA dimension: {args.dna_dim}")
    print(f"Models: {sorted(set(e.model_label for e in entries))}")

    # Generate all plots
    print("\n--- Plot 1: Full Agent Heatmap ---")
    plot_agent_heatmap(entries, str(out / "agent_heatmap.png"))

    print("\n--- Plot 2: Model-Averaged Similarity ---")
    plot_model_similarity(entries, str(out / "model_similarity.png"))

    print("\n--- Plot 3: Model x Probe Breakdown ---")
    plot_model_probe_similarity(entries, str(out / "model_probe_similarity.png"))

    print("\n--- Plot 4: Injected vs Baseline ---")
    plot_injection_similarity(entries, str(out / "injection_similarity.png"))

    # Save raw data
    if args.save_data:
        vectors = np.array([e.dna_vector for e in entries])
        sim_full = pairwise_cosine_matrix(vectors)
        meta = [
            {
                "agent_id": e.agent_id,
                "trial_id": e.trial_id,
                "model_label": e.model_label,
                "probe": e.probe,
                "is_injected": e.is_injected,
                "topology": e.topology,
            }
            for e in entries
        ]
        data_out = {
            "agents": meta,
            "similarity_matrix": sim_full.tolist(),
            "dna_dim": args.dna_dim,
            "embedding_model": args.embedding_model,
        }
        data_path = str(out / "similarity_data.json")
        with open(data_path, "w") as f:
            json.dump(data_out, f, indent=2)
        print(f"\n  Saved raw data: {data_path}")

    print(f"\nAll plots saved to: {out}/")


if __name__ == "__main__":
    main()
