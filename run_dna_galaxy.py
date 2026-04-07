#!/usr/bin/env python3
"""
DNA Galaxy Visualizer
======================

Extracts LLM-DNA embeddings from agent conversation transcripts using
a sentence embedding model, then projects them into a 2D galaxy plot.

Single-model mode:   color = probe type, marker = agent role
Multi-model mode:    color = model,      marker = probe type
                     (pass --model-label to tag result files with model names)

Usage:
  # Single model
  python run_dna_galaxy.py \\
      --results experiment_results/h1_results_20260326_100042.json \\
                experiment_results/claude_judge/h3_results_20260325_081500.json \\
      --embedding-model sentence-transformers/all-mpnet-base-v2 \\
      --device cuda:5 -o experiment_results/dna_galaxy_pca.png

  # Multi-model (--model-label pairs each file with a model name)
  python run_dna_galaxy.py \\
      --results \\
          experiment_results/h1_results_20260326_100042.json \\
          experiment_results/claude_judge/h3_results_20260325_081500.json \\
          experiment_results/eleutherai_pythia_1b/h1_results_20260329_232014.json \\
          experiment_results/eleutherai_pythia_1b/h3_results_20260329_232014.json \\
          experiment_results/eleutherai_pythia_2_8b/h1_results_20260330_231405.json \\
          experiment_results/eleutherai_pythia_2_8b/h3_results_20260330_231405.json \\
          experiment_results/google_gemma_3_1b_it/h1_results_20260330_064849.json \\
          experiment_results/google_gemma_3_1b_it/h3_results_20260330_064849.json \\
      --model-label \\
          qwen3_4b qwen3_4b \\
          pythia_1b pythia_1b \\
          pythia_2.8b pythia_2.8b \\
          gemma_3_1b gemma_3_1b \\
      --embedding-model sentence-transformers/all-mpnet-base-v2 \\
      --device cuda:5 -o experiment_results/dna_galaxy_multimodel_pca.png --save-data
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════

@dataclass
class AgentDNAEntry:
    """One agent's DNA from one experiment trial."""
    agent_id: str
    trial_id: str
    experiment: str
    probe: str
    is_injected: bool
    injection_strength: float
    topology: str
    num_responses: int
    model_label: str = "unknown"
    dna_vector: Optional[np.ndarray] = None
    mean_score: float = 0.0
    xy: Optional[Tuple[float, float]] = None


# ═════════════════════════════════════════════════════════════════
# Step 1: Load per-agent responses
# ═════════════════════════════════════════════════════════════════

def load_agent_responses(
    result_files: List[str],
    model_labels: Optional[List[str]] = None,
) -> List[Tuple[AgentDNAEntry, List[str]]]:
    """
    Parse result JSON files and extract per-agent response texts.
    model_labels: one label per file (same length as result_files).
    """
    if model_labels and len(model_labels) != len(result_files):
        raise ValueError(
            f"--model-label count ({len(model_labels)}) must match "
            f"--results count ({len(result_files)})"
        )

    entries = []
    for idx, path in enumerate(result_files):
        label = model_labels[idx] if model_labels else "unknown"
        print(f"Loading [{label}] {path} ...")
        with open(path) as f:
            trials = json.load(f)

        for trial in trials:
            transcript = trial.get("transcript", [])
            if not transcript:
                continue

            experiment    = trial["experiment"]
            trial_id      = trial["trial_id"]
            probe         = trial["probe"]
            strength      = trial["injection_strength"]
            topology      = trial["topology"]
            injected_set  = set(trial.get("injected_agents", []))

            agent_responses: Dict[str, List[str]] = defaultdict(list)
            agent_scores:    Dict[str, List[float]] = defaultdict(list)
            for e in transcript:
                aid  = e["agent_id"]
                resp = e.get("response", e.get("text", ""))
                if resp:
                    agent_responses[aid].append(resp)
                    if "score" in e:
                        agent_scores[aid].append(e["score"])

            for aid, responses in agent_responses.items():
                mean_sc = float(np.mean(agent_scores[aid])) if agent_scores[aid] else 0.0
                entry = AgentDNAEntry(
                    agent_id=aid,
                    trial_id=trial_id,
                    experiment=experiment,
                    probe=probe,
                    is_injected=aid in injected_set,
                    injection_strength=strength,
                    topology=topology,
                    num_responses=len(responses),
                    model_label=label,
                    mean_score=mean_sc,
                )
                entries.append((entry, responses))

    print(f"Extracted {len(entries)} agent entries with transcripts")
    return entries


# ═════════════════════════════════════════════════════════════════
# Step 2: Compute DNA vectors
# ═════════════════════════════════════════════════════════════════

def compute_dna_vectors(
    entries: List[Tuple[AgentDNAEntry, List[str]]],
    embedding_model: str,
    device: str,
    dna_dim: int = 128,
    projection_seed: int = 42,
    batch_size: int = 4,
) -> List[AgentDNAEntry]:
    from core.llm_dna_extractor import LLMDNAExtractor

    extractor = LLMDNAExtractor(
        embedding_model=embedding_model,
        dna_dim=dna_dim,
        projection_seed=projection_seed,
        batch_size=batch_size,
    )

    min_responses = min(len(resps) for _, resps in entries)
    print(f"Using {min_responses} responses per agent (min across all entries)")

    results = []
    total = len(entries)
    for i, (entry, responses) in enumerate(entries):
        responses_trimmed = responses[:min_responses]
        label = f"{'*' if entry.is_injected else ' '}{entry.agent_id}"
        print(f"  [{i+1}/{total}] [{entry.model_label}] {entry.trial_id} / {label}")

        dna_result = extractor.extract_from_responses(
            responses=responses_trimmed,
            device=device,
            model_name=f"{entry.model_label}/{entry.trial_id}/{entry.agent_id}",
            verbose=False,
        )
        entry.dna_vector = dna_result.vector
        results.append(entry)

    return results


# ═════════════════════════════════════════════════════════════════
# Step 3: Dimensionality reduction → 2D
# ═════════════════════════════════════════════════════════════════

def reduce_to_2d(
    entries: List[AgentDNAEntry],
    method: str = "pca",
    perplexity: float = 15.0,
) -> List[AgentDNAEntry]:
    vectors = np.array([e.dna_vector for e in entries])

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords  = reducer.fit_transform(vectors)
        exp     = reducer.explained_variance_ratio_
        print(f"PCA explained variance: {exp[0]:.1%}, {exp[1]:.1%} (total {sum(exp):.1%})")
    elif method == "tsne":
        from sklearn.manifold import TSNE
        perp    = min(perplexity, len(entries) - 1)
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca")
        coords  = reducer.fit_transform(vectors)
        print(f"t-SNE with perplexity={perp:.0f}")
    else:
        raise ValueError(f"Unknown method: {method}")

    for entry, (x, y) in zip(entries, coords):
        entry.xy = (float(x), float(y))
    return entries


# ═════════════════════════════════════════════════════════════════
# Step 4: Plot
# ═════════════════════════════════════════════════════════════════

# Fixed palettes ─ extend as needed
MODEL_COLORS = {
    "qwen3_4b":   "#1565c0",   # blue
    "pythia_1b":  "#6a1b9a",   # purple
    "pythia_2.8b":"#ad1457",   # pink-red
    "gemma_3_1b": "#2e7d32",   # green
    # fallback colours for unexpected labels
    "_fallback":  ["#e65100", "#00695c", "#37474f", "#558b2f"],
}

PROBE_MARKERS = {
    "sentiment":  "o",
    "sycophancy": "s",
    "toxicity":   "D",
}

ROLE_MARKERS = {
    "proposer":           "o",
    "critic":             "s",
    "judge":              "D",
    "plaintiff_attorney": "^",
    "defense_attorney":   "v",
    "court_judge":        "P",
    "expert_1":           "h",
    "defender":           "X",
    "attacker":           "p",
}


def _resolve_model_color(model_label: str, seen_models: List[str]) -> str:
    if model_label in MODEL_COLORS:
        return MODEL_COLORS[model_label]
    fallbacks = MODEL_COLORS["_fallback"]
    idx = seen_models.index(model_label) if model_label in seen_models else 0
    return fallbacks[idx % len(fallbacks)]


def plot_galaxy(
    entries: List[AgentDNAEntry],
    output_path: str,
    method: str = "pca",
    title: str = "Agent DNA Galaxy",
    multi_model: bool = False,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.spatial import ConvexHull

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#ffffff")
    ax.set_facecolor("#f8f9fa")

    method_label  = "PCA" if method == "pca" else "t-SNE"
    model_labels  = sorted(set(e.model_label for e in entries))
    probe_labels  = sorted(set(e.probe for e in entries))

    # ── Plot points ──
    for entry in entries:
        x, y = entry.xy

        if multi_model:
            color  = _resolve_model_color(entry.model_label, model_labels)
            marker = PROBE_MARKERS.get(entry.probe, "o")
        else:
            color  = {
                "sentiment":  "#1565c0",
                "sycophancy": "#2e7d32",
                "toxicity":   "#c62828",
            }.get(entry.probe, "#888888")
            marker = ROLE_MARKERS.get(entry.agent_id, "o")

        size       = 180 if entry.is_injected else 80
        edge_color = "#000000" if entry.is_injected else color
        edge_width = 2.0      if entry.is_injected else 0.5
        alpha      = 0.95     if entry.is_injected else 0.55
        zorder     = 10       if entry.is_injected else 5

        ax.scatter(x, y, c=color, marker=marker, s=size,
                   edgecolors=edge_color, linewidths=edge_width,
                   alpha=alpha, zorder=zorder)

        if entry.is_injected:
            ax.scatter(x, y, c=color, marker=marker, s=size * 3,
                       alpha=0.12, zorder=zorder - 1)

    # ── Labels for injected agents ──
    labeled_positions = []
    for entry in entries:
        if not entry.is_injected:
            continue
        x, y = entry.xy
        if any(abs(x - px) < 0.3 and abs(y - py) < 0.3
               for px, py in labeled_positions):
            continue
        labeled_positions.append((x, y))
        if multi_model:
            lbl = f"{entry.model_label}\n{entry.probe} s={entry.injection_strength}"
        else:
            lbl = f"{entry.agent_id}\n{entry.probe} s={entry.injection_strength}"
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(12, 12),
                    fontsize=7, color="#222222", alpha=0.85,
                    arrowprops=dict(arrowstyle="-", color="#00000040", lw=0.5))

    # ── Convex hulls ──
    if multi_model:
        # Hull per model
        for mlabel in model_labels:
            color = _resolve_model_color(mlabel, model_labels)
            pts   = np.array([e.xy for e in entries if e.model_label == mlabel])
            if len(pts) >= 3:
                try:
                    hull     = ConvexHull(pts)
                    hull_pts = np.append(hull.vertices, hull.vertices[0])
                    ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                            color=color, alpha=0.30, linewidth=1.2, linestyle="--")
                    ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                            color=color, alpha=0.05)
                except Exception:
                    pass
    else:
        # Hull per probe
        probe_colors = {
            "sentiment":  "#1565c0",
            "sycophancy": "#2e7d32",
            "toxicity":   "#c62828",
        }
        for probe, color in probe_colors.items():
            pts = np.array([e.xy for e in entries if e.probe == probe])
            if len(pts) >= 3:
                try:
                    hull     = ConvexHull(pts)
                    hull_pts = np.append(hull.vertices, hull.vertices[0])
                    ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                            color=color, alpha=0.35, linewidth=1.0, linestyle="--")
                    ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                            color=color, alpha=0.06)
                except Exception:
                    pass

    # ── Legend ──
    legend_elements = []

    if multi_model:
        legend_elements.append(
            Line2D([0], [0], color="none", label="— Model (color) —"))
        for mlabel in model_labels:
            color = _resolve_model_color(mlabel, model_labels)
            legend_elements.append(
                Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=color, markersize=10, label=mlabel))

        legend_elements.append(Line2D([0], [0], color="none", label=""))
        legend_elements.append(
            Line2D([0], [0], color="none", label="— Probe (marker) —"))
        for probe, marker in PROBE_MARKERS.items():
            if any(e.probe == probe for e in entries):
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color="none",
                           markerfacecolor="#777777", markersize=9, label=probe))
    else:
        probe_colors = {
            "sentiment":  "#1565c0",
            "sycophancy": "#2e7d32",
            "toxicity":   "#c62828",
        }
        legend_elements.append(
            Line2D([0], [0], color="none", label="— Probe (color) —"))
        for probe, color in probe_colors.items():
            legend_elements.append(
                Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=color, markersize=10, label=probe))

        legend_elements.append(Line2D([0], [0], color="none", label=""))
        legend_elements.append(
            Line2D([0], [0], color="none", label="— Role (marker) —"))
        for role, marker in ROLE_MARKERS.items():
            if any(e.agent_id == role for e in entries):
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color="none",
                           markerfacecolor="#777777", markersize=8, label=role))

    legend_elements.append(Line2D([0], [0], color="none", label=""))
    legend_elements.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777",
               markersize=12, markeredgecolor="#000000", markeredgewidth=2,
               label="Injected (black border)"))
    legend_elements.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777",
               markersize=8, alpha=0.6, label="Baseline"))

    legend = ax.legend(handles=legend_elements, loc="upper left",
                       fontsize=8, facecolor="#ffffffd0", edgecolor="#cccccc",
                       labelcolor="#333333", framealpha=0.9)
    legend.get_frame().set_linewidth(0.5)

    # ── Axis styling ──
    ax.set_xlabel(f"{method_label} Dimension 1", color="#444444", fontsize=10)
    ax.set_ylabel(f"{method_label} Dimension 2", color="#444444", fontsize=10)
    ax.set_title(title, color="#222222", fontsize=14, pad=15)
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.grid(True, alpha=0.3, color="#cccccc")

    # ── Stats box ──
    n_inj   = sum(1 for e in entries if e.is_injected)
    n_bas   = len(entries) - n_inj
    n_trials = len(set(e.trial_id for e in entries))
    models_str = ", ".join(model_labels) if multi_model else "single model"
    stats_text = (
        f"Agents: {len(entries)} ({n_inj} injected, {n_bas} baseline)\n"
        f"Models: {models_str}\n"
        f"Trials: {n_trials} | DNA dim: {entries[0].dna_vector.shape[0]} → 2D via {method_label}"
    )
    ax.text(0.99, 0.02, stats_text, transform=ax.transAxes,
            fontsize=7, color="#666666", ha="right", va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#ffffff", edgecolor="#cccccc"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor="#ffffff", bbox_inches="tight")
    print(f"\nGalaxy saved to {output_path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════
# Step 5: Save DNA data as JSON
# ═════════════════════════════════════════════════════════════════

def save_dna_data(entries: List[AgentDNAEntry], output_path: str):
    data = []
    for e in entries:
        data.append({
            "agent_id":          e.agent_id,
            "trial_id":          e.trial_id,
            "experiment":        e.experiment,
            "probe":             e.probe,
            "is_injected":       e.is_injected,
            "injection_strength":e.injection_strength,
            "topology":          e.topology,
            "num_responses":     e.num_responses,
            "model_label":       e.model_label,
            "mean_score":        e.mean_score,
            "dna_vector":        e.dna_vector.tolist(),
            "xy":                list(e.xy) if e.xy else None,
        })
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"DNA data saved to {output_path}")


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate DNA Galaxy from agent experiment transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", "-r", nargs="+", required=True,
                        help="Result JSON files (h1/h3, in order matching --model-label)")
    parser.add_argument("--model-label", nargs="+", default=None,
                        help="Model label per result file (enables multi-model mode)")
    parser.add_argument("--embedding-model", "-e",
                        default="sentence-transformers/all-mpnet-base-v2",
                        help="Embedding model path or HF name")
    parser.add_argument("--device", "-d", default="cuda:0")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--method", "-m", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--dna-dim", type=int, default=128)
    parser.add_argument("--perplexity", type=float, default=15.0)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    multi_model = args.model_label is not None
    suffix = "multimodel" if multi_model else "single"
    if args.output is None:
        args.output = f"experiment_results/dna_galaxy_{suffix}_{args.method}.png"

    title = args.title or (
        "Agent DNA Galaxy — Multi-Model" if multi_model else "Agent DNA Galaxy"
    )

    # Step 1
    entries = load_agent_responses(args.results, args.model_label)
    if not entries:
        print("ERROR: No transcripts found. H2 results have empty transcripts — use H1 or H3.")
        sys.exit(1)

    # Step 2
    print(f"\nComputing DNA embeddings ({args.embedding_model}) on {args.device} ...")
    agent_entries = compute_dna_vectors(
        entries, embedding_model=args.embedding_model,
        device=args.device, dna_dim=args.dna_dim,
    )

    # Step 3
    print(f"\nReducing to 2D via {args.method} ...")
    agent_entries = reduce_to_2d(agent_entries, method=args.method,
                                  perplexity=args.perplexity)

    # Step 4
    plot_galaxy(agent_entries, args.output, method=args.method,
                title=title, multi_model=multi_model)

    # Step 5
    if args.save_data:
        data_path = str(Path(args.output).with_suffix(".json"))
        save_dna_data(agent_entries, data_path)

    # Distance summary
    print(f"\n{'='*60}")
    print("  DNA Distance Summary (cosine)")
    print(f"{'='*60}")
    from scipy.spatial.distance import cosine as cos_dist

    injected = [e for e in agent_entries if e.is_injected]
    baseline = [e for e in agent_entries if not e.is_injected]
    if injected and baseline:
        ii = [cos_dist(injected[i].dna_vector, injected[j].dna_vector)
              for i in range(len(injected)) for j in range(i+1, len(injected))]
        bb = [cos_dist(baseline[i].dna_vector, baseline[j].dna_vector)
              for i in range(len(baseline)) for j in range(i+1, len(baseline))]
        xd = [cos_dist(ie.dna_vector, be.dna_vector)
              for ie in injected for be in baseline]
        print(f"  Injected ↔ Injected:  {np.mean(ii):.4f} ± {np.std(ii):.4f}")
        print(f"  Baseline ↔ Baseline:  {np.mean(bb):.4f} ± {np.std(bb):.4f}")
        print(f"  Injected ↔ Baseline:  {np.mean(xd):.4f} ± {np.std(xd):.4f}")

    if multi_model:
        print(f"\n  Per-model inter-model distances:")
        model_labels = sorted(set(e.model_label for e in agent_entries))
        for i in range(len(model_labels)):
            for j in range(i+1, len(model_labels)):
                m1 = [e for e in agent_entries if e.model_label == model_labels[i]]
                m2 = [e for e in agent_entries if e.model_label == model_labels[j]]
                d  = [cos_dist(a.dna_vector, b.dna_vector) for a in m1 for b in m2]
                print(f"    {model_labels[i]:12s} ↔ {model_labels[j]:12s}: {np.mean(d):.4f}")

    print(f"\nDone. Galaxy plot: {args.output}")


if __name__ == "__main__":
    main()
