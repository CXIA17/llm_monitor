#!/usr/bin/env python3
"""
SAE Validation Script
=====================

Validates trained SAE checkpoints by running live-mode SAE experiments
on existing experiment transcripts. Each model uses its own trained SAE
as the reader, processing one model at a time to fit in GPU memory.

Runs E1, E3, E5 for each available trained SAE and compares results
against the sim-mode baseline.

Usage:
  python run_sae_validation.py --device cuda:0

  # Specific models only
  python run_sae_validation.py --models pythia_1b gemma_3_1b --device cuda:0

  # Custom output directory
  python run_sae_validation.py -o experiment_results/sae_live --device cuda:0
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Registry: maps model label → SAE meta + experiment results ────────────────

TRAINED_SAE_DIR = Path(os.environ.get("MODEL_DIR", "models")) / "trained_saes"

EXPERIMENT_RESULTS = {
    "pythia_1b": "experiment_results/eleutherai_pythia_1b/all_results_20260329_232014.json",
    "pythia_2.8b": "experiment_results/eleutherai_pythia_2_8b/all_results_20260330_231405.json",
    "gemma_3_1b": "experiment_results/google_gemma_3_1b_it/all_results_20260330_064849.json",
}


def discover_trained_saes() -> dict:
    """Find all trained SAEs with valid metadata."""
    available = {}
    for model_dir in TRAINED_SAE_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        meta_path = model_dir / "sae_meta.json"
        weights_path = model_dir / "sae_weights.pt"
        if meta_path.exists() and weights_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            label = meta.get("label", model_dir.name)
            if label in EXPERIMENT_RESULTS:
                available[label] = {
                    "meta": meta,
                    "weights_path": str(weights_path),
                    "results_path": EXPERIMENT_RESULTS[label],
                }
    return available


def run_single_validation(
    label: str,
    info: dict,
    output_dir: Path,
    device: str,
    experiments: list[str],
) -> dict:
    """Run SAE experiments for a single model using its trained SAE."""
    meta = info["meta"]
    model_output = output_dir / label
    model_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Validating: {label}")
    print(f"  Reader model:  {meta['model_name']}")
    print(f"  SAE path:      {info['weights_path']}")
    print(f"  Layer:         {meta['layer_idx']}")
    print(f"  d_sae:         {meta['d_sae']}")
    print(f"  Results file:  {info['results_path']}")
    print(f"  Output:        {model_output}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "run_sae_experiments.py",
        "--results", info["results_path"],
        "--model-label", label,
        "--reader-model", meta["model_name"],
        "--sae-path", info["weights_path"],
        "--layer-idx", str(meta["layer_idx"]),
        "--d-sae", str(meta["d_sae"]),
        "--device", device,
        "--experiments", *experiments,
        "-o", str(model_output),
    ]

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    # Check outputs
    expected_files = []
    for exp in experiments:
        expected_files.append(f"{exp}_results.json")
    success = result.returncode == 0 and all(
        (model_output / f).exists() for f in expected_files
    )

    return {
        "status": "success" if success else "failed",
        "returncode": result.returncode,
        "time_minutes": round(elapsed / 60, 1),
        "output_dir": str(model_output),
    }


def compare_with_sim(live_dir: Path, sim_dir: Path, label: str, experiments: list[str]):
    """Compare live results with sim-mode baseline."""
    print(f"\n--- Comparison: {label} (live vs sim) ---")

    for exp in experiments:
        live_path = live_dir / label / f"{exp}_results.json"
        sim_path = sim_dir / f"{exp}_results.json"

        if not live_path.exists():
            print(f"  [{exp}] Live results missing, skipping")
            continue
        if not sim_path.exists():
            print(f"  [{exp}] Sim results missing, skipping comparison")
            continue

        with open(live_path) as f:
            live = json.load(f)
        with open(sim_path) as f:
            sim = json.load(f)

        # Find the relevant keys for this model
        live_keys = [k for k in live if label in k or k == label]
        sim_keys = [k for k in sim if label in k or k == label]

        if not live_keys:
            # For E1, the key is just the model label
            live_keys = list(live.keys())[:1]
        if not sim_keys:
            sim_keys = list(sim.keys())[:1]

        for lk in live_keys:
            live_data = live[lk]
            print(f"\n  [{exp}] {lk}:")
            if isinstance(live_data, dict) and "n_significant" in live_data:
                print(f"    Live:  n_significant={live_data['n_significant']}, "
                      f"mean_diff={live_data['mean_abs_diff']:.4f}")
                # Find matching sim key
                for sk in sim_keys:
                    if sk in sim and isinstance(sim[sk], dict) and "n_significant" in sim[sk]:
                        print(f"    Sim:   n_significant={sim[sk]['n_significant']}, "
                              f"mean_diff={sim[sk]['mean_abs_diff']:.4f}")
                        break
            elif isinstance(live_data, dict):
                # E5 format: nested by strength
                for strength, sdata in live_data.items():
                    if isinstance(sdata, dict) and "n_significant" in sdata:
                        print(f"    Live [s={strength}]: n_sig={sdata['n_significant']}, "
                              f"mean_diff={sdata['mean_abs_diff']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Validate trained SAEs with live experiments")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Which models to validate (default: all available)")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--experiments", nargs="+", default=["e1", "e3", "e5"],
                        choices=["e1", "e3", "e5"],
                        help="Which experiments to run")
    parser.add_argument("-o", "--output-dir", default="experiment_results/sae_live",
                        help="Output directory for live results")
    parser.add_argument("--sim-dir", default="experiment_results/sae_experiments",
                        help="Sim-mode results directory for comparison")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip comparison with sim-mode results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = Path(args.sim_dir)

    # Discover trained SAEs
    available = discover_trained_saes()
    if not available:
        print("No trained SAEs found with matching experiment results.")
        print(f"Checked: {TRAINED_SAE_DIR}")
        sys.exit(1)

    models = args.models or list(available.keys())
    models = [m for m in models if m in available]

    if not models:
        print(f"None of the requested models have trained SAEs.")
        print(f"Available: {list(available.keys())}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" SAE Validation Pipeline")
    print(f"{'='*60}")
    print(f" Models:      {', '.join(models)}")
    print(f" Experiments: {args.experiments}")
    print(f" Device:      {args.device}")
    print(f" Output:      {output_dir}")
    print(f"{'='*60}")

    # Run validation for each model (subprocess for GPU isolation)
    results = {}
    total_start = time.time()

    for i, label in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] Validating {label}...")
        results[label] = run_single_validation(
            label=label,
            info=available[label],
            output_dir=output_dir,
            device=args.device,
            experiments=args.experiments,
        )

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f" Validation Complete — {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    for label, res in results.items():
        status = "OK" if res["status"] == "success" else "FAILED"
        print(f"  [{status}] {label}: {res['time_minutes']}min — {res['output_dir']}")

    # Compare with sim-mode baseline
    if not args.skip_comparison and sim_dir.exists():
        print(f"\n{'='*60}")
        print(f" Live vs Sim-Mode Comparison")
        print(f"{'='*60}")
        for label in models:
            if results[label]["status"] == "success":
                compare_with_sim(output_dir, sim_dir, label, args.experiments)

    # Jaccard k-sweep (runs post-hoc on saved diff vectors, no GPU needed)
    if "e3" in args.experiments:
        successful = [m for m in models if results[m]["status"] == "success"]
        if len(successful) >= 2:
            run_jaccard_k_sweep(output_dir, successful)
        else:
            print("\n  [k-sweep] Fewer than 2 successful models, skipping.")

    # Save summary
    summary = {
        "models": results,
        "config": {
            "experiments": args.experiments,
            "device": args.device,
            "total_time_minutes": round(total_elapsed / 60, 1),
        },
    }
    summary_path = output_dir / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


PROBES = ["sentiment", "sycophancy", "toxicity"]

PROBE_COLORS = {
    "sentiment":  "#e65100",
    "sycophancy": "#1565c0",
    "toxicity":   "#b71c1c",
}


def run_jaccard_k_sweep(live_dir: Path, models: list[str]) -> None:
    """
    Post-hoc Jaccard k-sweep across all live-validated models.

    Loads per-model e3_diff_vectors.npz files (saved by run_e3), then
    computes Jaccard between probe-type top-k feature sets at a range
    of k values.  Reports two variants:

      (a) Per-model:  For each model, Jaccard of the top-k positively-
          shifted latents between each probe pair, averaged across models.
      (b) Cross-model shared:  Union of top-k per model, filtered to
          latents present in >= 2 models, then Jaccard across probe pairs.
          This matches the operationalisation used in the original E3
          transfer matrix.
    """
    probes = PROBES

    # ── Load diff vectors from all models ──────────────────────────────
    # ranked[(model, probe)] = indices sorted descending by signed diff
    ranked: dict[tuple[str, str], np.ndarray] = {}
    d_sae = None

    for model in models:
        npz_path = live_dir / model / "e3_diff_vectors.npz"
        if not npz_path.exists():
            print(f"  [k-sweep] Missing {npz_path}, skipping {model}")
            continue
        data = np.load(npz_path)
        for key in data.files:
            # Keys are "model__probe"
            m, p = key.split("__", 1)
            vec = data[key]
            if d_sae is None:
                d_sae = len(vec)
            ranked[(m, p)] = np.argsort(vec)[::-1]  # descending by signed diff
        data.close()

    found_models = sorted({m for m, _ in ranked})
    if len(found_models) < 2:
        print(f"  [k-sweep] Need >= 2 models, found {len(found_models)}. Skipping.")
        return

    print(f"\n{'='*60}")
    print(f" E3 Jaccard k-sweep  (d_sae={d_sae}, models={found_models})")
    print(f"{'='*60}")

    k_values = [k for k in [20, 50, 100, 200, 500, 1000, 2000, 5000] if k <= d_sae]
    probe_pairs = list(itertools.combinations(probes, 2))
    pair_keys = [f"{p1}_vs_{p2}" for p1, p2 in probe_pairs]

    per_model_mean: dict[str, list[float]] = {pk: [] for pk in pair_keys}
    cross_model_shared: dict[str, list[float]] = {pk: [] for pk in pair_keys}

    for k in k_values:
        # (a) Per-model Jaccard, averaged across models
        for (p1, p2), pk in zip(probe_pairs, pair_keys):
            jvals = []
            for model in found_models:
                a = ranked.get((model, p1))
                b = ranked.get((model, p2))
                if a is None or b is None:
                    continue
                s1 = set(a[:k].tolist())
                s2 = set(b[:k].tolist())
                union = s1 | s2
                j = len(s1 & s2) / len(union) if union else 0.0
                jvals.append(j)
            per_model_mean[pk].append(float(np.mean(jvals)) if jvals else 0.0)

        # (b) Cross-model shared (>= 2 models)
        probe_shared: dict[str, set] = {}
        for probe in probes:
            counts: dict[int, int] = defaultdict(int)
            for model in found_models:
                order = ranked.get((model, probe))
                if order is None:
                    continue
                for idx in order[:k].tolist():
                    counts[idx] += 1
            probe_shared[probe] = {idx for idx, c in counts.items() if c >= 2}

        for (p1, p2), pk in zip(probe_pairs, pair_keys):
            s1 = probe_shared.get(p1, set())
            s2 = probe_shared.get(p2, set())
            union = s1 | s2
            j = len(s1 & s2) / len(union) if union else 0.0
            cross_model_shared[pk].append(j)

    # ── Print table ────────────────────────────────────────────────────
    header = f"  {'k':>6}"
    for pk in pair_keys:
        short = pk.replace("_vs_", "/").replace("sentiment", "sen").replace("sycophancy", "syc").replace("toxicity", "tox")
        header += f"  {short:>10}(pm) {short:>10}(cm)"
    print(header)
    for i, k in enumerate(k_values):
        row = f"  {k:>6}"
        for pk in pair_keys:
            row += f"  {per_model_mean[pk][i]:>10.4f}    {cross_model_shared[pk][i]:>10.4f}  "
        print(row)

    # ── Save JSON ──────────────────────────────────────────────────────
    sweep_results = {
        "d_sae": int(d_sae),
        "models": found_models,
        "k_values": k_values,
        "per_model_mean_jaccard": per_model_mean,
        "cross_model_shared_jaccard": cross_model_shared,
    }
    out_json = live_dir / "e3_jaccard_k_sweep.json"
    with open(out_json, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n  Saved: {out_json}")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#ffffff")
    for ax in (ax1, ax2):
        ax.set_facecolor("#f8f9fa")

    colors = ["#e65100", "#1565c0", "#2e7d32"]
    for idx, pk in enumerate(pair_keys):
        label = pk.replace("_", " ")
        ax1.plot(k_values, per_model_mean[pk], marker="o", color=colors[idx],
                 label=label, linewidth=2, markersize=5)
        ax2.plot(k_values, cross_model_shared[pk], marker="s", color=colors[idx],
                 label=label, linewidth=2, markersize=5)

    for ax, title in [
        (ax1, "(a) Per-model Jaccard (mean across models)"),
        (ax2, "(b) Cross-model shared (≥2 models, current E3 definition)"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel("top-k latents", fontsize=11)
        ax.set_ylabel("Jaccard similarity", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.02, max(0.15, ax.get_ylim()[1] * 1.1))

    fig.suptitle(
        f"E3: Probe-type Jaccard vs top-k  (d_sae={d_sae}, {len(found_models)} models)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out_png = live_dir / "e3_jaccard_k_sweep.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"  Saved: {out_png}")


if __name__ == "__main__":
    main()
