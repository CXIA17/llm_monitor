#!/usr/bin/env python3
"""
Download Pretrained SAE Checkpoints
====================================

Downloads pretrained Sparse Autoencoder checkpoints from SAELens
for use with the SAE fingerprinting pipeline.

Supported models:
  - EleutherAI/pythia-1b       → pythia-70m-deduped SAEs (or pythia-1b if available)
  - EleutherAI/pythia-2.8b     → pythia-2.8b SAEs
  - google/gemma-3-1b-it       → gemma-2-2b SAEs (closest available)

The script downloads SAE weights and converts them to the format expected
by core/sae_fingerprint.py's SparseAutoencoder.from_pretrained().

Usage:
  # Install SAELens first
  pip install sae-lens

  # Download all available SAEs
  python download_sae_checkpoints.py --output-dir pretrained_saes

  # Download for specific models only
  python download_sae_checkpoints.py --models pythia_1b pythia_2.8b --output-dir pretrained_saes

  # List available SAE releases without downloading
  python download_sae_checkpoints.py --list-releases --filter pythia
"""

import argparse
import json
import os
import sys
from pathlib import Path


# ── SAE release registry ─────────────────────────────────────────────────────
# Maps our model labels to SAELens release IDs and recommended SAE configs.
# Each entry specifies:
#   release:   SAELens release name (from https://github.com/jbloomAudit/SAELens)
#   sae_id:    Specific SAE hook point within the release
#   layer_idx: Recommended layer index for activation extraction
#   d_model:   Reader model hidden dimension
#   notes:     Compatibility notes

SAE_REGISTRY = {
    "gemma_2b": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_12/width_16k/canonical",
        "reader_model": "google/gemma-2-2b",
        "layer_idx": 12,
        "d_model": 2304,
        "d_sae": 16384,
        "notes": "Gemma Scope 2B residual SAE (Google, canonical). Reader for gemma-3-1b transcripts.",
    },
    "gemma_2b_layer6": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_6/width_16k/canonical",
        "reader_model": "google/gemma-2-2b",
        "layer_idx": 6,
        "d_model": 2304,
        "d_sae": 16384,
        "notes": "Gemma Scope 2B residual SAE, early layer (layer 6).",
    },
    "gemma_2b_layer20": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_20/width_16k/canonical",
        "reader_model": "google/gemma-2-2b",
        "layer_idx": 20,
        "d_model": 2304,
        "d_sae": 16384,
        "notes": "Gemma Scope 2B residual SAE, late layer (layer 20).",
    },
    "pythia_70m": {
        "release": "pythia-70m-deduped-res-sm",
        "sae_id": "blocks.3.hook_resid_post",
        "reader_model": "EleutherAI/pythia-70m-deduped",
        "layer_idx": 3,
        "d_model": 512,
        "d_sae": 32768,
        "notes": "Pythia-70M residual SAE (layer 3). Only Pythia SAE available in SAELens.",
    },
}

# NOTE: No pretrained SAEs exist for Pythia-1B, Pythia-2.8B, Llama-3.2, or Qwen3 in SAELens.
# For those models, you need Option 2 (train your own SAE) or use a different reader model
# (e.g., gemma-2-2b as a universal reader).


def list_available_releases(filter_str: str = None):
    """List SAE releases from the registry."""
    print("\n Available SAE Checkpoints in Registry")
    print("=" * 70)
    for label, cfg in SAE_REGISTRY.items():
        if filter_str and filter_str.lower() not in label.lower():
            continue
        print(f"\n  Model label:   {label}")
        print(f"  SAELens release: {cfg['release']}")
        print(f"  SAE hook ID:     {cfg['sae_id']}")
        print(f"  Reader model:    {cfg['reader_model']}")
        print(f"  Layer index:     {cfg['layer_idx']}")
        print(f"  Dimensions:      d_model={cfg['d_model']}, d_sae={cfg['d_sae']}")
        print(f"  Notes:           {cfg['notes']}")
    print()


def download_sae(label: str, cfg: dict, output_dir: Path, force: bool = False):
    """Download a single SAE checkpoint via SAELens."""
    try:
        from sae_lens import SAE
    except ImportError:
        print("ERROR: sae-lens not installed. Run: pip install sae-lens")
        sys.exit(1)

    out_path = output_dir / label
    meta_path = out_path / "sae_meta.json"

    if meta_path.exists() and not force:
        print(f"  [{label}] Already downloaded at {out_path}, skipping (use --force to re-download)")
        return out_path

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"  [{label}] Downloading from SAELens: {cfg['release']} / {cfg['sae_id']} ...")

    try:
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=cfg["release"],
            sae_id=cfg["sae_id"],
        )
    except Exception as e:
        print(f"  [{label}] SAELens download failed: {e}")
        print(f"  [{label}] Trying alternative download method...")
        return try_alternative_download(label, cfg, output_dir)

    # Save in the format expected by SparseAutoencoder.from_pretrained()
    import torch

    save_dict = {
        "cfg": {
            "d_model": cfg["d_model"],
            "d_sae": cfg["d_sae"],
        },
        "state_dict": {},
    }

    # Extract weights from SAELens SAE object
    state = sae.state_dict()
    for key, tensor in state.items():
        save_dict["state_dict"][key] = tensor.cpu()

    # Also save raw SAELens format for reference
    torch_path = out_path / "sae_weights.pt"
    torch.save(save_dict, torch_path)
    print(f"  [{label}] Saved weights to {torch_path}")

    # Save metadata
    meta = {
        "label": label,
        "release": cfg["release"],
        "sae_id": cfg["sae_id"],
        "reader_model": cfg["reader_model"],
        "layer_idx": cfg["layer_idx"],
        "d_model": cfg["d_model"],
        "d_sae": cfg["d_sae"],
        "weight_file": "sae_weights.pt",
        "saelens_cfg": cfg_dict if isinstance(cfg_dict, dict) else {},
        "notes": cfg["notes"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [{label}] Saved metadata to {meta_path}")

    return out_path


def try_alternative_download(label: str, cfg: dict, output_dir: Path):
    """Fallback: download SAE weights directly from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(f"  [{label}] huggingface_hub not installed. Run: pip install huggingface-hub")
        return None

    out_path = output_dir / label
    out_path.mkdir(parents=True, exist_ok=True)

    release = cfg["release"]
    sae_id = cfg["sae_id"]

    print(f"  [{label}] Trying HuggingFace Hub: {release} ...")

    # Try common file patterns for SAE weights on HF
    possible_files = [
        f"{sae_id}/sae_weights.safetensors",
        f"{sae_id}/sae_weights.pt",
        f"{sae_id}.pt",
        "sae_weights.safetensors",
        "sae_weights.pt",
    ]

    for fname in possible_files:
        try:
            downloaded = hf_hub_download(
                repo_id=release,
                filename=fname,
                local_dir=out_path,
            )
            print(f"  [{label}] Downloaded: {downloaded}")

            # Also get config if available
            try:
                cfg_file = fname.rsplit("/", 1)[0] + "/cfg.json" if "/" in fname else "cfg.json"
                hf_hub_download(
                    repo_id=release,
                    filename=cfg_file,
                    local_dir=out_path,
                )
            except Exception:
                pass

            # Save metadata
            meta = {
                "label": label,
                "release": release,
                "sae_id": sae_id,
                "reader_model": cfg["reader_model"],
                "layer_idx": cfg["layer_idx"],
                "d_model": cfg["d_model"],
                "d_sae": cfg["d_sae"],
                "weight_file": fname,
                "notes": cfg["notes"],
                "download_method": "huggingface_hub",
            }
            with open(out_path / "sae_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            return out_path

        except Exception:
            continue

    print(f"  [{label}] Could not download weights. You may need to download manually from:")
    print(f"           https://huggingface.co/{release}")
    return None


def print_usage_instructions(output_dir: Path, downloaded: dict):
    """Print instructions for using downloaded SAEs with run_sae_experiments.py."""
    print("\n" + "=" * 70)
    print(" Download Complete")
    print("=" * 70)

    if not downloaded:
        print("\nNo SAEs were downloaded. Check errors above.")
        return

    print("\nTo run SAE experiments in live mode:\n")

    for label, path in downloaded.items():
        if path is None:
            continue
        cfg = SAE_REGISTRY[label]
        meta_path = path / "sae_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            weight_file = path / meta.get("weight_file", "sae_weights.pt")
        else:
            weight_file = path / "sae_weights.pt"

        print(f"  # {label} (reader: {cfg['reader_model']})")
        print(f"  python run_sae_experiments.py \\")
        print(f"      --results experiment_results/<your_results>.json \\")
        print(f"      --model-label {label} \\")
        print(f"      --reader-model {cfg['reader_model']} \\")
        print(f"      --sae-path {weight_file} \\")
        print(f"      --layer-idx {cfg['layer_idx']} \\")
        print(f"      --d-sae {cfg['d_sae']} \\")
        print(f"      --device cuda:0 \\")
        print(f"      --experiments e1 e3 e5")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained SAE checkpoints for SAE fingerprinting"
    )
    parser.add_argument(
        "--output-dir", "-o", default=os.path.join(os.environ.get("MODEL_DIR", "models"), "pretrained_saes"),
        help="Directory to save downloaded SAE checkpoints (default: $MODEL_DIR/pretrained_saes)"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(SAE_REGISTRY.keys()),
        default=None,
        help="Which models to download SAEs for (default: all)"
    )
    parser.add_argument(
        "--list-releases", action="store_true",
        help="List available SAE releases and exit"
    )
    parser.add_argument(
        "--filter", default=None,
        help="Filter releases by name (used with --list-releases)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if checkpoint already exists"
    )
    args = parser.parse_args()

    if args.list_releases:
        list_available_releases(args.filter)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or list(SAE_REGISTRY.keys())

    print(f"\n Downloading SAE checkpoints to: {output_dir}/")
    print(f" Models: {', '.join(models)}\n")

    downloaded = {}
    for label in models:
        cfg = SAE_REGISTRY[label]
        result = download_sae(label, cfg, output_dir, force=args.force)
        downloaded[label] = result

    print_usage_instructions(output_dir, downloaded)


if __name__ == "__main__":
    main()
