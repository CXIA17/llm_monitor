#!/usr/bin/env python3
"""
Train Sparse Autoencoders for SAE Fingerprinting
=================================================

Trains SAEs for each target model using SAELens. Models are loaded one
at a time and offloaded after training to prevent CUDA OOM.

Supported models:
  - EleutherAI/pythia-1b          (16 layers, d_model=2048)
  - EleutherAI/pythia-2.8b        (32 layers, d_model=2560)
  - google/gemma-3-1b-it          (26 layers, d_model=1152)
  - Qwen/Qwen3-4B                (36 layers, d_model=2560)
  - meta-llama/Llama-3.2-3B-Instruct (28 layers, d_model=3072)

Usage:
  # Train SAEs for all models (sequential, one GPU)
  python train_sae.py --device cuda:0

  # Train for specific models only
  python train_sae.py --models pythia_1b pythia_2.8b --device cuda:0

  # Custom training tokens and dictionary size
  python train_sae.py --models gemma_3_1b --training-tokens 50000000 --d-sae 32768

  # Resume from checkpoint
  python train_sae.py --models pythia_1b --resume
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────

# Base directory for local model weights. Override with MODEL_DIR env var.
_MODEL_BASE = os.environ.get("MODEL_DIR", "models")

MODEL_REGISTRY = {
    "pythia_1b": {
        "model_name": "EleutherAI/pythia-1b",
        "local_path": os.path.join(_MODEL_BASE, "EleutherAI/pythia-1b"),
        "hook_name": "blocks.{layer}.hook_resid_post",
        "layer_idx": 8,          # middle of 16 layers
        "d_model": 2048,
        "n_layers": 16,
    },
    "pythia_2.8b": {
        "model_name": "EleutherAI/pythia-2.8b",
        "local_path": os.path.join(_MODEL_BASE, "EleutherAI/pythia-2.8b"),
        "hook_name": "blocks.{layer}.hook_resid_post",
        "layer_idx": 16,         # middle of 32 layers
        "d_model": 2560,
        "n_layers": 32,
    },
    "gemma_3_1b": {
        "model_name": "google/gemma-3-1b-it",
        "local_path": os.path.join(_MODEL_BASE, "google/gemma-3-1b-it"),
        "hook_name": "blocks.{layer}.hook_resid_post",
        "layer_idx": 13,         # middle of 26 layers
        "d_model": 1152,
        "n_layers": 26,
    },
    "qwen3_4b": {
        "model_name": "Qwen/Qwen3-4B",
        "local_path": os.path.join(_MODEL_BASE, "Qwen/Qwen3-4B"),
        "hook_name": "blocks.{layer}.hook_resid_post",
        "layer_idx": 18,         # middle of 36 layers
        "d_model": 2560,
        "n_layers": 36,
    },
    "llama_3.2_3b": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "local_path": os.path.join(_MODEL_BASE, "meta-llama/Llama-3.2-3B-Instruct"),
        "hook_name": "blocks.{layer}.hook_resid_post",
        "layer_idx": 14,         # middle of 28 layers
        "d_model": 3072,
        "n_layers": 28,
    },
}


def clear_gpu_memory():
    """Aggressively free GPU memory between model trainings."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("GPU memory cleared")


def get_gpu_memory_info(device: str) -> str:
    """Return a human-readable GPU memory status."""
    import torch

    if not torch.cuda.is_available():
        return "CUDA not available"
    idx = int(device.split(":")[-1]) if ":" in device else 0
    total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    used = torch.cuda.memory_allocated(idx) / 1024**3
    free = total - used
    return f"GPU {idx}: {used:.1f}GB used / {total:.1f}GB total ({free:.1f}GB free)"


def train_single_model(
    label: str,
    model_cfg: dict,
    output_dir: Path,
    device: str,
    d_sae: int,
    training_tokens: int,
    context_size: int,
    batch_size: int,
    n_checkpoints: int,
    dataset_path: str,
    resume: bool,
    layer_override: int = None,
):
    """Train an SAE for a single model, then offload everything."""
    from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner, StandardTrainingSAEConfig
    import torch

    layer = layer_override if layer_override is not None else model_cfg["layer_idx"]
    hook_name = model_cfg["hook_name"].format(layer=layer)
    d_model = model_cfg["d_model"]
    local_path = model_cfg["local_path"]
    model_name = model_cfg["model_name"]

    model_output = output_dir / label
    checkpoint_dir = model_output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Training SAE for: {label}")
    logger.info(f"  Model:     {model_name} (local: {local_path})")
    logger.info(f"  Hook:      {hook_name}")
    logger.info(f"  d_model:   {d_model}")
    logger.info(f"  d_sae:     {d_sae}")
    logger.info(f"  Tokens:    {training_tokens:,}")
    logger.info(f"  Device:    {device}")
    logger.info(f"  Output:    {model_output}")
    logger.info(f"  {get_gpu_memory_info(device)}")
    logger.info(f"{'='*60}")

    # Check for existing checkpoint to resume from
    resume_path = None
    if resume:
        existing = sorted(checkpoint_dir.glob("*.pt"))
        if existing:
            resume_path = str(existing[-1])
            logger.info(f"  Resuming from checkpoint: {resume_path}")

    # Buffer constraint: n_batches_in_buffer * context_size >= train_batch_size_tokens
    # With context_size=256 and batch_size=4096: need n_batches_in_buffer >= 16
    # Use 20 universally (20 * 256 = 5120 > 4096)
    store_batch_size = 32
    n_batches_in_buffer = 20

    # Build SAE config (SAE trains in fp32 for stability)
    sae_cfg = StandardTrainingSAEConfig(
        d_in=d_model,
        d_sae=d_sae,
        dtype="float32",
        device=device,
    )

    # Build runner config
    # TransformerLens requires HF model name for architecture lookup.
    runner_cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=model_name,
        # Force TransformerLens to load model weights in fp16 (halves model VRAM)
        model_from_pretrained_kwargs={"dtype": "float16"},
        hook_name=hook_name,
        dataset_path=dataset_path,
        dataset_trust_remote_code=False,
        streaming=True,
        is_dataset_tokenized=False,
        context_size=context_size,
        training_tokens=training_tokens,
        train_batch_size_tokens=batch_size,
        store_batch_size_prompts=store_batch_size,
        n_batches_in_buffer=n_batches_in_buffer,
        device=device,
        seed=42,
        dtype="float32",
        autocast_lm=True,  # fp16 forward pass
        # Checkpointing
        n_checkpoints=n_checkpoints,
        checkpoint_path=str(checkpoint_dir),
        save_final_checkpoint=True,
        output_path=str(model_output),
        # Resume
        resume_from_checkpoint=resume_path,
        # Logging
        verbose=True,
    )

    # Train
    t0 = time.time()
    try:
        runner = SAETrainingRunner(runner_cfg)
        sae = runner.run()
        elapsed = time.time() - t0
        logger.info(f"  Training complete in {elapsed/60:.1f} minutes")
    except Exception as e:
        logger.error(f"  Training FAILED for {label}: {e}")
        raise
    finally:
        # ── Offload: delete runner, model, and free GPU ───────────────
        logger.info(f"  Offloading {label} from GPU...")
        try:
            # Delete the runner which holds the model
            del runner
        except NameError:
            pass
        try:
            del sae
        except NameError:
            pass
        clear_gpu_memory()
        logger.info(f"  {get_gpu_memory_info(device)}")

    # Save the final weights in our standard format for sae_fingerprint.py
    final_pt = model_output / "sae_weights.pt"
    # Find the latest checkpoint or final output
    final_candidates = sorted(model_output.glob("**/*.pt"), key=lambda p: p.stat().st_mtime)
    if final_candidates:
        latest = final_candidates[-1]
        if latest != final_pt:
            # Load and re-save in our standard format
            state = torch.load(latest, map_location="cpu", weights_only=True)
            save_dict = {
                "cfg": {"d_model": d_model, "d_sae": d_sae},
                "state_dict": state.get("state_dict", state),
            }
            torch.save(save_dict, final_pt)
            logger.info(f"  Saved standardized weights to {final_pt}")

    # Save metadata
    meta = {
        "label": label,
        "model_name": model_name,
        "local_path": local_path,
        "hook_name": hook_name,
        "layer_idx": layer,
        "d_model": d_model,
        "d_sae": d_sae,
        "training_tokens": training_tokens,
        "dataset": dataset_path,
        "context_size": context_size,
        "device": device,
        "training_time_minutes": round(elapsed / 60, 1) if 'elapsed' in dir() else None,
    }
    with open(model_output / "sae_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Metadata saved to {model_output / 'sae_meta.json'}")

    return model_output


def main():
    parser = argparse.ArgumentParser(
        description="Train SAEs for SAE fingerprinting experiments"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help="Which models to train SAEs for (default: all)",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="GPU device (default: cuda:0)",
    )
    parser.add_argument(
        "--d-sae", type=int, default=16384,
        help="SAE dictionary size (default: 16384)",
    )
    parser.add_argument(
        "--training-tokens", type=int, default=30_000_000,
        help="Number of training tokens (default: 30M)",
    )
    parser.add_argument(
        "--context-size", type=int, default=256,
        help="Context window for activation collection (default: 256)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096,
        help="Training batch size in tokens (default: 4096)",
    )
    parser.add_argument(
        "--n-checkpoints", type=int, default=5,
        help="Number of intermediate checkpoints (default: 5)",
    )
    parser.add_argument(
        "--dataset", default="HuggingFaceFW/fineweb",
        help="HuggingFace dataset for training (default: HuggingFaceFW/fineweb)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=os.path.join(_MODEL_BASE, "trained_saes"),
        help="Output directory for trained SAEs",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint if available",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Override layer index for all models",
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Run directly (used by subprocess dispatch, do not set manually)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or list(MODEL_REGISTRY.keys())

    # Single mode: train one model directly and exit (called by subprocess)
    if args.single:
        assert len(models) == 1, "--single requires exactly one model"
        label = models[0]
        train_single_model(
            label=label,
            model_cfg=MODEL_REGISTRY[label],
            output_dir=output_dir,
            device=args.device,
            d_sae=args.d_sae,
            training_tokens=args.training_tokens,
            context_size=args.context_size,
            batch_size=args.batch_size,
            n_checkpoints=args.n_checkpoints,
            dataset_path=args.dataset,
            resume=args.resume,
            layer_override=args.layer,
        )
        return

    print(f"\n{'='*60}")
    print(f" SAE Training Pipeline")
    print(f"{'='*60}")
    print(f" Models:    {', '.join(models)}")
    print(f" d_sae:     {args.d_sae:,}")
    print(f" Tokens:    {args.training_tokens:,}")
    print(f" Device:    {args.device}")
    print(f" Output:    {args.output_dir}")
    print(f" Dataset:   {args.dataset}")
    print(f"{'='*60}\n")

    results = {}
    total_start = time.time()

    for i, label in enumerate(models):
        model_cfg = MODEL_REGISTRY[label]
        print(f"\n[{i+1}/{len(models)}] Training {label}...")

        # Run each model in a subprocess to guarantee clean GPU memory.
        # Without this, TransformerLens/PyTorch may hold GPU references
        # that survive del/gc, causing OOM on subsequent models.
        import subprocess
        cmd = [
            sys.executable, __file__,
            "--models", label,
            "--device", args.device,
            "--d-sae", str(args.d_sae),
            "--training-tokens", str(args.training_tokens),
            "--context-size", str(args.context_size),
            "--batch-size", str(args.batch_size),
            "--n-checkpoints", str(args.n_checkpoints),
            "--dataset", args.dataset,
            "--output-dir", str(output_dir),
        ]
        if args.resume:
            cmd.append("--resume")
        if args.layer is not None:
            cmd.extend(["--layer", str(args.layer)])
        cmd.append("--single")  # flag to run directly without subprocess nesting

        result = subprocess.run(cmd, capture_output=False)
        meta_path = output_dir / label / "sae_meta.json"
        if result.returncode == 0 and meta_path.exists():
            results[label] = {"status": "success", "path": str(output_dir / label)}
        else:
            results[label] = {"status": "failed", "error": f"Subprocess exited with code {result.returncode}"}

    total_elapsed = time.time() - total_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Training Complete — {total_elapsed/60:.1f} minutes total")
    print(f"{'='*60}")
    for label, res in results.items():
        status = "OK" if res["status"] == "success" else "FAILED"
        print(f"  [{status}] {label}: {res.get('path', res.get('error'))}")

    print(f"\nTo run live SAE experiments:")
    for label, res in results.items():
        if res["status"] != "success":
            continue
        cfg = MODEL_REGISTRY[label]
        layer = args.layer if args.layer is not None else cfg["layer_idx"]
        print(f"\n  # {label}")
        print(f"  python run_sae_experiments.py \\")
        print(f"      --results experiment_results/<your_results>.json \\")
        print(f"      --model-label {label} \\")
        print(f"      --reader-model {cfg['local_path']} \\")
        print(f"      --sae-path {res['path']}/sae_weights.pt \\")
        print(f"      --layer-idx {layer} \\")
        print(f"      --d-sae {args.d_sae} \\")
        print(f"      --device {args.device} \\")
        print(f"      --experiments e1 e3 e5")

    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "models": results,
            "config": {
                "d_sae": args.d_sae,
                "training_tokens": args.training_tokens,
                "dataset": args.dataset,
                "context_size": args.context_size,
                "total_time_minutes": round(total_elapsed / 60, 1),
            },
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
