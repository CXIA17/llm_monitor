#!/usr/bin/env python3
"""
Experiment Pre-flight Checker
==============================

Validates that a given model + probe combination is compatible and ready
to run before launching the full injection experiment suite. Checks:

  1. Model path exists (local) or is a valid HuggingFace repo name
  2. Probe file exists and can be loaded
  3. Requested probe categories are present in the probe file
  4. Probe hidden_size matches the model's actual hidden_size
  5. Probe layer_idx is within the model's num_layers
  6. Model can be loaded and tokenizer works
  7. (Optional) Judge API key is set for H1

Reports PASS / WARN / FAIL for each check, then prints a final verdict
and the exact CLI command to run if everything is valid.

Usage:
  # Check a specific model + probe file
  python check_experiment_config.py \\
      -m Qwen/Qwen3-4B \\
      --model-dir ./models \\
      --probe-path trained_probes_qwen3_4b.pkl \\
      --device cuda:5

  # Check all available model/probe combinations automatically
  python check_experiment_config.py --scan --model-dir ./models

  # Check with H1 judge validation
  python check_experiment_config.py \\
      -m EleutherAI/pythia-1b \\
      --model-dir ./models \\
      --probe-path trained_probes/EleutherAI_pythia-1b_probes.pkl \\
      --judge-api chatfire \\
      --device cuda:5
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple

# ── ANSI colours ──
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"

# Default probes to test (mirrors run_injection_experiments.py)
DEFAULT_PROBES = ["sentiment", "sycophancy", "toxicity"]

# Known probe file locations
PROBE_DIR = "trained_probes"
PROBE_CANDIDATES = [
    ("Qwen/Qwen3-4B",                     "trained_probes_qwen3_4b.pkl"),
    ("Qwen/Qwen2.5-0.5B-Instruct",        "trained_probes/Qwen_Qwen2.5-0.5B-Instruct_probes.pkl"),
    ("EleutherAI/pythia-1b",              "trained_probes/EleutherAI_pythia-1b_probes.pkl"),
    ("EleutherAI/pythia-2.8b",            "trained_probes/EleutherAI_pythia-2.8b_probes.pkl"),
    ("meta-llama/Llama-3.2-1B-Instruct",  "trained_probes/meta-llama_Llama-3.2-1B-Instruct_probes.pkl"),
    ("google/gemma-3-1b-it",              "trained_probes/google_gemma-3-1b-it_probes.pkl"),
]


# ═════════════════════════════════════════════════════════════════
# Individual checks
# ═════════════════════════════════════════════════════════════════

def check_model_path(model_name: str, model_dir: Optional[str]) -> Tuple[bool, str, str]:
    """Check 1: Model path or HuggingFace name is valid."""
    if model_dir:
        local_path = os.path.join(model_dir, model_name)
        if os.path.isdir(local_path):
            return True, PASS, f"Local path exists: {local_path}"
        # Also try replacing / with _ (legacy naming)
        alt_path = os.path.join(model_dir, model_name.replace("/", "_"))
        if os.path.isdir(alt_path):
            return True, PASS, f"Local path exists: {alt_path}"
        return False, FAIL, f"Local path not found: {local_path}"
    else:
        # HuggingFace name — check format only, don't download
        if "/" in model_name or os.path.isdir(model_name):
            return True, WARN, f"Will download from HuggingFace: {model_name}"
        return False, FAIL, f"Not a valid model name or path: {model_name}"


def check_probe_file(probe_path: str) -> Tuple[bool, str, str]:
    """Check 2: Probe file exists and is loadable."""
    if not os.path.exists(probe_path):
        return False, FAIL, f"Probe file not found: {probe_path}"
    try:
        with open(probe_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            return False, FAIL, f"Unexpected probe format (expected dict, got {type(data).__name__})"
        return True, PASS, f"Loaded probe file: {probe_path}  (keys: {list(data.keys())})"
    except Exception as e:
        return False, FAIL, f"Failed to load probe file: {e}"


def check_probe_categories(probe_path: str, requested: list) -> Tuple[bool, str, str]:
    """Check 3: All requested probe categories are present."""
    with open(probe_path, "rb") as f:
        data = pickle.load(f)
    available = list(data.keys())
    missing = [p for p in requested if p not in available]
    if missing:
        return False, FAIL, f"Missing probes: {missing}  (available: {available})"
    return True, PASS, f"All requested probes found: {requested}"


def check_probe_architecture(probe_path: str, model_name: str, model_dir: Optional[str]) -> Tuple[bool, str, str]:
    """Check 4+5: Probe hidden_size and layer_idx match model architecture."""
    with open(probe_path, "rb") as f:
        data = pickle.load(f)

    # Get probe architecture from first available probe
    sample_probe = next(iter(data.values()))
    probe_hidden = sample_probe.get("metadata", {}).get("hidden_size") or \
                   (sample_probe.get("direction").shape[0] if sample_probe.get("direction") is not None else None)
    probe_layers = {k: v["metadata"].get("layer_idx", v.get("layer_idx")) for k, v in data.items()}
    max_layer = max(probe_layers.values()) if probe_layers else 0

    # Load model config (fast, no weights)
    try:
        from transformers import AutoConfig
        local_path = os.path.join(model_dir, model_name) if model_dir else model_name
        if model_dir and not os.path.isdir(local_path):
            local_path = os.path.join(model_dir, model_name.replace("/", "_"))
        config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)

        model_hidden = getattr(config, "hidden_size", None)
        model_layers = getattr(config, "num_hidden_layers",
                       getattr(config, "num_layers", None))

        issues = []
        if model_hidden and probe_hidden and model_hidden != probe_hidden:
            issues.append(f"hidden_size mismatch: probe={probe_hidden} vs model={model_hidden}")
        if model_layers and max_layer >= model_layers:
            issues.append(f"probe layer {max_layer} >= model num_layers {model_layers}")

        if issues:
            return False, FAIL, "Architecture mismatch: " + "; ".join(issues)

        return True, PASS, (
            f"Architecture match: hidden_size={model_hidden}, "
            f"probe max_layer={max_layer} < num_layers={model_layers}"
        )
    except Exception as e:
        return False, WARN, f"Could not verify architecture (config load failed): {e}"


def check_model_loadable(model_name: str, model_dir: Optional[str], device: str) -> Tuple[bool, str, str]:
    """Check 6: Model and tokenizer can actually be loaded."""
    local_path = os.path.join(model_dir, model_name) if model_dir else model_name
    if model_dir and not os.path.isdir(local_path):
        local_path = os.path.join(model_dir, model_name.replace("/", "_"))

    try:
        from core.model_compatibility import load_model_and_tokenizer
        print(f"    Loading model on {device} (this may take a moment)...")
        model, tokenizer, compat = load_model_and_tokenizer(
            local_path, device=device, dtype=None,
            load_in_8bit=False, load_in_4bit=False,
        )
        # Quick tokenizer smoke test
        tokens = tokenizer("Hello world", return_tensors="pt")
        info = (f"family={compat.family}, hidden={compat.hidden_size}, "
                f"layers={compat.num_layers}")
        # Free memory
        del model
        import torch, gc
        gc.collect()
        if "cuda" in device:
            torch.cuda.empty_cache()
        return True, PASS, f"Model loaded successfully ({info})"
    except Exception as e:
        return False, FAIL, f"Model load failed: {e}"


def check_judge_api_key(judge_api: str) -> Tuple[bool, str, str, Optional[str]]:
    """Check 7a: Judge API key exists in environment. Returns (ok, tag, msg, key)."""
    key_map = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "chatfire":  "CHATFIRE_API_KEY",
    }
    env_var = key_map.get(judge_api)
    if not env_var:
        return False, FAIL, f"Unknown judge API: {judge_api}", None
    key = os.environ.get(env_var)
    if key:
        masked = key[:8] + "..." + key[-4:]
        return True, PASS, f"{env_var} is set ({masked})", key
    return False, FAIL, f"{env_var} is not set in environment", None


def check_judge_api_live(judge_api: str, judge_model: str, api_key: Optional[str]) -> Tuple[bool, str, str]:
    """Check 7b: Make a minimal live call to verify key, base URL, and model are valid."""
    base_url = None
    if judge_api == "chatfire":
        base_url = os.environ.get("CHATFIRE_BASE_URL", "https://api.chatfire.cn/v1")

    test_prompt = "Reply with one word: ready"

    try:
        if judge_api in ("openai", "chatfire"):
            from openai import OpenAI
            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=10,
            )
            reply = resp.choices[0].message.content.strip()
            return True, PASS, f"Live API call succeeded. Model replied: '{reply}'"

        elif judge_api == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=judge_model,
                max_tokens=10,
                messages=[{"role": "user", "content": test_prompt}],
            )
            reply = resp.content[0].text.strip()
            return True, PASS, f"Live API call succeeded. Model replied: '{reply}'"

    except ImportError as e:
        return False, FAIL, f"Missing library: {e}  (run: pip install {str(e).split()[-1]})"
    except Exception as e:
        short = str(e)[:120]
        return False, FAIL, f"Live API call failed: {short}"

    return False, FAIL, f"Unknown judge API: {judge_api}"


# ═════════════════════════════════════════════════════════════════
# Run all checks for one config
# ═════════════════════════════════════════════════════════════════

def run_checks(
    model_name: str,
    model_dir: Optional[str],
    probe_path: str,
    device: str,
    probes: list,
    judge_api: Optional[str],
    judge_model: str,
    load_model: bool,
) -> bool:
    """Run all checks and print results. Returns True if all critical checks pass."""

    print(f"\n{BOLD}{'─'*65}{RESET}")
    print(f"{BOLD}  Config: {model_name} | {probe_path}{RESET}")
    print(f"{'─'*65}")

    failures = 0
    warnings = 0

    def report(label, ok, tag, msg):
        nonlocal failures, warnings
        print(f"  {tag}  {label}: {msg}")
        if tag == FAIL:
            failures += 1
        elif tag == WARN:
            warnings += 1

    # 1. Model path
    ok, tag, msg = check_model_path(model_name, model_dir)
    report("Model path", ok, tag, msg)
    if not ok:
        print(f"\n  {FAIL} Cannot continue without a valid model path.")
        return False

    # 2. Probe file
    ok, tag, msg = check_probe_file(probe_path)
    report("Probe file", ok, tag, msg)
    if not ok:
        print(f"\n  {FAIL} Cannot continue without a valid probe file.")
        return False

    # 3. Probe categories
    ok, tag, msg = check_probe_categories(probe_path, probes)
    report("Probe categories", ok, tag, msg)

    # 4+5. Architecture match
    ok, tag, msg = check_probe_architecture(probe_path, model_name, model_dir)
    report("Architecture match", ok, tag, msg)

    # 6. Model loadable (optional — slow, requires GPU)
    if load_model:
        ok, tag, msg = check_model_loadable(model_name, model_dir, device)
        report("Model loadable", ok, tag, msg)
    else:
        print(f"  {WARN}  Model loadable: SKIPPED (use --load-model to test)")
        warnings += 1

    # 7a. Judge API key present
    api_key = None
    if judge_api:
        ok, tag, msg, api_key = check_judge_api_key(judge_api)
        report("Judge API key", ok, tag, msg)

    # 7b. Judge API live call
    if judge_api and api_key:
        ok, tag, msg = check_judge_api_live(judge_api, judge_model, api_key)
        report(f"Judge API live ({judge_model})", ok, tag, msg)
    elif judge_api and not api_key:
        print(f"  {WARN}  Judge API live: SKIPPED (no API key)")

    # Verdict
    print(f"\n  {'─'*55}")
    if failures == 0:
        print(f"  {GREEN}{BOLD}VERDICT: READY TO RUN{RESET}  "
              f"({warnings} warning(s), 0 failures)")
    else:
        print(f"  {RED}{BOLD}VERDICT: NOT READY{RESET}  "
              f"({failures} failure(s), {warnings} warning(s))")

    return failures == 0


def print_run_command(model_name: str, model_dir: Optional[str], probe_path: str,
                      device: str, judge_api: Optional[str], judge_model: Optional[str]):
    """Print the CLI command to run the experiment."""
    parts = ["python run_injection_experiments.py"]
    parts.append(f"    -m {model_name}")
    if model_dir:
        parts.append(f"    --model-dir {model_dir}")
    parts.append(f"    --probe-path {probe_path}")
    parts.append(f"    --device {device}")
    if judge_api:
        parts.append(f"    --judge-api {judge_api}")
        if judge_model:
            parts.append(f"    --judge-model {judge_model}")
    parts.append(f"    --experiments h1 h2 h3")

    # Derive output dir name from model
    safe_name = model_name.replace("/", "_").replace("-", "_").lower()
    parts.append(f"    -o experiment_results/{safe_name}/")

    print(f"\n  Run command:")
    print("  " + " \\\n".join(parts))


# ═════════════════════════════════════════════════════════════════
# Scan mode: auto-detect all known model/probe pairs
# ═════════════════════════════════════════════════════════════════

def scan_all(model_dir: Optional[str], device: str, probes: list,
             judge_api: Optional[str], judge_model: str, load_model: bool):
    """Check all known model/probe combinations."""
    print(f"\n{BOLD}Scanning all known model/probe combinations...{RESET}")
    ready = []
    not_ready = []

    for model_name, probe_path in PROBE_CANDIDATES:
        ok = run_checks(
            model_name=model_name,
            model_dir=model_dir,
            probe_path=probe_path,
            device=device,
            probes=probes,
            judge_api=judge_api,
            judge_model=judge_model,
            load_model=False,  # Don't load model for scan (too slow)
        )
        if ok:
            ready.append((model_name, probe_path))
        else:
            not_ready.append(model_name)

    print(f"\n{'═'*65}")
    print(f"{BOLD}  SCAN SUMMARY{RESET}")
    print(f"{'═'*65}")
    print(f"  {GREEN}Ready ({len(ready)}):{RESET}")
    for m, p in ready:
        print(f"    {m:<45s}  {p}")
    if not_ready:
        print(f"\n  {RED}Not ready ({len(not_ready)}):{RESET}")
        for m in not_ready:
            print(f"    {m}")

    if ready and load_model:
        print(f"\n  {YELLOW}Note: --load-model was ignored in scan mode "
              f"(would require loading each model sequentially).{RESET}")

    return ready


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight check for injection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--model", default=None,
                        help="HuggingFace model name (e.g. Qwen/Qwen3-4B)")
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing local model folders")
    parser.add_argument("--probe-path", default=None,
                        help="Path to probe .pkl file")
    parser.add_argument("--probes", nargs="+", default=DEFAULT_PROBES,
                        help=f"Probe categories to validate (default: {DEFAULT_PROBES})")
    parser.add_argument("--device", default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--judge-api", default=None,
                        choices=["openai", "anthropic", "chatfire"],
                        help="Judge API to validate (for H1)")
    parser.add_argument("--judge-model", default="claude-sonnet-4-5-20250929",
                        help="Judge model name (default: claude-sonnet-4-5-20250929)")
    parser.add_argument("--load-model", action="store_true",
                        help="Actually load the model to verify it works (slow, needs GPU)")
    parser.add_argument("--scan", action="store_true",
                        help="Check all known model/probe combinations automatically")
    args = parser.parse_args()

    if args.scan:
        ready = scan_all(
            model_dir=args.model_dir,
            device=args.device,
            probes=args.probes,
            judge_api=args.judge_api,
            judge_model=args.judge_model,
            load_model=args.load_model,
        )
        sys.exit(0 if ready else 1)

    # Single config check
    if not args.model:
        parser.error("Provide -m/--model or use --scan")
    if not args.probe_path:
        # Try auto-detect
        normalized = args.model.replace("/", "_")
        candidates = [
            f"trained_probes/{normalized}_probes.pkl",
            f"trained_probes_qwen3_4b.pkl",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.probe_path = c
                print(f"  Auto-detected probe file: {c}")
                break
        if not args.probe_path:
            parser.error(
                "Could not auto-detect probe file. Provide --probe-path explicitly.\n"
                f"Known probe files: {[p for _, p in PROBE_CANDIDATES]}"
            )

    ok = run_checks(
        model_name=args.model,
        model_dir=args.model_dir,
        probe_path=args.probe_path,
        device=args.device,
        probes=args.probes,
        judge_api=args.judge_api,
        judge_model=args.judge_model,
        load_model=args.load_model,
    )

    if ok:
        print_run_command(
            model_name=args.model,
            model_dir=args.model_dir,
            probe_path=args.probe_path,
            device=args.device,
            judge_api=args.judge_api,
            judge_model=args.judge_model,
        )
    else:
        print(f"\n  Fix the failures above before running the experiment.")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
