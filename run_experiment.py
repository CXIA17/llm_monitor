#!/usr/bin/env python3
"""
CLI for running multi-agent interaction experiments.

Usage examples:

  # Basic debate with default model (no injection)
  python run_experiment.py --question "Is AI dangerous?" --topology debate

  # Court trial with sycophancy injection on plaintiff
  python run_experiment.py \
      --question "Should the defendant pay damages for breach of contract?" \
      --topology court \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --probe-category sycophancy \
      --injection-type gated \
      --injection-strength 2.0 \
      --inject-agents plaintiff_attorney

  # Adversarial with toxicity steering on attacker
  python run_experiment.py \
      --question "Should social media be regulated?" \
      --topology adversarial \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --probe-category toxicity \
      --injection-type steer \
      --injection-strength 3.0 \
      --injection-direction subtract \
      --inject-agents attacker_1

  # Comparison mode (baseline vs injected)
  python run_experiment.py \
      --question "Is capitalism ethical?" \
      --topology debate \
      --probe-category overconfidence \
      --injection-type gated \
      --injection-strength 2.5 \
      --inject-agents critic \
      --compare

  # Extract LLM DNA after experiment (5-step paper pipeline)
  python run_experiment.py \
      --question "Is AI dangerous?" \
      --topology debate \
      --extract-dna \
      --dna-output dna_results.json

  # List available options
  python run_experiment.py --list-topologies
  python run_experiment.py --list-probes
  python run_experiment.py --list-agents
"""

import os
import sys
import json
import time
import argparse
import pickle
import glob
import numpy as np

import torch


def find_probe_file(model_name: str, probe_dir: str = "trained_probes") -> str | None:
    """Find a probe pickle file matching the model name."""
    # Normalize: Qwen/Qwen2.5-0.5B-Instruct -> Qwen_Qwen2.5-0.5B-Instruct
    normalized = model_name.replace("/", "_")

    # Check trained_probes/ directory
    candidate = os.path.join(probe_dir, f"{normalized}_probes.pkl")
    if os.path.exists(candidate):
        return candidate

    # Check root directory
    candidate = os.path.join(os.path.dirname(__file__), f"trained_probes_{normalized}.pkl")
    if os.path.exists(candidate):
        return candidate

    # Fuzzy match
    for pkl in glob.glob(os.path.join(probe_dir, "*.pkl")):
        base = os.path.basename(pkl).replace("_probes.pkl", "")
        if normalized.lower() in base.lower() or base.lower() in normalized.lower():
            return pkl

    return None


def list_available_probes(probe_dir: str = "trained_probes"):
    """List all available probe files and their categories."""
    print("\n=== Available Probe Files ===\n")

    for pkl_path in sorted(glob.glob(os.path.join(probe_dir, "*.pkl"))):
        basename = os.path.basename(pkl_path)
        model_name = basename.replace("_probes.pkl", "")
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            categories = list(data.keys())
            print(f"  {model_name}")
            for cat in categories:
                meta = data[cat].get("metadata", {})
                acc = meta.get("accuracy", "?")
                print(f"    - {cat} (accuracy: {acc})")
        except Exception as e:
            print(f"  {model_name} (error reading: {e})")
        print()

    # Also check root-level pkl files
    for pkl_path in sorted(glob.glob("trained_probes_*.pkl")):
        basename = os.path.basename(pkl_path)
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            categories = list(data.keys())
            print(f"  {basename}")
            for cat in categories:
                meta = data[cat].get("metadata", {})
                acc = meta.get("accuracy", "?")
                print(f"    - {cat} (accuracy: {acc})")
        except Exception as e:
            print(f"  {basename} (error: {e})")
        print()


def list_topologies():
    """List available interaction topologies."""
    print("\n=== Available Topologies ===\n")
    topologies = {
        "debate": "Proposer <-> Critic -> Judge (3 agents)",
        "panel": "Moderator -> Expert_1, Expert_2 (3 agents)",
        "adversarial": "Defender <-> Attacker -> Evaluator (3 agents)",
        "court": "Plaintiff <-> Defense -> Judge (3 agents, legal simulation)",
        "linear": "A -> B -> C sequential chain (custom agents via --agents)",
        "round_robin": "A -> B -> C -> A cyclic (custom agents via --agents)",
        "hub_spoke": "Hub <-> Spoke1, Spoke2, ... (custom agents via --agents)",
    }
    for name, desc in topologies.items():
        print(f"  {name:15s} {desc}")
    print()


def list_agent_templates():
    """List available agent templates."""
    from core.agent_registry import AgentRegistry
    print("\n=== Available Agent Templates ===\n")
    for name, config in AgentRegistry.TEMPLATES.items():
        print(f"  {name:25s} {config.display_name:25s} behavior={config.behavior.value}, temp={config.temperature}")
    print()


def load_probe(probe_path: str, category: str):
    """Load a probe from a pickle file."""
    from core.orchestrator import ProbeConfig

    with open(probe_path, "rb") as f:
        probe_data = pickle.load(f)

    if category not in probe_data:
        available = list(probe_data.keys())
        print(f"Error: probe category '{category}' not found in {probe_path}")
        print(f"Available categories: {available}")
        sys.exit(1)

    p = probe_data[category]
    direction = p.get("direction") or p.get("weights")
    metadata = p.get("metadata", {})

    if direction is None:
        print(f"Error: no direction vector found for probe '{category}'")
        sys.exit(1)

    return ProbeConfig(
        category=category,
        direction=np.array(direction),
        layer_idx=metadata.get("layer_idx", 12),
        hidden_size=metadata.get("hidden_size", 4096),
    )


def build_topology(args, agent_ids: list):
    """Build an InteractionGraph from CLI arguments."""
    from core.interaction_graph import InteractionGraph

    topo = args.topology

    if topo == "debate":
        return InteractionGraph.create_debate_topology(agent_ids)
    elif topo == "panel":
        return InteractionGraph.create_panel_discussion(agent_ids[0], agent_ids[1:])
    elif topo == "adversarial":
        return InteractionGraph.create_adversarial(agent_ids[0], agent_ids[1:-1], agent_ids[-1])
    elif topo == "court":
        return InteractionGraph.create_court_topology(*agent_ids[:3])
    elif topo == "linear":
        return InteractionGraph.create_linear(agent_ids)
    elif topo == "round_robin":
        return InteractionGraph.create_round_robin(agent_ids)
    elif topo == "hub_spoke":
        return InteractionGraph.create_hub_spoke(agent_ids[0], agent_ids[1:])
    else:
        print(f"Error: unknown topology '{topo}'")
        sys.exit(1)


def build_registry_and_graph(args):
    """Build agent registry and interaction graph from CLI args."""
    from core.agent_registry import (
        AgentRegistry, create_debate_agents, create_court_agents,
        create_adversarial_team,
    )

    registry = AgentRegistry()
    topo = args.topology

    if topo == "debate":
        agent_ids = create_debate_agents(registry)
    elif topo == "panel":
        registry.register_from_template("mediator", "moderator")
        registry.register_from_template("researcher", "expert_1")
        registry.register_from_template("devil_advocate", "expert_2")
        agent_ids = ["moderator", "expert_1", "expert_2"]
    elif topo == "adversarial":
        agent_ids = create_adversarial_team(registry)
    elif topo == "court":
        agent_ids = create_court_agents(registry)
    elif topo in ("linear", "round_robin", "hub_spoke"):
        # Use custom --agents or default to debate agents
        if args.agents:
            templates = args.agents
        else:
            templates = ["proposer", "critic", "judge"]
        agent_ids = []
        for t in templates:
            name = t.strip()
            if name in AgentRegistry.TEMPLATES:
                registry.register_from_template(name)
                agent_ids.append(name)
            else:
                # Register as custom agent
                registry.register(name, system_prompt=f"You are {name}. Participate in the discussion.")
                agent_ids.append(name)
    else:
        print(f"Error: unknown topology '{topo}'")
        sys.exit(1)

    graph = build_topology(args, agent_ids)
    return registry, graph, agent_ids


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-agent interaction experiments with probe injection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --question "Is AI dangerous?" --topology debate
  %(prog)s --question "Contract dispute" --topology court --probe-category sycophancy --injection-type gated --injection-strength 2.0 --inject-agents plaintiff_attorney
  %(prog)s --question "Is AI dangerous?" --topology debate --extract-dna --dna-output dna_results.json
  %(prog)s --list-topologies
  %(prog)s --list-probes
  %(prog)s --list-agents
        """,
    )

    # ── Info commands ──
    info = parser.add_argument_group("info")
    info.add_argument("--list-topologies", action="store_true", help="List available topologies")
    info.add_argument("--list-probes", action="store_true", help="List available probe files and categories")
    info.add_argument("--list-agents", action="store_true", help="List available agent templates")

    # ── Experiment settings ──
    exp = parser.add_argument_group("experiment")
    exp.add_argument("-q", "--question", type=str, help="The question/topic for agents to discuss")
    exp.add_argument("-t", "--topology", type=str, default="debate",
                     choices=["debate", "panel", "adversarial", "court", "linear", "round_robin", "hub_spoke"],
                     help="Interaction topology (default: debate)")
    exp.add_argument("-r", "--rounds", type=int, default=4, help="Number of interaction rounds (default: 4)")
    exp.add_argument("--agents", nargs="+", help="Agent templates for linear/round_robin/hub_spoke topologies")

    # ── Model settings ──
    model_grp = parser.add_argument_group("model")
    model_grp.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                           help="HuggingFace model name or local path (default: Qwen/Qwen2.5-0.5B-Instruct)")
    model_grp.add_argument("--model-dir", type=str, default=None,
                           help="Directory containing local models (prepended to model name)")
    model_grp.add_argument("--device", type=str, default="auto",
                           help="Device: auto, cpu, cuda, cuda:0, etc. (default: auto)")
    model_grp.add_argument("--dtype", type=str, default=None,
                           choices=["float16", "bfloat16", "float32"],
                           help="Model data type")
    model_grp.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit quantization")
    model_grp.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization")

    # ── Probe & injection settings ──
    inject = parser.add_argument_group("injection")
    inject.add_argument("--probe-category", type=str, default=None,
                        help="Probe category to use (e.g., sycophancy, toxicity, overconfidence)")
    inject.add_argument("--probe-path", type=str, default=None,
                        help="Explicit path to probe pickle file (auto-detected from model if not set)")
    inject.add_argument("--injection-type", type=str, default="gated",
                        choices=["gated", "amplify", "ablate", "steer"],
                        help="Injection type (default: gated)")
    inject.add_argument("--injection-strength", type=float, default=2.0,
                        help="Injection strength (default: 2.0)")
    inject.add_argument("--injection-direction", type=str, default="add",
                        choices=["add", "subtract"],
                        help="Injection direction (default: add)")
    inject.add_argument("--gate-bias", type=float, default=-0.3,
                        help="Gate bias threshold for gated injection (default: -0.3)")
    inject.add_argument("--inject-agents", nargs="+", default=None,
                        help="Agent IDs to inject (e.g., critic proposer)")

    # ── DNA extraction settings ──
    dna_grp = parser.add_argument_group("dna", "LLM DNA extraction (5-step paper pipeline)")
    dna_grp.add_argument("--extract-dna", action="store_true",
                         help="Extract LLM DNA from agent responses after the experiment")
    dna_grp.add_argument("--dna-output", type=str, default=None,
                         help="Save DNA results to JSON file")
    dna_grp.add_argument("--dna-dim", type=int, default=128,
                         help="DNA vector dimensionality (default: 128)")
    dna_grp.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-8B",
                         help="Sentence embedding model for DNA extraction (default: Qwen/Qwen3-Embedding-8B)")
    dna_grp.add_argument("--dna-projection-seed", type=int, default=42,
                         help="Seed for Gaussian projection matrix (default: 42)")

    # ── Output settings ──
    output = parser.add_argument_group("output")
    output.add_argument("-o", "--output", type=str, default=None,
                        help="Save results to JSON file")
    output.add_argument("--compare", action="store_true",
                        help="Run comparison: baseline (no injection) vs injected")
    output.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output (default: True)")
    output.add_argument("--quiet", action="store_true",
                        help="Minimal output")

    args = parser.parse_args()

    # ── Handle info commands ──
    if args.list_topologies:
        list_topologies()
        return
    if args.list_probes:
        list_available_probes()
        return
    if args.list_agents:
        list_agent_templates()
        return

    # ── Validate ──
    if not args.question:
        parser.error("--question is required (or use --list-topologies, --list-probes, --list-agents)")

    verbose = not args.quiet

    # ── Load model ──
    from core.model_compatibility import load_model_and_tokenizer

    model_name = args.model
    if args.model_dir:
        model_name = os.path.join(args.model_dir, model_name.replace("/", "_"))

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype) if args.dtype else None

    print(f"\n{'='*60}")
    print(f"  LLM Monitor - CLI Experiment Runner")
    print(f"{'='*60}")
    print(f"  Model:     {model_name}")
    print(f"  Topology:  {args.topology}")
    print(f"  Rounds:    {args.rounds}")
    print(f"  Question:  {args.question[:80]}{'...' if len(args.question) > 80 else ''}")

    model, tokenizer, compat = load_model_and_tokenizer(
        model_name,
        device=args.device,
        dtype=dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    device = str(next(model.parameters()).device)

    # ── Load probe ──
    probe = None
    if args.probe_category:
        probe_path = args.probe_path or find_probe_file(model_name)
        if probe_path is None:
            print(f"\nWarning: no probe file found for model '{model_name}'")
            print("Use --probe-path to specify one, or --list-probes to see available probes.")
            print("Continuing without probe.\n")
        else:
            print(f"  Probe:     {args.probe_category} from {probe_path}")
            probe = load_probe(probe_path, args.probe_category)

    # ── Build orchestrator ──
    from core.orchestrator import MultiAgentOrchestrator, InjectionConfig

    orchestrator = MultiAgentOrchestrator(model, tokenizer, device, probe, compat)
    registry, graph, agent_ids = build_registry_and_graph(args)
    orchestrator.load_registry(registry)
    orchestrator.set_interaction_graph(graph)

    # ── Configure injection ──
    injection_config = None
    if args.inject_agents and args.probe_category and probe is not None:
        injection_config = InjectionConfig(
            injection_type=args.injection_type,
            strength=args.injection_strength,
            direction=args.injection_direction,
            gate_bias=args.gate_bias,
        )
        print(f"  Injection: {args.injection_type} x{args.injection_strength} ({args.injection_direction})")
        print(f"  Targets:   {args.inject_agents}")

        for agent_id in args.inject_agents:
            if agent_id not in orchestrator.agents:
                print(f"\nError: agent '{agent_id}' not found. Available: {list(orchestrator.agents.keys())}")
                sys.exit(1)
            orchestrator.set_injection(agent_id, injection_config)

    if args.extract_dna:
        print(f"  DNA:       dim={args.dna_dim}, embedding={args.embedding_model}")

    print(f"{'='*60}\n")

    # ── Run experiment ──
    if args.compare and injection_config and args.inject_agents:
        results = orchestrator.run_comparison(
            question=args.question,
            target_agents=args.inject_agents,
            injection_config=injection_config,
            num_rounds=args.rounds,
            verbose=verbose,
        )
        _print_comparison_summary(results)

        if args.output:
            output_data = {k: v.to_dict() for k, v in results.items()}
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")

        # DNA extraction for comparison mode
        if args.extract_dna:
            _extract_and_save_dna(
                results={"baseline": results["baseline"], "injected": results["injected"]},
                args=args,
                device=device,
                verbose=verbose,
                mode="comparison",
            )
    else:
        result = orchestrator.run_experiment(
            question=args.question,
            num_rounds=args.rounds,
            verbose=verbose,
        )
        _print_summary(result)

        if args.output:
            result.save(args.output)
            print(f"\nResults saved to {args.output}")

        # DNA extraction for single experiment
        if args.extract_dna:
            _extract_and_save_dna(
                results={"experiment": result},
                args=args,
                device=device,
                verbose=verbose,
                mode="single",
            )


def _extract_and_save_dna(results, args, device, verbose, mode):
    """Extract LLM DNA from experiment results using the 5-step pipeline."""
    from core.llm_dna_extractor import LLMDNAExtractor

    print(f"\n{'='*60}")
    print(f"  LLM DNA EXTRACTION (5-step pipeline)")
    print(f"{'='*60}")

    extractor = LLMDNAExtractor(
        embedding_model=args.embedding_model,
        dna_dim=args.dna_dim,
        projection_seed=args.dna_projection_seed,
    )

    all_dna = {}

    for exp_name, result in results.items():
        # Collect responses per agent from transcript
        agent_responses = {}
        for entry in result.transcript:
            agent_id = entry.get("agent_id", "")
            response = entry.get("response", "")
            if agent_id and response:
                agent_responses.setdefault(agent_id, []).append(response)

        if not agent_responses:
            print(f"  Warning: no agent responses found in {exp_name} transcript")
            continue

        for agent_id, responses in agent_responses.items():
            label = f"{exp_name}/{agent_id}" if mode == "comparison" else agent_id
            if verbose:
                print(f"\n  Extracting DNA for: {label} ({len(responses)} responses)")

            dna = extractor.extract_from_responses(
                responses=responses,
                device=device,
                model_name=f"{args.model}_{label}",
                verbose=verbose,
            )
            all_dna[label] = dna

    # Print DNA summary
    if all_dna:
        print(f"\n  {'Agent':<35s} {'DNA Norm':>10s} {'Dim':>5s}")
        print(f"  {'-'*50}")
        for label, dna in all_dna.items():
            print(f"  {label:<35s} {np.linalg.norm(dna.vector):>10.3f} {dna.dna_dim:>5d}")

        # Print pairwise distances if multiple agents
        if len(all_dna) > 1:
            labels = list(all_dna.keys())
            print(f"\n  Pairwise Cosine Distances:")
            print(f"  {'-'*50}")
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    dist = all_dna[labels[i]].distance_to(all_dna[labels[j]], "cosine")
                    print(f"  {labels[i]:>20s} <-> {labels[j]:<20s}  {dist:.4f}")

    # Save DNA results
    dna_output = args.dna_output
    if dna_output and all_dna:
        dna_data = {label: dna.to_dict() for label, dna in all_dna.items()}
        with open(dna_output, "w") as f:
            json.dump(dna_data, f, indent=2)
        print(f"\n  DNA results saved to {dna_output}")

    print()


def _print_summary(result):
    """Print a summary of experiment results."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Duration: {result.duration():.1f}s")
    print(f"  Rounds:   {result.num_rounds}")
    print()

    for agent_id, metrics in result.agent_metrics.items():
        scores = metrics.probe_scores
        if scores:
            print(f"  {agent_id:25s}  mean={metrics.mean_score():.3f}  "
                  f"min={min(scores):.3f}  max={max(scores):.3f}  "
                  f"tokens={metrics.total_tokens_generated}")
        else:
            print(f"  {agent_id:25s}  (no scores)")

    if result.disagreement_scores:
        print(f"\n  Disagreement (std): {[f'{d:.3f}' for d in result.disagreement_scores]}")
    print()


def _print_comparison_summary(results):
    """Print comparison summary between baseline and injected."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")

    baseline = results.get("baseline")
    injected = results.get("injected")

    if not baseline or not injected:
        return

    print(f"\n  {'Agent':<25s} {'Baseline':>10s} {'Injected':>10s} {'Delta':>10s}")
    print(f"  {'-'*55}")

    for agent_id in baseline.agent_metrics:
        b_score = baseline.agent_metrics[agent_id].mean_score()
        i_score = injected.agent_metrics[agent_id].mean_score()
        delta = i_score - b_score
        marker = " ***" if abs(delta) > 0.1 else ""
        print(f"  {agent_id:<25s} {b_score:>10.3f} {i_score:>10.3f} {delta:>+10.3f}{marker}")

    b_disagree = np.mean(baseline.disagreement_scores) if baseline.disagreement_scores else 0
    i_disagree = np.mean(injected.disagreement_scores) if injected.disagreement_scores else 0
    print(f"\n  {'Avg Disagreement':<25s} {b_disagree:>10.3f} {i_disagree:>10.3f} {i_disagree - b_disagree:>+10.3f}")
    print()


if __name__ == "__main__":
    main()
