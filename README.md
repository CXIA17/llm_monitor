# LLM Behavioral Monitor

A system for monitoring, steering, and analyzing behavioral patterns in multi-agent LLM interactions using linear probes and activation engineering.

## Why This Project Exists

Large language models exhibit subtle behavioral patterns — sycophancy, overconfidence, toxicity, deception — that are invisible in the output text but detectable in the model's internal activations. This project lets you:

- **Detect** these behaviors in real time using linear probes trained on hidden-state activations
- **Steer** agent behavior by injecting learned direction vectors into the model's residual stream during generation
- **Compare** how different models and agent roles exhibit these behaviors across structured interactions
- **Visualize** behavioral signatures as 128-dimensional DNA fingerprints projected into a 2D galaxy

## Real-World Applications

### AI Safety Research
Train probes on deception, sycophancy, or toxicity datasets and inject them into multi-agent debates to study how behavioral traits propagate between agents. Observe whether one sycophantic agent shifts the group's tone, or whether a toxic agent's influence decays over rounds.

### Red-Teaming and Alignment Evaluation
Use the court simulation to stress-test models in adversarial settings. Inject a sycophancy probe into a judge agent and observe whether it produces contradictory rulings that validate both sides. Compare how different model families (Qwen, Llama, Mistral) resist or amplify injected behaviors.

### Model Selection and Benchmarking
Run the same multi-agent scenario across different models and compare their behavioral DNA fingerprints in the galaxy visualization. Identify which models are more susceptible to sycophancy under pressure, which maintain calibration, and which degrade under adversarial prompting.

### Interpretability Research
Use SAE fingerprinting to decompose model activations into sparse, interpretable features. The latent interpreter automatically labels what each SAE feature represents (e.g., "legal procedural language", "hedging markers") by analyzing top-activating examples.

### LLM-Powered Application Auditing
If you deploy multi-agent systems in production (e.g., AI-assisted legal review, automated content moderation), this tool lets you monitor behavioral drift over time. Train probes on your domain-specific concerns and track whether agents maintain their intended behavioral profile.

## Architecture

```
localhost:8000/              Landing page
localhost:8000/court/        Federal Court Simulation
localhost:8000/dashboard/    Multi-Agent Monitor
```

Both dashboards run on a single FastAPI server. The launcher mounts them as sub-applications.

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (or CPU for simulated mode)
- A local model (Qwen, Llama, Mistral, etc.)

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
# GPU with Qwen3-4B
python launcher.py --model Qwen_Qwen3-4B --device cuda:0

# CPU / simulated mode
python launcher.py --device cpu

# Specific GPU and model directory
python launcher.py --model Qwen_Qwen3-4B --device cuda:6 --model-dir /path/to/models

# Only court dashboard
python launcher.py --model Qwen_Qwen3-4B --device cuda:0 --no-dashboard

# Only multi-agent dashboard
python launcher.py --model Qwen_Qwen3-4B --device cuda:0 --no-court
```

### Docker
```bash
# Build
docker build -t llm-monitor .

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -v /path/to/models:/models \
  llm-monitor --model Qwen_Qwen3-4B --device cuda:0

# Docker Compose
docker compose up
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen_Qwen2.5-0.5B-Instruct` | Model name (directory name under model-dir) |
| `--model-dir` | `models` (or `$MODEL_DIR`) | Directory containing local models |
| `--device` | `cuda:0` | Device for inference (`cuda:N` or `cpu`) |
| `--port` | `8000` | Server port |
| `--host` | `0.0.0.0` | Server host |
| `--no-court` | | Skip mounting court dashboard |
| `--no-dashboard` | | Skip mounting multi-agent dashboard |

## Two Dashboards

### Federal Court Simulation (`/court/`)

A structured legal proceeding with 6 fixed roles:

| Role | Description |
|------|-------------|
| Judge | Presides, enforces procedure, delivers rulings |
| Plaintiff's Counsel | Advocates for the plaintiff |
| Defense Counsel | Defends against claims |
| Jury Foreperson | Leads jury deliberation |
| Clerk | Records proceedings |
| Witness | Provides testimony |

The trial proceeds through 6 phases: Motions, Opening Statements, Examination, Closing Arguments, Jury Deliberation, and Verdict. You can inject probes into any agent and observe how steering affects their legal reasoning across phases.

### Multi-Agent Monitor (`/dashboard/`)

Configurable multi-agent interactions with:

**8 Agent Templates:** Proposer, Critic, Judge, Researcher, Devil's Advocate, Mediator, Fact Checker, Strategist

**6 Interaction Topologies:**
- Linear Chain — A → B → C sequential
- Round Robin — all agents take repeating turns
- Hub & Spoke — central mediator coordinates others
- Debate — proposer vs critic with judge evaluator
- Panel Discussion — moderator guides panelists
- Adversarial — challengers vs defender with judge

## Probe System

### What Probes Are

Linear classifiers trained on model hidden-state activations. A probe learns a direction vector in activation space that separates two behavioral classes (e.g., sycophantic vs. calibrated responses). During generation, the probe scores each token by projecting the current activation onto this direction.

### Available Probe Categories

Overconfidence, Uncertainty, Hedging, Sycophancy, Toxicity, Formality, Emotional Valence, Deception, Safety, Refusal, Helpfulness, Specificity

### Training Probes

```bash
# Download real datasets
python download_dataset.py

# Or generate synthetic datasets
python dataset_generator.py
```

Probes are trained through the dashboard UI or via the `MultiProbeTrainer` API. Training uses logistic regression on hidden-state activations collected from labeled text pairs.

### How Injection Works

When a probe is injected into an agent, three mechanisms activate during generation:

**1. Multi-Layer Injection**
The probe's direction vector is injected across multiple transformer layers (from layer N/4 to N-1), with per-layer strength scaled by `strength / n_layers^0.65`. This provides sustained steering without triggering self-repair mechanisms.

**2. Dynamic Scaling**
Instead of a fixed injection magnitude, the perturbation scales proportionally to the current residual-stream norm: `delta = strength * sign * stream_norm * 0.1`. This prevents over-injection when activations are small and under-injection when they are large.

**3. Gated Injection**
Before injecting, the hook reads the probe score from the monitor layer. Injection only fires when the model is drifting from the desired behavior:
- `direction="add"`: inject when score < threshold (push score up)
- `direction="subtract"`: inject when score > threshold (push score down)

The gate threshold is controlled by `gate_bias` (default: -0.3).

## Behavioral DNA

Each agent's behavior across an experiment is compressed into a 128-dimensional fingerprint:

| Feature Group | Dimensions | What It Captures |
|---------------|-----------|------------------|
| Token-Level | 12 | Score distribution, gradients, peaks |
| Temporal | 10 | Drift, oscillation, convergence |
| Cross-Agent | 8 | Reactivity, mirroring, dominance |
| Linguistic | 10 | Hedging, assertiveness, emotion |
| Probe Interaction | 6 | Confidence-persuasion balance |
| Injection Effects | 8 | Absorption, decay, resistance |
| Role Compliance | 6 | In-role vs out-of-role behavior |
| Composite | 8 | Advocacy, judicial quality, calibration |

These vectors are projected into a 2D galaxy visualization for cross-experiment comparison. Injected agents cluster separately from baseline agents, and different model families form distinct regions.

### SAE Enrichment

Optionally, 16 sparse autoencoder features can be appended (total: 144D) for finer-grained behavioral decomposition with interpretable semantic labels.

## Supported Models

21+ model families including: Qwen/Qwen-2/Qwen-3, Llama/Llama-2/Llama-3, Mistral/Mixtral, Phi-1/2/3, Gemma/Gemma-2, GPT-2/GPT-Neo/GPT-J, Falcon, OLMo, StableLM, Bloom, OPT, Pythia, and more. The `ModelCompatibility` layer auto-detects architecture and adapts layer access patterns.

## Project Structure

```
launcher.py                  Main entry point (unified server)
dashboard_server.py          Multi-agent dashboard
court_dashboard.py           Court simulation dashboard
behavioural_dna.py           128D behavioral DNA extraction
dna_extractor.py             DNA extraction utilities
cross_model_galaxy.py        Cross-model comparison
dataset_generator.py         Synthetic dataset generation
download_dataset.py          Real dataset downloading
core/
  steered_agent.py           Probe-based steering (dynamic scaling, gating)
  orchestrator.py            Multi-agent orchestration
  probe_trainer.py           Probe training pipeline
  agent_registry.py          Agent templates and configuration
  interaction_graph.py       Topology definitions
  court_orchestrator.py      Court simulation runner
  llm_dna.py                 DNA signature computation
  sae_fingerprint.py         SAE feature extraction
  latent_interpreter.py      Latent feature labeling
  model_compatibility.py     Model architecture abstraction
  multi_model_dna.py         Multi-model DNA analysis
  causal_validation.py       Causal analysis tools
  tools.py                   RAG and fact-checking utilities
trained_probes/              Pre-trained probe files
probe_real_dataset/          Training datasets
```
