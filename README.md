# LLM Behavioral Monitor

A system for monitoring, steering, and analyzing behavioral patterns in multi-agent LLM interactions using linear probes and activation engineering.

## Overview

Large language models exhibit subtle behavioral patterns — sycophancy, overconfidence, toxicity, deception — that are invisible in the output text but detectable in the model's internal activations. This project lets you:

- **Detect** these behaviors in real time using linear probes trained on hidden-state activations
- **Steer** agent behavior by injecting learned direction vectors into the model's residual stream during generation
- **Compare** how different models and agent roles exhibit these behaviors across structured interactions
- **Extract LLM DNA** — compact behavioral fingerprints comparable across models using the 5-step pipeline (prompt sampling → response generation → semantic embedding → concatenation → Gaussian projection)
- **Visualize** behavioral signatures projected into a 2D galaxy for cross-experiment comparison

## Quick Start (from a fresh server)

```bash
# 1. Clone
git clone https://github.com/CXIA17/llm_monitor.git
cd llm_monitor

# 2. Create environment
conda create -n llm_monitor python=3.10 -y
conda activate llm_monitor

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run a basic experiment (model auto-downloads from HuggingFace)
python run_experiment.py \
    -q "Is AI dangerous?" \
    -t debate \
    -m Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda:0

# 5. Or launch the web dashboard
python launcher.py --model Qwen/Qwen2.5-0.5B-Instruct --device cuda:0
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/CXIA17/llm_monitor.git
cd llm_monitor
```

### 2. Create a virtual environment

```bash
# Using conda (recommended)
conda create -n llm_monitor python=3.10 -y
conda activate llm_monitor

# Or using venv
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For optional features:

```bash
# For downloading real probe-training datasets
pip install datasets pandas tqdm

# For downloading gated models (Llama, Gemma)
pip install huggingface_hub

# For 8-bit / 4-bit quantization
pip install bitsandbytes accelerate
```

### 4. Download models

You need at least one HuggingFace causal LM. Models are downloaded automatically by `transformers` (cached to `~/.cache/huggingface/`). No manual download is required unless you want to pre-download or use gated models.

**Option A: Auto-download (simplest)**

Models download automatically the first time you run an experiment. No extra steps needed:

```bash
python run_experiment.py -q "Is AI dangerous?" -m Qwen/Qwen2.5-0.5B-Instruct --device cuda:0
```

**Option B: Pre-download to a local directory**

If you want models stored in a specific directory (e.g., a shared NFS mount or fast SSD):

```bash
# Login to HuggingFace (required for gated models like Llama and Gemma)
huggingface-cli login

# Download with transformers CLI
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='./models')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='./models')
"
```

Then point experiments to your local directory with `--model-dir`:

```bash
python run_experiment.py -q "Is AI dangerous?" -m Qwen_Qwen2.5-0.5B-Instruct --model-dir ./models --device cuda:0
```

> **Note on `--model` naming with `--model-dir`**: When using `--model-dir`, the `--model` value is treated as a subdirectory name. Use underscores instead of slashes (e.g., `Qwen_Qwen3-4B` instead of `Qwen/Qwen3-4B`).

### 5. Train or obtain probes

Probes are linear classifiers trained on model hidden-state activations. You need probe files matching your model to use injection features.

**Option A: Use pre-trained probes**

Pre-trained probe metadata (`.json`) for several models is included in `trained_probes/`. The matching `.pkl` files (containing the actual probe weights) are gitignored due to size. To use injection, you need to either train your own probes or obtain the `.pkl` files.

Check available probe metadata:

```bash
python run_experiment.py --list-probes
```

**Option B: Train your own probes**

First, prepare training data:

```bash
# Download real datasets (sycophancy, toxicity, etc.)
python download_dataset.py

# Or generate synthetic datasets
python dataset_generator.py
```

Then train probes through the dashboard UI or via the `MultiProbeTrainer` API in `core/probe_trainer.py`.

## Usage

### CLI Experiment Runner (`run_experiment.py`)

The main CLI for running multi-agent interaction experiments. It supports configuring interaction type, model, injection type, and injection strength.

```bash
python run_experiment.py [options]
```

**List available options:**

```bash
python run_experiment.py --list-topologies   # Show interaction topologies
python run_experiment.py --list-probes       # Show available probes
python run_experiment.py --list-agents       # Show agent templates
```

**Basic debate (no injection):**

```bash
python run_experiment.py \
    -q "Is AI dangerous?" \
    -t debate \
    -m Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda:0 \
    -r 4
```

**Court trial with sycophancy injection:**

```bash
python run_experiment.py \
    -q "Should the defendant pay damages for breach of contract?" \
    -t court \
    -m Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda:0 \
    --probe-category sycophancy \
    --injection-type gated \
    --injection-strength 2.0 \
    --inject-agents plaintiff_attorney
```

**Adversarial with toxicity steering:**

```bash
python run_experiment.py \
    -q "Should social media be regulated?" \
    -t adversarial \
    -m Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda:0 \
    --probe-category toxicity \
    --injection-type steer \
    --injection-strength 3.0 \
    --injection-direction subtract \
    --inject-agents attacker_1
```

**Baseline vs injected comparison:**

```bash
python run_experiment.py \
    -q "Is capitalism ethical?" \
    -t debate \
    -m Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda:0 \
    --probe-category overconfidence \
    --injection-type gated \
    --injection-strength 2.5 \
    --inject-agents critic \
    --compare \
    -o results.json
```

**With LLM DNA extraction (5-step pipeline):**

```bash
python run_experiment.py \
    -q "Is AI dangerous?" \
    -t debate \
    -m Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda:0 \
    --extract-dna \
    --dna-output dna_results.json \
    --dna-dim 128 \
    --embedding-model Qwen/Qwen3-Embedding-8B
```

**Using a local model directory (e.g., pre-downloaded models):**

```bash
python run_experiment.py \
    -q "Is AI dangerous?" \
    -t debate \
    -m Qwen_Qwen3-4B \
    --model-dir /path/to/your/models \
    --device cuda:6
```

### CLI Reference

| Group | Argument | Default | Description |
|-------|----------|---------|-------------|
| **Experiment** | `-q, --question` | (required) | Topic for agents to discuss |
| | `-t, --topology` | `debate` | `debate`, `panel`, `adversarial`, `court`, `linear`, `round_robin`, `hub_spoke` |
| | `-r, --rounds` | `4` | Number of interaction rounds |
| | `--agents` | | Agent templates for custom topologies |
| **Model** | `-m, --model` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model name or local path |
| | `--model-dir` | `$MODEL_DIR` or `models` | Directory containing local models |
| | `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, etc. |
| | `--dtype` | | `float16`, `bfloat16`, `float32` |
| | `--load-in-8bit` | | Use 8-bit quantization |
| | `--load-in-4bit` | | Use 4-bit quantization |
| **Injection** | `--probe-category` | | `sycophancy`, `toxicity`, `overconfidence`, etc. |
| | `--probe-path` | (auto-detect) | Explicit path to probe `.pkl` file |
| | `--injection-type` | `gated` | `gated`, `amplify`, `ablate`, `steer` |
| | `--injection-strength` | `2.0` | Injection magnitude multiplier |
| | `--injection-direction` | `add` | `add` or `subtract` |
| | `--gate-bias` | `-0.3` | Threshold for gated injection |
| | `--inject-agents` | | Agent IDs to inject (e.g., `critic proposer`) |
| **DNA** | `--extract-dna` | | Extract LLM DNA after experiment |
| | `--dna-output` | | Save DNA vectors to JSON |
| | `--dna-dim` | `128` | DNA vector dimensionality |
| | `--embedding-model` | `Qwen/Qwen3-Embedding-8B` | Sentence embedding model |
| | `--dna-projection-seed` | `42` | Seed for Gaussian projection matrix |
| **Output** | `-o, --output` | | Save experiment results to JSON |
| | `--compare` | | Run baseline vs injected comparison |
| | `--quiet` | | Minimal output |

### Web Dashboards (`launcher.py`)

The launcher serves a landing page with links to both dashboards on a single port.

**Using HuggingFace model name (auto-downloads):**

```bash
python launcher.py --model Qwen/Qwen2.5-0.5B-Instruct --device cuda:0
```

**Using a local model directory:**

```bash
python launcher.py --model Qwen_Qwen3-4B --model-dir /path/to/your/models --device cuda:6
```

Access at:
- `http://localhost:8000/` — Landing page
- `http://localhost:8000/court/` — Federal Court Simulation
- `http://localhost:8000/dashboard/` — Multi-Agent Monitor

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen_Qwen2.5-0.5B-Instruct` | HuggingFace model name or directory name under `--model-dir` |
| `--model-dir` | `$MODEL_DIR` or `models` | Directory containing local models |
| `--device` | `cuda:0` | Device for inference |
| `--port` | `8000` | Server port |
| `--host` | `0.0.0.0` | Server host |
| `--no-court` | | Skip court dashboard |
| `--no-dashboard` | | Skip multi-agent dashboard |

### Docker

```bash
# Build
docker build -t llm-monitor .

# Run with GPU and mounted model directory
docker run --gpus all -p 8000:8000 \
  -v /path/to/your/models:/models \
  llm-monitor --model Qwen_Qwen3-4B --device cuda:0

# Docker Compose (set MODEL_DIR to your local model directory)
MODEL_DIR=/path/to/your/models docker compose up
```

## Environment Variables

| Variable | Description | Used by |
|----------|-------------|---------|
| `MODEL_DIR` | Directory containing local model weights | All scripts |
| `HF_TOKEN` | HuggingFace API token (for gated models) | `huggingface-cli login` |
| `DASHBOARD_MODEL` | Override model for dashboard | `dashboard_server.py` |
| `DASHBOARD_DEVICE` | Override device for dashboard | `dashboard_server.py` |
| `COURT_MODEL` | Override model for court dashboard | `court_dashboard.py` |
| `COURT_DEVICE` | Override device for court dashboard | `court_dashboard.py` |

## Interaction Topologies

| Topology | Agents | Pattern |
|----------|--------|---------|
| `debate` | proposer, critic, judge | Proposer ↔ Critic → Judge |
| `panel` | moderator, expert_1, expert_2 | Moderator broadcasts, experts respond |
| `adversarial` | defender, attacker, evaluator | Defender ↔ Attacker → Evaluator |
| `court` | plaintiff_attorney, defense_attorney, court_judge | Plaintiff → Defense → Judge |
| `linear` | custom (via `--agents`) | A → B → C sequential |
| `round_robin` | custom (via `--agents`) | A → B → C → A cyclic |
| `hub_spoke` | custom (via `--agents`) | Hub ↔ Spoke1, Hub ↔ Spoke2 |

## Injection Types

| Type | Behavior |
|------|----------|
| `gated` | Conditional: only injects when probe score drifts past threshold |
| `amplify` | Direct amplification of activation dimensions |
| `ablate` | Suppression/ablation of activations |
| `steer` | Multi-layer steering with dynamic norm-based scaling |

## LLM DNA Extraction

The 5-step pipeline from the LLM-DNA paper, implemented in `core/llm_dna_extractor.py`:

1. **Prompt Sampling** — Draw *t* representative prompts from real-world datasets (SQuAD, CommonsenseQA, HellaSwag, MMLU)
2. **Response Generation** — Feed each prompt into the target LLM to generate a textual response
3. **Semantic Embedding** — Pass each response through a sentence-embedding model (default: `Qwen/Qwen3-Embedding-8B`) to get a fixed-size semantic vector
4. **Concatenation** — Concatenate all *t* embedding vectors end-to-end into one high-dimensional vector
5. **Random Gaussian Projection** — Multiply by a pre-computed random Gaussian matrix (Johnson-Lindenstrauss lemma) to project into a compact DNA vector (default: 128D)

The resulting DNA vectors are comparable across different models — agents from different model families can be placed into the same galaxy/phylogenetic tree, because the representation is based on output text semantics (not internal activations).

## Probe System

### What Probes Are

Linear classifiers trained on model hidden-state activations. A probe learns a direction vector in activation space that separates two behavioral classes (e.g., sycophantic vs. calibrated responses). During generation, the probe scores each token by projecting the current activation onto this direction.

### Available Probe Categories

Overconfidence, Uncertainty, Hedging, Sycophancy, Toxicity, Formality, Emotional Valence, Deception, Safety, Refusal, Helpfulness, Specificity

### How Injection Works

When a probe is injected into an agent, three mechanisms activate during generation:

1. **Multi-Layer Injection** — The probe's direction vector is injected across multiple transformer layers (from layer N/4 to N-1), with per-layer strength scaled by `strength / n_layers^0.65`.

2. **Dynamic Scaling** — The perturbation scales proportionally to the current residual-stream norm: `delta = strength * sign * stream_norm * 0.1`.

3. **Gated Injection** — Injection only fires when the model is drifting from the desired behavior. The gate threshold is controlled by `--gate-bias` (default: -0.3).

## Supported Models

21+ model families: Qwen/Qwen-2/Qwen-3, Llama/Llama-2/Llama-3, Mistral/Mixtral, Phi-1/2/3, Gemma/Gemma-2, GPT-2/GPT-Neo/GPT-J, Falcon, OLMo, StableLM, Bloom, OPT, Pythia, and more. The `ModelCompatibility` layer auto-detects architecture and adapts layer access patterns.

## Project Structure

```
run_experiment.py              CLI experiment runner (main entry point)
launcher.py                    Unified web dashboard server
dashboard_server.py            Multi-agent dashboard
court_dashboard.py             Court simulation dashboard
behavioural_dna.py             128D behavioral DNA extraction (probe-based)
dna_extractor.py               DNA extraction utilities
cross_model_galaxy.py          Cross-model comparison visualization
dataset_generator.py           Synthetic dataset generation
download_dataset.py            Real dataset downloading
requirements.txt               Python dependencies
Dockerfile                     Docker image definition
docker-compose.yml             Docker Compose config

core/
  orchestrator.py              Multi-agent orchestration engine
  agent_registry.py            Agent templates and configuration
  interaction_graph.py         Topology definitions (7 topologies)
  model_compatibility.py       Model architecture abstraction (21+ families)
  steered_agent.py             Probe-based steering (dynamic scaling, gating)
  probe_trainer.py             Probe training pipeline
  llm_dna_extractor.py         5-step LLM DNA pipeline (paper implementation)
  llm_dna.py                   DNA signatures and phylogenetic trees
  multi_model_dna.py           Cross-model DNA normalization
  sae_fingerprint.py           SAE feature extraction
  latent_interpreter.py        Latent feature labeling
  court_orchestrator.py        Court simulation runner
  causal_validation.py         Causal analysis tools
  tools.py                     RAG and fact-checking utilities

trained_probes/                Pre-trained probe metadata (.json) and weights (.pkl, gitignored)
probe_real_dataset/            Training datasets (gitignored, download via download_dataset.py)
```

## Troubleshooting

**"No probe file found for model X"**
- Run `python run_experiment.py --list-probes` to see available probes
- Use `--probe-path` to point to a specific `.pkl` file
- Train a new probe via the dashboard or `MultiProbeTrainer`

**CUDA out of memory**
- Use `--load-in-4bit` or `--load-in-8bit` for quantization
- Use a smaller model (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)
- Reduce `--rounds` to generate less data

**Gated model access denied (Llama, Gemma)**
- Accept the model license on HuggingFace
- Run `huggingface-cli login` or set `HF_TOKEN` environment variable

**DNA extraction requires embedding model**
- The default embedding model (`Qwen/Qwen3-Embedding-8B`) will be auto-downloaded
- Use `--embedding-model` to specify a different/smaller model
- Ensure you have enough GPU memory to load both the target model and embedding model

**Model download is slow or fails**
- Check your internet connection
- For large models, consider pre-downloading: `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct`
- Set `HF_HUB_CACHE` to change the download cache location
