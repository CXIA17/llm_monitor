#!/usr/bin/env python3
"""
LLM DNA Extractor — 5-Step Pipeline
=====================================

Implements the LLM-DNA extraction pipeline from the paper:

1. **Prompt Sampling**: Draw t representative prompts from real-world datasets
   (SQuAD, CommonsenseQA, HellaSwag, MMLU mixture).

2. **Response Generation**: Feed each prompt into the target LLM to produce
   a textual response.

3. **Semantic Embedding**: Pass each response through a sentence-embedding model
   (default: Qwen/Qwen3-Embedding-8B) to get a fixed-size semantic vector.

4. **Concatenation**: Concatenate the t embedding vectors end-to-end into one
   high-dimensional vector.

5. **Random Gaussian Projection**: Multiply by a pre-computed random Gaussian
   matrix (Johnson–Lindenstrauss lemma) to project into a compact DNA vector
   (default: 128 dimensions).

The resulting DNA vector is comparable across different models — agents from
different model families can be placed into the same galaxy/phylogenetic tree.

Usage:
    from core.llm_dna_extractor import LLMDNAExtractor

    extractor = LLMDNAExtractor(
        embedding_model="Qwen/Qwen3-Embedding-8B",
        dna_dim=128,
        num_prompts=600,
    )

    # Extract DNA for a model
    dna = extractor.extract(model, tokenizer, device="cuda:0")
    # dna.vector is a (128,) numpy array

    # Compare two models
    dist = extractor.distance(dna_a, dna_b, metric="cosine")
"""

import json
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

import torch


# =============================================================================
# DNA RESULT
# =============================================================================

@dataclass
class LLMDNAResult:
    """Result of DNA extraction for one model (or agent)."""
    model_name: str
    vector: np.ndarray              # Final DNA vector (dna_dim,)
    dna_dim: int
    num_prompts: int
    embedding_model: str
    embedding_dim: int              # Per-response embedding dimension
    projection_seed: int
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "vector": self.vector.tolist(),
            "dna_dim": self.dna_dim,
            "num_prompts": self.num_prompts,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "projection_seed": self.projection_seed,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMDNAResult":
        return cls(
            model_name=d["model_name"],
            vector=np.array(d["vector"], dtype=np.float32),
            dna_dim=d["dna_dim"],
            num_prompts=d["num_prompts"],
            embedding_model=d["embedding_model"],
            embedding_dim=d["embedding_dim"],
            projection_seed=d["projection_seed"],
            timestamp=d.get("timestamp", ""),
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LLMDNAResult":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def distance_to(self, other: "LLMDNAResult", metric: str = "cosine") -> float:
        return compute_dna_distance(self.vector, other.vector, metric)


# =============================================================================
# DISTANCE
# =============================================================================

def compute_dna_distance(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> float:
    if metric == "cosine":
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))
    elif metric == "euclidean":
        return float(np.linalg.norm(a - b))
    elif metric == "nei":
        p1 = np.abs(a) + 1e-10
        p2 = np.abs(b) + 1e-10
        p1 /= p1.sum()
        p2 /= p2.sum()
        sim = np.sum(p1 * p2) / np.sqrt(np.sum(p1**2) * np.sum(p2**2))
        return float(-np.log(max(sim, 1e-15)))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# =============================================================================
# PROMPT SAMPLER  (Step 1)
# =============================================================================

# Default prompt pool — a balanced mix of representative tasks.
# In production, draw from SQuAD / CommonsenseQA / HellaSwag / MMLU.
# This built-in set is used as a fallback when no external dataset is provided.

_DEFAULT_PROMPTS = [
    # Reading comprehension (SQuAD-style)
    "What is the main cause of climate change according to scientific consensus?",
    "Summarize the key events that led to the fall of the Berlin Wall.",
    "Explain how photosynthesis works in plants.",
    "What were the primary factors behind the 2008 financial crisis?",
    "Describe the process of DNA replication.",
    # Commonsense reasoning
    "If you put a heavy object on thin ice, what is likely to happen?",
    "Why do people wear sunscreen at the beach?",
    "What would happen if all bees disappeared?",
    "Why do we need to sleep?",
    "What happens when you mix baking soda and vinegar?",
    # Sentence completion (HellaSwag-style)
    "The chef carefully placed the dough in the oven and then",
    "After studying for hours, the student closed her books and",
    "The mechanic examined the engine and noticed that",
    "Walking through the forest at dawn, they could hear",
    "The scientist reviewed the experimental data and concluded that",
    # Knowledge / MMLU-style
    "What is the difference between mitosis and meiosis?",
    "Explain the concept of supply and demand in economics.",
    "What is quantum entanglement?",
    "Describe the structure of the United States federal government.",
    "What are the main differences between TCP and UDP protocols?",
    # Ethical reasoning
    "Is it ethical to use AI for hiring decisions? Explain your reasoning.",
    "Should autonomous vehicles prioritize passenger safety over pedestrian safety?",
    "What are the ethical implications of genetic engineering in humans?",
    "Is universal basic income a good idea? Why or why not?",
    "Should social media companies be responsible for user-generated content?",
    # Creative / open-ended
    "Write a short paragraph about the future of space exploration.",
    "Describe what a day in the life of a deep-sea explorer might look like.",
    "What would the world look like if humans could photosynthesize?",
    "Imagine a city powered entirely by renewable energy. Describe it.",
    "What lessons can we learn from ancient civilizations?",
]


def load_prompts(
    source: Optional[str] = None,
    num_prompts: int = 30,
    seed: int = 42,
) -> List[str]:
    """
    Step 1: Load or sample representative prompts.

    Args:
        source: Path to a JSON or text file containing prompts.
                If None, uses the built-in default prompt pool.
        num_prompts: Number of prompts to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        List of prompt strings.
    """
    if source is not None:
        path = Path(source)
        if path.suffix == ".json":
            with open(path) as f:
                pool = json.load(f)
            if isinstance(pool, dict):
                # Assume {"prompts": [...]} or flatten values
                pool = pool.get("prompts", [v for v in pool.values() if isinstance(v, list)][0])
        else:
            with open(path) as f:
                pool = [line.strip() for line in f if line.strip()]
    else:
        pool = list(_DEFAULT_PROMPTS)

    rng = np.random.RandomState(seed)
    if num_prompts >= len(pool):
        return pool
    indices = rng.choice(len(pool), size=num_prompts, replace=False)
    return [pool[i] for i in indices]


# =============================================================================
# LLM DNA EXTRACTOR (Steps 2–5)
# =============================================================================

class LLMDNAExtractor:
    """
    Full 5-step DNA extraction pipeline.

    The projection matrix is deterministic given (projection_seed, input_dim,
    dna_dim), so two runs with the same seed produce identical projections.
    This is critical: distances between DNA vectors are only meaningful when
    they share the same projection matrix.
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-8B",
        dna_dim: int = 128,
        num_prompts: int = 30,
        prompt_source: Optional[str] = None,
        projection_seed: int = 42,
        prompt_seed: int = 42,
        max_new_tokens: int = 256,
        batch_size: int = 4,
        embedding_device: Optional[str] = None,
    ):
        """
        Args:
            embedding_model: HuggingFace model for sentence embedding.
            dna_dim: Final DNA vector dimensionality.
            num_prompts: Number of prompts to use (t in the paper).
            prompt_source: Path to external prompt file (JSON/txt).
            projection_seed: Seed for the Gaussian projection matrix.
            prompt_seed: Seed for prompt sampling.
            max_new_tokens: Max tokens per LLM response.
            batch_size: Batch size for embedding.
            embedding_device: Device for embedding model (defaults to target device).
        """
        self.embedding_model_name = embedding_model
        self.dna_dim = dna_dim
        self.num_prompts = num_prompts
        self.prompt_source = prompt_source
        self.projection_seed = projection_seed
        self.prompt_seed = prompt_seed
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.embedding_device = embedding_device

        # Lazy-loaded
        self._embedding_model = None
        self._embedding_tokenizer = None
        self._projection_matrix: Optional[np.ndarray] = None
        self._prompts: Optional[List[str]] = None

    @property
    def prompts(self) -> List[str]:
        if self._prompts is None:
            self._prompts = load_prompts(
                source=self.prompt_source,
                num_prompts=self.num_prompts,
                seed=self.prompt_seed,
            )
        return self._prompts

    # -----------------------------------------------------------------
    # Embedding model management
    # -----------------------------------------------------------------

    def _load_embedding_model(self, device: str):
        """Lazy-load the sentence embedding model."""
        if self._embedding_model is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        emb_device = self.embedding_device or device
        print(f"  Loading embedding model: {self.embedding_model_name} on {emb_device}")

        self._embedding_tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name, trust_remote_code=True
        )
        self._embedding_model = AutoModel.from_pretrained(
            self.embedding_model_name, trust_remote_code=True
        ).to(emb_device).eval()

    def _embed_texts(self, texts: List[str], device: str) -> np.ndarray:
        """
        Step 3: Compute semantic embeddings for a list of texts.

        Returns:
            (num_texts, embedding_dim) numpy array.
        """
        self._load_embedding_model(device)
        emb_device = self.embedding_device or device
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self._embedding_tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(emb_device)

            with torch.no_grad():
                outputs = self._embedding_model(**inputs)

            # Mean-pool over token dimension (ignore padding)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = outputs.last_hidden_state
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embeddings.append(pooled.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    # -----------------------------------------------------------------
    # Projection matrix (Step 5 setup)
    # -----------------------------------------------------------------

    def _get_projection_matrix(self, input_dim: int) -> np.ndarray:
        """
        Get or create the random Gaussian projection matrix.

        The matrix is (dna_dim, input_dim) and is drawn from N(0, 1/dna_dim).
        Per the Johnson–Lindenstrauss lemma, this preserves pairwise distances
        in expectation.
        """
        if self._projection_matrix is not None and self._projection_matrix.shape[1] == input_dim:
            return self._projection_matrix

        rng = np.random.RandomState(self.projection_seed)
        self._projection_matrix = rng.randn(self.dna_dim, input_dim).astype(np.float32)
        self._projection_matrix /= np.sqrt(self.dna_dim)
        return self._projection_matrix

    # -----------------------------------------------------------------
    # Main extraction
    # -----------------------------------------------------------------

    def extract(
        self,
        model,
        tokenizer,
        device: str = "cuda:0",
        model_name: str = "unknown",
        verbose: bool = True,
    ) -> LLMDNAResult:
        """
        Run the full 5-step DNA extraction pipeline.

        Args:
            model: HuggingFace causal LM (AutoModelForCausalLM).
            tokenizer: Corresponding tokenizer.
            device: Device the target model runs on.
            model_name: Human-readable model identifier.
            verbose: Print progress.

        Returns:
            LLMDNAResult with the final DNA vector.
        """
        prompts = self.prompts
        t = len(prompts)
        if verbose:
            print(f"\n  [DNA] Step 1: {t} prompts sampled")

        # --- Step 2: Generate responses ---
        responses = self._generate_responses(model, tokenizer, prompts, device, verbose)
        if verbose:
            avg_len = np.mean([len(r.split()) for r in responses])
            print(f"  [DNA] Step 2: Generated {len(responses)} responses (avg {avg_len:.0f} words)")

        # --- Step 3: Semantic embedding ---
        embeddings = self._embed_texts(responses, device)  # (t, emb_dim)
        embedding_dim = embeddings.shape[1]
        if verbose:
            print(f"  [DNA] Step 3: Embedded → ({t}, {embedding_dim})")

        # --- Step 4: Concatenation ---
        concatenated = embeddings.flatten()  # (t * emb_dim,)
        if verbose:
            print(f"  [DNA] Step 4: Concatenated → ({len(concatenated)},)")

        # --- Step 5: Random Gaussian projection ---
        proj = self._get_projection_matrix(len(concatenated))
        dna_vector = proj @ concatenated  # (dna_dim,)
        if verbose:
            print(f"  [DNA] Step 5: Projected → ({len(dna_vector)},)")

        return LLMDNAResult(
            model_name=model_name,
            vector=dna_vector.astype(np.float32),
            dna_dim=self.dna_dim,
            num_prompts=t,
            embedding_model=self.embedding_model_name,
            embedding_dim=embedding_dim,
            projection_seed=self.projection_seed,
            metadata={
                "max_new_tokens": self.max_new_tokens,
                "prompt_seed": self.prompt_seed,
                "avg_response_words": float(np.mean([len(r.split()) for r in responses])),
            },
        )

    def extract_from_responses(
        self,
        responses: List[str],
        device: str = "cuda:0",
        model_name: str = "unknown",
        verbose: bool = True,
    ) -> LLMDNAResult:
        """
        Extract DNA from pre-generated responses (skips Steps 1–2).

        Useful when you already have agent transcripts from an experiment.
        """
        t = len(responses)
        if verbose:
            print(f"  [DNA] Using {t} pre-generated responses")

        embeddings = self._embed_texts(responses, device)
        embedding_dim = embeddings.shape[1]
        if verbose:
            print(f"  [DNA] Embedded → ({t}, {embedding_dim})")

        concatenated = embeddings.flatten()
        proj = self._get_projection_matrix(len(concatenated))
        dna_vector = proj @ concatenated

        if verbose:
            print(f"  [DNA] Projected → ({len(dna_vector)},)")

        return LLMDNAResult(
            model_name=model_name,
            vector=dna_vector.astype(np.float32),
            dna_dim=self.dna_dim,
            num_prompts=t,
            embedding_model=self.embedding_model_name,
            embedding_dim=embedding_dim,
            projection_seed=self.projection_seed,
            metadata={
                "source": "pre_generated_responses",
            },
        )

    # -----------------------------------------------------------------
    # Response generation (Step 2)
    # -----------------------------------------------------------------

    def _generate_responses(
        self,
        model,
        tokenizer,
        prompts: List[str],
        device: str,
        verbose: bool = True,
    ) -> List[str]:
        """Step 2: Generate one response per prompt."""
        responses = []
        model.eval()

        for i, prompt in enumerate(prompts):
            if verbose and (i + 1) % 10 == 0:
                print(f"    Generating {i + 1}/{len(prompts)}...")

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Deterministic for reproducibility
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(response if response else "(empty)")

        return responses

    # -----------------------------------------------------------------
    # Batch extraction for multiple models
    # -----------------------------------------------------------------

    def extract_batch(
        self,
        models: Dict[str, Tuple],
        device: str = "cuda:0",
        verbose: bool = True,
    ) -> Dict[str, LLMDNAResult]:
        """
        Extract DNA for multiple models.

        Args:
            models: Dict of {name: (model, tokenizer)} pairs.
            device: Device for inference.

        Returns:
            Dict of {name: LLMDNAResult}.
        """
        results = {}
        for name, (model, tokenizer) in models.items():
            if verbose:
                print(f"\n{'='*50}")
                print(f"  Extracting DNA: {name}")
                print(f"{'='*50}")
            results[name] = self.extract(model, tokenizer, device, model_name=name, verbose=verbose)
        return results

    # -----------------------------------------------------------------
    # Distance matrix
    # -----------------------------------------------------------------

    @staticmethod
    def distance_matrix(
        results: Dict[str, LLMDNAResult],
        metric: str = "cosine",
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute pairwise distance matrix between DNA results."""
        names = list(results.keys())
        n = len(names)
        mat = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                d = compute_dna_distance(
                    results[names[i]].vector,
                    results[names[j]].vector,
                    metric,
                )
                mat[i, j] = d
                mat[j, i] = d

        return mat, names


# =============================================================================
# CONVENIENCE: Extract DNA for agents after an experiment
# =============================================================================

def extract_agent_dnas(
    experiment_result,
    embedding_model: str = "Qwen/Qwen3-Embedding-8B",
    dna_dim: int = 128,
    device: str = "cuda:0",
    projection_seed: int = 42,
) -> Dict[str, LLMDNAResult]:
    """
    Extract DNA for each agent from an ExperimentResult's transcript.

    Each agent's responses across all rounds are used as the input texts.
    """
    # Collect responses per agent from transcript
    agent_responses: Dict[str, List[str]] = {}
    for entry in experiment_result.transcript:
        agent_id = entry.get("agent_id", "")
        response = entry.get("response", "")
        if agent_id and response:
            agent_responses.setdefault(agent_id, []).append(response)

    if not agent_responses:
        return {}

    extractor = LLMDNAExtractor(
        embedding_model=embedding_model,
        dna_dim=dna_dim,
        projection_seed=projection_seed,
    )

    results = {}
    for agent_id, responses in agent_responses.items():
        results[agent_id] = extractor.extract_from_responses(
            responses=responses,
            device=device,
            model_name=f"{experiment_result.config.get('model_name', 'unknown')}_{agent_id}",
        )

    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LLMDNAResult",
    "LLMDNAExtractor",
    "compute_dna_distance",
    "load_prompts",
    "extract_agent_dnas",
]
