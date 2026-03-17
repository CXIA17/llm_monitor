#!/usr/bin/env python3
"""
SAE-Based Functional Model Fingerprinting
==========================================

Implements a Sparse Autoencoder (SAE) based pipeline for generating
interpretable, functional fingerprints of LLMs. Instead of analyzing
individual text outputs, this module generates "functional model fingerprints"
for the LLMs themselves.

Pipeline (5 Steps):
    1. Data Generation: Collect LLM responses to evaluation prompts
    2. SAE Feature Extraction: Feed responses through a frozen "reader LLM" + SAE
    3. Document-Level Binarization: Max-pool SAE activations → binary vectors
    4. Aggregation: Compute frequency distributions → Model Fingerprint
    5. Dataset Diffing: Compare fingerprints to identify distinguishing features

Architecture:

    Target LLM A          Target LLM B          Target LLM C
    ┌────────────┐        ┌────────────┐        ┌────────────┐
    │ Generate   │        │ Generate   │        │ Generate   │
    │ responses  │        │ responses  │        │ responses  │
    └─────┬──────┘        └─────┬──────┘        └─────┬──────┘
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                │
                                ▼
                  ┌──────────────────────────┐
                  │  Reader LLM + SAE        │
                  │  (Frozen, e.g. Llama-70B)│
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  Per-Token SAE Activations│
                  │  (sparse, d_SAE dims)    │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  Max-Pool + Binarize     │
                  │  → Binary doc vector     │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  Aggregate across N      │
                  │  responses → Frequency   │
                  │  Distribution (Fingerprint)│
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  Dataset Diffing         │
                  │  (Model A vs Model B)    │
                  └──────────────────────────┘

Usage:
    from core.sae_fingerprint import (
        SAEFeatureExtractor,
        SAEModelFingerprint,
        SAEFingerprintAnalyzer,
    )

    # Step 1-2: Extract SAE features from responses
    extractor = SAEFeatureExtractor(
        reader_model_name="meta-llama/Llama-3.3-70B",
        sae_path="/path/to/pretrained_sae.pt",
        layer_idx=24,
    )

    # Step 3: Binarize each response
    binary_vectors = extractor.extract_binary_vectors(responses)

    # Step 4: Aggregate into fingerprint
    fingerprint = SAEModelFingerprint.from_binary_vectors(
        model_id="GPT-5",
        binary_vectors=binary_vectors,
        feature_labels=extractor.get_feature_labels(),
    )

    # Step 5: Diff fingerprints
    analyzer = SAEFingerprintAnalyzer()
    analyzer.add_fingerprint(fingerprint_a)
    analyzer.add_fingerprint(fingerprint_b)
    diff = analyzer.diff_fingerprints("GPT-5", "Grok-4")
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional imports for model loading
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.spatial.distance import cosine as cosine_distance
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_BINARIZATION_THRESHOLD = 0.0  # Features with max-pooled activation > 0
DEFAULT_TOP_K_FEATURES = 20  # Top features for diffing reports
DEFAULT_SAE_DIM = 65536  # Typical SAE dictionary size


# =============================================================================
# SPARSE AUTOENCODER WRAPPER
# =============================================================================

class SparseAutoencoder(nn.Module if HAS_TORCH else object):
    """
    Minimal Sparse Autoencoder for feature extraction.

    This wraps a pretrained SAE that decomposes LLM activations into
    interpretable sparse features. Each latent dimension corresponds
    to a learned semantic concept.

    The SAE operates on residual stream activations from a specific
    layer of the reader LLM:
        h -> encoder -> sparse code z -> decoder -> h_reconstructed

    We only use the encoder path to get the sparse feature activations.
    """

    def __init__(self, d_model: int, d_sae: int):
        if HAS_TORCH:
            super().__init__()
            self.d_model = d_model
            self.d_sae = d_sae
            self.encoder = nn.Linear(d_model, d_sae)
            self.encoder_bias = nn.Parameter(torch.zeros(d_sae))
            self.pre_bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.d_model = d_model
            self.d_sae = d_sae

    def encode(self, x):
        """
        Encode activations into sparse features.

        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Sparse activations of shape (batch, seq_len, d_sae)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for SAE encoding")

        # Subtract pre-bias (learned centering)
        x_centered = x - self.pre_bias
        # Linear projection + ReLU for sparsity
        z = torch.relu(self.encoder(x_centered) + self.encoder_bias)
        return z

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> 'SparseAutoencoder':
        """Load a pretrained SAE from disk."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required to load SAE")

        state = torch.load(path, map_location=device, weights_only=True)

        # Support multiple save formats
        if "cfg" in state:
            d_model = state["cfg"].get("d_in", state["cfg"].get("d_model", 4096))
            d_sae = state["cfg"].get("d_sae", state["cfg"].get("d_hidden", 65536))
        elif "d_model" in state:
            d_model = state["d_model"]
            d_sae = state["d_sae"]
        else:
            # Infer from weight shapes
            for key in state:
                if "encoder" in key and "weight" in key:
                    d_sae, d_model = state[key].shape
                    break
            else:
                raise ValueError("Cannot infer SAE dimensions from checkpoint")

        sae = cls(d_model, d_sae)

        # Load weights (handle different key naming conventions)
        model_state = state.get("state_dict", state)
        compatible_state = {}
        for k, v in model_state.items():
            # Normalize key names
            clean_key = k.replace("W_enc", "encoder.weight") \
                         .replace("b_enc", "encoder_bias") \
                         .replace("b_pre", "pre_bias")
            compatible_state[clean_key] = v

        sae.load_state_dict(compatible_state, strict=False)
        sae.eval()
        sae.to(device)
        return sae


# =============================================================================
# SAE FEATURE EXTRACTOR (Steps 2-3)
# =============================================================================

class SAEFeatureExtractor:
    """
    Extracts SAE features from text responses using a frozen reader LLM.

    This is the "Reader Model" phase: we feed every generated text response
    into a single, frozen reader LLM equipped with a pretrained SAE.
    The SAE labels each token's activations with thousands of semantic
    properties simultaneously.

    Supports two modes:
    1. Live mode: Load reader model + SAE and extract features in real-time
    2. Offline mode: Load pre-extracted SAE activations from disk
    """

    def __init__(
        self,
        reader_model_name: Optional[str] = None,
        sae_path: Optional[str] = None,
        layer_idx: int = 24,
        device: str = "auto",
        d_sae: int = DEFAULT_SAE_DIM,
        binarization_threshold: float = DEFAULT_BINARIZATION_THRESHOLD,
        feature_labels: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize SAE Feature Extractor.

        Args:
            reader_model_name: HuggingFace model name for the reader LLM
                (e.g., "meta-llama/Llama-3.3-70B-Instruct")
            sae_path: Path to pretrained SAE checkpoint
            layer_idx: Which layer to hook into for activations
            device: Device for inference ("auto", "cpu", "cuda", "cuda:0", etc.)
            d_sae: SAE dictionary size (number of latent features)
            binarization_threshold: Threshold for binarizing max-pooled activations
            feature_labels: Optional dict mapping latent index → human-readable label
        """
        self.reader_model_name = reader_model_name
        self.sae_path = sae_path
        self.layer_idx = layer_idx
        self.d_sae = d_sae
        self.binarization_threshold = binarization_threshold
        self.feature_labels = feature_labels or {}

        # Resolve device
        if device == "auto":
            self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model and SAE (lazily loaded)
        self._reader_model = None
        self._tokenizer = None
        self._sae = None
        self._activation_hook = None
        self._cached_activations = None

    def _ensure_models_loaded(self):
        """Lazily load reader model and SAE."""
        if self._reader_model is not None:
            return

        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is required for live SAE feature extraction. "
                "Install with: pip install torch transformers"
            )

        if not self.reader_model_name or not self.sae_path:
            raise ValueError(
                "reader_model_name and sae_path are required for live extraction. "
                "Use from_precomputed() for offline mode."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading reader model: {self.reader_model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.reader_model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._reader_model = AutoModelForCausalLM.from_pretrained(
            self.reader_model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
        )
        self._reader_model.eval()

        logger.info(f"Loading SAE from: {self.sae_path}")
        self._sae = SparseAutoencoder.from_pretrained(self.sae_path, device=self.device)
        self.d_sae = self._sae.d_sae

        # Register forward hook to capture activations
        self._register_activation_hook()

    def _register_activation_hook(self):
        """Register a forward hook on the target layer to capture activations."""
        target_layer = self._reader_model.model.layers[self.layer_idx]

        def hook_fn(module, input, output):
            # output is typically a tuple; first element is the hidden state
            if isinstance(output, tuple):
                self._cached_activations = output[0].detach()
            else:
                self._cached_activations = output.detach()

        self._activation_hook = target_layer.register_forward_hook(hook_fn)

    def extract_sae_activations(
        self,
        text: str,
        max_tokens: int = 512,
    ) -> np.ndarray:
        """
        Extract per-token SAE activations for a single text response.

        Args:
            text: The text response to analyze
            max_tokens: Maximum tokens to process

        Returns:
            SAE activations array of shape (seq_len, d_sae)
        """
        self._ensure_models_loaded()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=max_tokens,
            truncation=True,
            padding=False,
        ).to(self.device)

        with torch.no_grad():
            self._reader_model(**inputs)

        # _cached_activations: (1, seq_len, d_model)
        activations = self._cached_activations[0]  # (seq_len, d_model)

        # Pass through SAE encoder
        sae_features = self._sae.encode(activations.unsqueeze(0))  # (1, seq_len, d_sae)
        sae_features = sae_features[0]  # (seq_len, d_sae)

        return sae_features.cpu().numpy()

    def extract_binary_vector(
        self,
        text: str,
        max_tokens: int = 512,
    ) -> np.ndarray:
        """
        Extract a binarized document-level SAE feature vector for a single response.

        Step 3: Document-Level Binarization
        - Max-pool SAE activations across all tokens
        - Binarize: 1 if feature activation > threshold, 0 otherwise

        Args:
            text: The text response to analyze
            max_tokens: Maximum tokens to process

        Returns:
            Binary vector of shape (d_sae,) with 0/1 values
        """
        sae_activations = self.extract_sae_activations(text, max_tokens)

        # Max-pool across tokens: (seq_len, d_sae) -> (d_sae,)
        max_pooled = np.max(sae_activations, axis=0)

        # Binarize
        binary = (max_pooled > self.binarization_threshold).astype(np.float32)

        return binary

    def extract_binary_vectors(
        self,
        texts: List[str],
        max_tokens: int = 512,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract binarized document-level vectors for multiple responses.

        Args:
            texts: List of text responses
            max_tokens: Maximum tokens per response
            batch_size: Processing batch size
            show_progress: Whether to log progress

        Returns:
            Binary matrix of shape (n_texts, d_sae)
        """
        binary_vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                bv = self.extract_binary_vector(text, max_tokens)
                binary_vectors.append(bv)

            if show_progress and (i + batch_size) % (batch_size * 10) == 0:
                logger.info(
                    f"SAE extraction progress: {min(i + batch_size, len(texts))}/{len(texts)}"
                )

        return np.array(binary_vectors)

    def extract_continuous_vectors(
        self,
        texts: List[str],
        max_tokens: int = 512,
    ) -> np.ndarray:
        """
        Extract continuous (non-binarized) max-pooled SAE vectors.

        Useful when you want to preserve activation magnitudes
        rather than just presence/absence.

        Args:
            texts: List of text responses
            max_tokens: Maximum tokens per response

        Returns:
            Matrix of shape (n_texts, d_sae) with continuous values
        """
        vectors = []
        for text in texts:
            sae_activations = self.extract_sae_activations(text, max_tokens)
            max_pooled = np.max(sae_activations, axis=0)
            vectors.append(max_pooled)
        return np.array(vectors)

    @classmethod
    def from_precomputed(
        cls,
        d_sae: int = DEFAULT_SAE_DIM,
        binarization_threshold: float = DEFAULT_BINARIZATION_THRESHOLD,
        feature_labels: Optional[Dict[int, str]] = None,
    ) -> 'SAEFeatureExtractor':
        """
        Create an extractor for offline mode (pre-extracted activations).

        Use this when you have already extracted SAE activations and saved
        them to disk, and just need binarization + aggregation.
        """
        return cls(
            reader_model_name=None,
            sae_path=None,
            d_sae=d_sae,
            binarization_threshold=binarization_threshold,
            feature_labels=feature_labels,
        )

    def binarize_precomputed(
        self,
        sae_activations: np.ndarray,
    ) -> np.ndarray:
        """
        Binarize pre-extracted SAE activations.

        Args:
            sae_activations: Array of shape (n_tokens, d_sae) for one document,
                or (n_docs, n_tokens, d_sae) for multiple documents.

        Returns:
            Binary vectors of shape (d_sae,) or (n_docs, d_sae)
        """
        if sae_activations.ndim == 2:
            # Single document: (n_tokens, d_sae) -> (d_sae,)
            max_pooled = np.max(sae_activations, axis=0)
            return (max_pooled > self.binarization_threshold).astype(np.float32)
        elif sae_activations.ndim == 3:
            # Multiple documents: (n_docs, n_tokens, d_sae) -> (n_docs, d_sae)
            max_pooled = np.max(sae_activations, axis=1)
            return (max_pooled > self.binarization_threshold).astype(np.float32)
        else:
            raise ValueError(
                f"Expected 2D or 3D array, got shape {sae_activations.shape}"
            )

    def get_feature_labels(self) -> Dict[int, str]:
        """Return the feature label mapping."""
        return self.feature_labels

    def cleanup(self):
        """Release model resources."""
        if self._activation_hook is not None:
            self._activation_hook.remove()
            self._activation_hook = None
        self._reader_model = None
        self._tokenizer = None
        self._sae = None
        self._cached_activations = None
        if HAS_TORCH:
            torch.cuda.empty_cache()


# =============================================================================
# SAE MODEL FINGERPRINT (Step 4)
# =============================================================================

@dataclass
class SAEModelFingerprint:
    """
    Functional model fingerprint based on SAE latent frequencies.

    This represents a model's behavioral DNA as a frequency distribution
    over SAE latent features, computed by aggregating binarized feature
    vectors across many responses.

    Each dimension represents the percentage of responses where a specific
    SAE-learned semantic concept was activated.

    Attributes:
        model_id: Identifier for the target LLM
        frequencies: Array of shape (d_sae,) with activation frequencies [0, 1]
        n_responses: Number of responses aggregated
        d_sae: SAE dictionary size
        feature_labels: Mapping of latent index → human-readable description
        metadata: Additional metadata (prompts used, timestamp, etc.)
    """
    model_id: str
    frequencies: np.ndarray  # shape (d_sae,), values in [0, 1]
    n_responses: int
    d_sae: int
    feature_labels: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_binary_vectors(
        cls,
        model_id: str,
        binary_vectors: np.ndarray,
        feature_labels: Optional[Dict[int, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'SAEModelFingerprint':
        """
        Step 4: Aggregate binary vectors into a model fingerprint.

        For each SAE latent, compute the percentage of responses
        where that feature was activated.

        Args:
            model_id: Target model identifier
            binary_vectors: Array of shape (n_responses, d_sae) with 0/1 values
            feature_labels: Optional feature label mapping
            metadata: Optional metadata dict

        Returns:
            SAEModelFingerprint with frequency distribution
        """
        n_responses, d_sae = binary_vectors.shape
        frequencies = np.mean(binary_vectors, axis=0)  # Fraction of responses

        return cls(
            model_id=model_id,
            frequencies=frequencies,
            n_responses=n_responses,
            d_sae=d_sae,
            feature_labels=feature_labels or {},
            metadata=metadata or {
                "created_at": datetime.now().isoformat(),
                "n_responses": n_responses,
            },
        )

    @classmethod
    def from_continuous_vectors(
        cls,
        model_id: str,
        continuous_vectors: np.ndarray,
        feature_labels: Optional[Dict[int, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'SAEModelFingerprint':
        """
        Create fingerprint from continuous (non-binarized) vectors.

        Uses the mean activation strength as the frequency measure.
        """
        n_responses, d_sae = continuous_vectors.shape
        frequencies = np.mean(continuous_vectors, axis=0)

        # Normalize to [0, 1] range
        max_val = np.max(frequencies)
        if max_val > 0:
            frequencies = frequencies / max_val

        return cls(
            model_id=model_id,
            frequencies=frequencies,
            n_responses=n_responses,
            d_sae=d_sae,
            feature_labels=feature_labels or {},
            metadata=metadata or {
                "created_at": datetime.now().isoformat(),
                "n_responses": n_responses,
                "mode": "continuous",
            },
        )

    def get_active_features(
        self,
        min_frequency: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """
        Get features that are active (above minimum frequency) in this model.

        Returns:
            List of {index, frequency, label} sorted by frequency descending
        """
        active = []
        for idx in range(self.d_sae):
            freq = float(self.frequencies[idx])
            if freq >= min_frequency:
                active.append({
                    "index": idx,
                    "frequency": freq,
                    "frequency_pct": freq * 100,
                    "label": self.feature_labels.get(idx, f"SAE Latent #{idx}"),
                })
        active.sort(key=lambda x: x["frequency"], reverse=True)
        return active

    def get_top_features(self, top_n: int = DEFAULT_TOP_K_FEATURES) -> List[Dict[str, Any]]:
        """Get the top-N most frequently activated features."""
        return self.get_active_features()[:top_n]

    def to_vector(self) -> np.ndarray:
        """Return the frequency distribution as a numpy vector."""
        return self.frequencies.copy()

    def cosine_similarity(self, other: 'SAEModelFingerprint') -> float:
        """Compute cosine similarity with another fingerprint."""
        dot = np.dot(self.frequencies, other.frequencies)
        norm_a = np.linalg.norm(self.frequencies)
        norm_b = np.linalg.norm(other.frequencies)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def save(self, filepath: Union[str, Path]):
        """Save fingerprint to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model_id": self.model_id,
            "frequencies": self.frequencies.tolist(),
            "n_responses": self.n_responses,
            "d_sae": self.d_sae,
            "feature_labels": {str(k): v for k, v in self.feature_labels.items()},
            "metadata": self.metadata,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SAEModelFingerprint':
        """Load fingerprint from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            model_id=data["model_id"],
            frequencies=np.array(data["frequencies"], dtype=np.float32),
            n_responses=data["n_responses"],
            d_sae=data["d_sae"],
            feature_labels={int(k): v for k, v in data.get("feature_labels", {}).items()},
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# FINGERPRINT DIFF RESULT
# =============================================================================

@dataclass
class FingerprintDiff:
    """
    Result of diffing two model fingerprints.

    Contains the signed frequency differences for all SAE latents,
    plus ranked lists of the most distinguishing features.
    """
    model_a: str
    model_b: str
    diff_vector: np.ndarray  # frequencies_b - frequencies_a, shape (d_sae,)
    d_sae: int

    # Top features that distinguish model_b from model_a
    top_positive: List[Dict[str, Any]] = field(default_factory=list)  # Higher in B
    top_negative: List[Dict[str, Any]] = field(default_factory=list)  # Higher in A

    # Summary statistics
    mean_absolute_diff: float = 0.0
    max_positive_diff: float = 0.0
    max_negative_diff: float = 0.0
    n_significant_diffs: int = 0

    def get_report(self, top_n: int = 10) -> str:
        """Generate a human-readable diff report."""
        lines = [
            f"=== Functional Model Fingerprint Comparison ===",
            f"Target Model: {self.model_b}",
            f"Comparison Model: {self.model_a}",
            f"Analysis Type: Dataset Diffing (Aggregated SAE Latent Frequencies)",
            f"",
            f"Summary:",
            f"  Mean |diff|: {self.mean_absolute_diff:.4f}",
            f"  Significant features: {self.n_significant_diffs}",
            f"",
            f"Top Identifying Semantic Features ({self.model_b} vs {self.model_a}):",
            f"",
        ]

        for i, feat in enumerate(self.top_positive[:top_n], 1):
            lines.append(
                f"{i}. Feature: \"{feat['label']}\"  "
                f"(Frequency Diff: +{feat['diff_pct']:.1f}%)"
            )
            if feat.get("description"):
                lines.append(f"   SAE Description: {feat['description']}")
            lines.append("")

        if self.top_negative:
            lines.append(f"Features more prominent in {self.model_a}:")
            lines.append("")
            for i, feat in enumerate(self.top_negative[:top_n], 1):
                lines.append(
                    f"{i}. Feature: \"{feat['label']}\"  "
                    f"(Frequency Diff: {feat['diff_pct']:.1f}%)"
                )
                lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "top_positive": self.top_positive,
            "top_negative": self.top_negative,
            "mean_absolute_diff": self.mean_absolute_diff,
            "max_positive_diff": self.max_positive_diff,
            "max_negative_diff": self.max_negative_diff,
            "n_significant_diffs": self.n_significant_diffs,
        }


# =============================================================================
# SAE FINGERPRINT ANALYZER (Step 5)
# =============================================================================

class SAEFingerprintAnalyzer:
    """
    Analyzes and compares SAE-based model fingerprints.

    Implements Step 5 (Dataset Diffing) and provides visualization
    and comparison tools for model fingerprints.

    Usage:
        analyzer = SAEFingerprintAnalyzer()
        analyzer.add_fingerprint(fp_gpt5)
        analyzer.add_fingerprint(fp_grok4)
        analyzer.add_fingerprint(fp_gemini)

        # Pairwise diff
        diff = analyzer.diff_fingerprints("Grok-4", "GPT-5")
        print(diff.get_report())

        # One-vs-rest
        unique = analyzer.one_vs_rest("Grok-4")

        # Distance matrix
        distances, labels = analyzer.compute_distance_matrix()
    """

    def __init__(self):
        self.fingerprints: Dict[str, SAEModelFingerprint] = {}

    def add_fingerprint(self, fingerprint: SAEModelFingerprint):
        """Add a model fingerprint to the analyzer."""
        self.fingerprints[fingerprint.model_id] = fingerprint

    def remove_fingerprint(self, model_id: str):
        """Remove a model fingerprint from the analyzer."""
        self.fingerprints.pop(model_id, None)

    def list_models(self) -> List[str]:
        """List all loaded model IDs."""
        return list(self.fingerprints.keys())

    # -----------------------------------------------------------------
    # Step 5: Dataset Diffing
    # -----------------------------------------------------------------

    def diff_fingerprints(
        self,
        model_a: str,
        model_b: str,
        significance_threshold: float = 0.05,
        top_n: int = DEFAULT_TOP_K_FEATURES,
    ) -> FingerprintDiff:
        """
        Step 5: Dataset Diffing for Semantic Labels.

        Subtract the latent frequencies of model_a's fingerprint from
        model_b's. The latents with the highest positive or negative
        frequency differences reveal the exact semantic behaviors that
        distinguish the models.

        Args:
            model_a: First model ID (baseline/reference)
            model_b: Second model ID (target)
            significance_threshold: Minimum |diff| to be considered significant
            top_n: Number of top features to include

        Returns:
            FingerprintDiff with ranked distinguishing features
        """
        if model_a not in self.fingerprints:
            raise KeyError(f"Model '{model_a}' not found. Available: {self.list_models()}")
        if model_b not in self.fingerprints:
            raise KeyError(f"Model '{model_b}' not found. Available: {self.list_models()}")

        fp_a = self.fingerprints[model_a]
        fp_b = self.fingerprints[model_b]

        # Signed difference: positive means higher in model_b
        diff_vector = fp_b.frequencies - fp_a.frequencies

        # Merge feature labels from both fingerprints
        all_labels = {**fp_a.feature_labels, **fp_b.feature_labels}

        # Rank by absolute difference
        sorted_indices = np.argsort(np.abs(diff_vector))[::-1]

        top_positive = []
        top_negative = []

        for idx in sorted_indices:
            idx = int(idx)
            diff_val = float(diff_vector[idx])
            if abs(diff_val) < significance_threshold / 100:  # Convert from pct
                continue

            entry = {
                "index": idx,
                "label": all_labels.get(idx, f"SAE Latent #{idx}"),
                "description": all_labels.get(idx, ""),
                "diff": diff_val,
                "diff_pct": diff_val * 100,
                "freq_a": float(fp_a.frequencies[idx]),
                "freq_a_pct": float(fp_a.frequencies[idx]) * 100,
                "freq_b": float(fp_b.frequencies[idx]),
                "freq_b_pct": float(fp_b.frequencies[idx]) * 100,
            }

            if diff_val > 0 and len(top_positive) < top_n:
                top_positive.append(entry)
            elif diff_val < 0 and len(top_negative) < top_n:
                top_negative.append(entry)

            if len(top_positive) >= top_n and len(top_negative) >= top_n:
                break

        # Statistics
        abs_diffs = np.abs(diff_vector)
        n_significant = int(np.sum(abs_diffs > significance_threshold / 100))

        return FingerprintDiff(
            model_a=model_a,
            model_b=model_b,
            diff_vector=diff_vector,
            d_sae=len(diff_vector),
            top_positive=top_positive,
            top_negative=top_negative,
            mean_absolute_diff=float(np.mean(abs_diffs)),
            max_positive_diff=float(np.max(diff_vector)),
            max_negative_diff=float(np.min(diff_vector)),
            n_significant_diffs=n_significant,
        )

    def one_vs_rest(
        self,
        target_model: str,
        significance_threshold: float = 0.05,
        top_n: int = DEFAULT_TOP_K_FEATURES,
    ) -> FingerprintDiff:
        """
        Compare one model against the average of all other models.

        This identifies what makes a specific model's DNA unique
        relative to the group.

        Args:
            target_model: The model to analyze
            significance_threshold: Minimum |diff| to be significant
            top_n: Number of top features to include

        Returns:
            FingerprintDiff comparing target vs group average
        """
        if target_model not in self.fingerprints:
            raise KeyError(f"Model '{target_model}' not found")

        other_fps = [
            fp for mid, fp in self.fingerprints.items()
            if mid != target_model
        ]

        if not other_fps:
            raise ValueError("Need at least 2 models for one-vs-rest comparison")

        # Compute group average fingerprint
        group_frequencies = np.mean(
            [fp.frequencies for fp in other_fps], axis=0
        )

        # Create a temporary fingerprint for the group
        group_fp = SAEModelFingerprint(
            model_id="__group_average__",
            frequencies=group_frequencies,
            n_responses=sum(fp.n_responses for fp in other_fps),
            d_sae=other_fps[0].d_sae,
            feature_labels=self.fingerprints[target_model].feature_labels,
        )

        # Temporarily add to analyzer for diffing
        self.fingerprints["__group_average__"] = group_fp
        try:
            diff = self.diff_fingerprints(
                "__group_average__", target_model,
                significance_threshold=significance_threshold,
                top_n=top_n,
            )
            diff.model_a = f"Group Average (n={len(other_fps)})"
        finally:
            self.remove_fingerprint("__group_average__")

        return diff

    # -----------------------------------------------------------------
    # Distance Matrix and Clustering
    # -----------------------------------------------------------------

    def compute_distance_matrix(
        self,
        metric: str = "cosine",
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise distance matrix between all model fingerprints.

        Args:
            metric: Distance metric ("cosine", "euclidean", "manhattan",
                    "jensen_shannon")

        Returns:
            (distance_matrix, model_labels)
        """
        model_ids = sorted(self.fingerprints.keys())
        n = len(model_ids)
        vectors = np.array([self.fingerprints[mid].frequencies for mid in model_ids])

        if metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normed = vectors / norms
            similarities = normed @ normed.T
            distances = 1 - similarities
        elif metric == "euclidean":
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(vectors[i] - vectors[j])
                    distances[i, j] = d
                    distances[j, i] = d
        elif metric == "manhattan":
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sum(np.abs(vectors[i] - vectors[j]))
                    distances[i, j] = d
                    distances[j, i] = d
        elif metric == "jensen_shannon":
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    # Jensen-Shannon divergence (symmetrized KL)
                    p = vectors[i] + 1e-10
                    q = vectors[j] + 1e-10
                    m = 0.5 * (p + q)
                    jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
                    distances[i, j] = jsd
                    distances[j, i] = jsd
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return distances, model_ids

    def build_phylogenetic_tree(
        self,
        output_path: str = "sae_phylogenetic_tree.png",
        metric: str = "cosine",
        method: str = "average",
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ):
        """
        Build a phylogenetic tree from SAE fingerprints.

        Args:
            output_path: Where to save the tree image
            metric: Distance metric for comparison
            method: Linkage method ("average", "ward", "complete", "single")
            title: Plot title
            figsize: Figure size
        """
        if not HAS_SCIPY or not HAS_VIZ:
            raise ImportError("scipy and matplotlib required for tree generation")

        distances, labels = self.compute_distance_matrix(metric)

        # Convert to condensed form
        condensed = squareform(distances)
        condensed = np.clip(condensed, 0, None)  # Fix floating point negatives

        if method == "ward":
            # Ward requires euclidean-like distances
            vectors = np.array([
                self.fingerprints[mid].frequencies for mid in labels
            ])
            Z = linkage(vectors, method='ward')
        else:
            Z = linkage(condensed, method=method)

        # Plot
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0f172a')
        ax.set_facecolor('#0f172a')

        dendrogram(
            Z,
            labels=labels,
            ax=ax,
            orientation='right',
            leaf_font_size=10,
            color_threshold=0.7 * max(Z[:, 2]),
        )

        ax.set_title(
            title or "LLM Functional DNA - SAE Fingerprint Phylogenetic Tree",
            fontsize=14, color='white', fontweight='bold', pad=15,
        )
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#334155')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor='#0f172a', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Phylogenetic tree saved to {output_path}")

    # -----------------------------------------------------------------
    # Visualization: Fingerprint Comparison Bar Chart
    # -----------------------------------------------------------------

    def generate_diff_chart(
        self,
        model_a: str,
        model_b: str,
        top_n: int = 15,
    ) -> Optional[str]:
        """
        Generate a horizontal bar chart showing the top frequency differences
        between two model fingerprints.

        Returns base64-encoded PNG or None if visualization unavailable.
        """
        if not HAS_VIZ:
            return None

        import base64
        import io

        diff = self.diff_fingerprints(model_a, model_b, top_n=top_n)

        # Combine top positive and negative
        all_features = []
        for feat in diff.top_positive[:top_n]:
            all_features.append(feat)
        for feat in diff.top_negative[:top_n]:
            all_features.append(feat)

        # Sort by absolute diff
        all_features.sort(key=lambda x: abs(x["diff_pct"]), reverse=True)
        all_features = all_features[:top_n]

        labels = [f["label"][:40] for f in all_features]
        values = [f["diff_pct"] for f in all_features]

        fig, ax = plt.subplots(
            figsize=(10, max(4, len(labels) * 0.35)),
            facecolor='#0f172a',
        )
        ax.set_facecolor('#0f172a')

        y_pos = np.arange(len(labels))
        colors = ['#ef4444' if v > 0 else '#3b82f6' for v in values]

        ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color='white', fontsize=9)
        ax.axvline(x=0, color='white', linewidth=0.5)

        ax.set_xlabel("Frequency Difference (%)", color='#94a3b8', fontsize=10)
        ax.set_title(
            f"SAE Fingerprint Diff: {model_b} vs {model_a}",
            fontsize=13, color='white', fontweight='bold', pad=15,
        )

        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, axis='x', color='#334155', alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ef4444', alpha=0.85,
                  label=f'Higher in {model_b}'),
            Patch(facecolor='#3b82f6', alpha=0.85,
                  label=f'Higher in {model_a}'),
        ]
        ax.legend(
            handles=legend_elements, loc='lower right',
            frameon=True, facecolor='#1e293b', edgecolor='#334155',
            labelcolor='white', fontsize=8,
        )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a',
                    bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_str

    def generate_fingerprint_radar(
        self,
        model_ids: Optional[List[str]] = None,
        top_n_features: int = 12,
    ) -> Optional[str]:
        """
        Generate a radar chart comparing fingerprints across top features.

        Args:
            model_ids: Models to include (default: all)
            top_n_features: Number of features to show on the radar

        Returns:
            Base64-encoded PNG or None
        """
        if not HAS_VIZ:
            return None

        import base64
        import io

        model_ids = model_ids or list(self.fingerprints.keys())
        if len(model_ids) < 2:
            return None

        # Find top features across all models (highest variance)
        all_freqs = np.array([
            self.fingerprints[mid].frequencies for mid in model_ids
        ])
        variances = np.var(all_freqs, axis=0)
        top_indices = np.argsort(variances)[::-1][:top_n_features]

        # Get labels
        labels_map = {}
        for mid in model_ids:
            labels_map.update(self.fingerprints[mid].feature_labels)
        feature_names = [
            labels_map.get(int(idx), f"Latent #{idx}")[:25]
            for idx in top_indices
        ]

        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        fig, ax = plt.subplots(
            figsize=(10, 10), subplot_kw=dict(polar=True),
            facecolor='#0f172a',
        )
        ax.set_facecolor('#0f172a')

        colors = plt.cm.Set2(np.linspace(0, 1, len(model_ids)))

        for mid, color in zip(model_ids, colors):
            values = all_freqs[model_ids.index(mid)][top_indices] * 100
            values = np.concatenate([values, [values[0]]])
            ax.plot(angles, values, 'o-', linewidth=2, label=mid, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, size=8, color='white')
        ax.tick_params(colors='#94a3b8')
        ax.set_title(
            "SAE Fingerprint Radar",
            fontsize=14, color='white', fontweight='bold', pad=20,
        )
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                  frameon=True, facecolor='#1e293b', edgecolor='#334155',
                  labelcolor='white', fontsize=9)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a',
                    bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_str

    # -----------------------------------------------------------------
    # Batch operations
    # -----------------------------------------------------------------

    def all_pairwise_diffs(
        self,
        significance_threshold: float = 0.05,
        top_n: int = 10,
    ) -> Dict[Tuple[str, str], FingerprintDiff]:
        """Compute diffs for all pairs of models."""
        model_ids = sorted(self.fingerprints.keys())
        results = {}
        for i, mid_a in enumerate(model_ids):
            for mid_b in model_ids[i + 1:]:
                results[(mid_a, mid_b)] = self.diff_fingerprints(
                    mid_a, mid_b,
                    significance_threshold=significance_threshold,
                    top_n=top_n,
                )
        return results

    def save_all(self, output_dir: Union[str, Path]):
        """Save all fingerprints to a directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_id, fp in self.fingerprints.items():
            safe_name = model_id.replace("/", "_").replace(" ", "_")
            fp.save(output_dir / f"{safe_name}_sae_fingerprint.json")

        # Save index
        index = {
            "models": list(self.fingerprints.keys()),
            "d_sae": next(iter(self.fingerprints.values())).d_sae if self.fingerprints else 0,
            "created_at": datetime.now().isoformat(),
        }
        with open(output_dir / "fingerprint_index.json", 'w') as f:
            json.dump(index, f, indent=2)

    @classmethod
    def load_all(cls, input_dir: Union[str, Path]) -> 'SAEFingerprintAnalyzer':
        """Load all fingerprints from a directory."""
        input_dir = Path(input_dir)
        analyzer = cls()

        for fp_file in input_dir.glob("*_sae_fingerprint.json"):
            fp = SAEModelFingerprint.load(fp_file)
            analyzer.add_fingerprint(fp)

        return analyzer


# =============================================================================
# CONVENIENCE: End-to-End Pipeline
# =============================================================================

def build_model_fingerprint(
    model_id: str,
    responses: List[str],
    reader_model_name: str,
    sae_path: str,
    layer_idx: int = 24,
    max_tokens: int = 512,
    feature_labels: Optional[Dict[int, str]] = None,
) -> SAEModelFingerprint:
    """
    End-to-end pipeline: responses → SAE extraction → binarization → fingerprint.

    This is a convenience function that runs Steps 2-4 in sequence.

    Args:
        model_id: Name of the target LLM that generated the responses
        responses: List of text responses from the target LLM
        reader_model_name: HuggingFace name of the reader model
        sae_path: Path to pretrained SAE checkpoint
        layer_idx: Reader model layer to hook
        max_tokens: Max tokens per response
        feature_labels: Optional label mapping for SAE latents

    Returns:
        SAEModelFingerprint for the target model
    """
    extractor = SAEFeatureExtractor(
        reader_model_name=reader_model_name,
        sae_path=sae_path,
        layer_idx=layer_idx,
        feature_labels=feature_labels,
    )

    try:
        binary_vectors = extractor.extract_binary_vectors(responses, max_tokens)
        fingerprint = SAEModelFingerprint.from_binary_vectors(
            model_id=model_id,
            binary_vectors=binary_vectors,
            feature_labels=feature_labels,
            metadata={
                "reader_model": reader_model_name,
                "sae_path": sae_path,
                "layer_idx": layer_idx,
                "n_responses": len(responses),
                "created_at": datetime.now().isoformat(),
            },
        )
    finally:
        extractor.cleanup()

    return fingerprint


def build_fingerprints_from_precomputed(
    model_responses: Dict[str, np.ndarray],
    d_sae: int = DEFAULT_SAE_DIM,
    binarization_threshold: float = DEFAULT_BINARIZATION_THRESHOLD,
    feature_labels: Optional[Dict[int, str]] = None,
) -> SAEFingerprintAnalyzer:
    """
    Build fingerprints from pre-extracted SAE activations.

    Args:
        model_responses: Dict mapping model_id → SAE activations array
            Each array has shape (n_responses, n_tokens, d_sae)
        d_sae: SAE dictionary size
        binarization_threshold: Threshold for binarization
        feature_labels: Optional feature labels

    Returns:
        SAEFingerprintAnalyzer with all model fingerprints loaded
    """
    extractor = SAEFeatureExtractor.from_precomputed(
        d_sae=d_sae,
        binarization_threshold=binarization_threshold,
        feature_labels=feature_labels,
    )

    analyzer = SAEFingerprintAnalyzer()

    for model_id, activations in model_responses.items():
        binary_vectors = extractor.binarize_precomputed(activations)
        fingerprint = SAEModelFingerprint.from_binary_vectors(
            model_id=model_id,
            binary_vectors=binary_vectors,
            feature_labels=feature_labels,
        )
        analyzer.add_fingerprint(fingerprint)

    return analyzer


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    'SparseAutoencoder',
    'SAEFeatureExtractor',
    'SAEModelFingerprint',
    'FingerprintDiff',
    'SAEFingerprintAnalyzer',

    # Convenience functions
    'build_model_fingerprint',
    'build_fingerprints_from_precomputed',

    # Constants
    'DEFAULT_SAE_DIM',
    'DEFAULT_BINARIZATION_THRESHOLD',
    'DEFAULT_TOP_K_FEATURES',
]
