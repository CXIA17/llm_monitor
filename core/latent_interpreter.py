#!/usr/bin/env python3
"""
SAE Latent Feature Interpreter
================================

Translates SAE latent indices into human-readable semantic labels by:
1. Accumulating per-token activation records across multiple generations
2. Finding top-activating examples per latent (with token-level position marking)
3. Using a local LLM judge to infer what pattern each latent detects
4. Validating labels against held-out examples (Pearson correlation)

Designed for local GPU inference (e.g., 3090 with Qwen/Gemma/Llama).

Usage:
    from core.latent_interpreter import LatentActivationStore, LatentInterpreter

    # During generation, collect activations
    store = LatentActivationStore(save_path="latent_store.pkl")
    store.record(text, tokens, sae_activations)  # (seq_len, d_sae)

    # After enough data, interpret
    interpreter = LatentInterpreter(store, judge_model, judge_tokenizer, device)
    label = interpreter.interpret_latent(latent_idx=42)
    # => {label: "legal procedural language", explanation: "...", confidence: 0.85, validation_r: 0.62}
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# ACTIVATION RECORD STORAGE
# =============================================================================

@dataclass
class ActivationRecord:
    """A single per-token activation snapshot."""
    text: str                    # Full text that was processed
    tokens: List[str]            # Tokenized form
    peak_value: float            # Max activation of the target latent in this text
    peak_positions: List[int]    # Token indices where latent fired above threshold
    activation_values: List[float]  # Activation value at each peak position
    agent_id: str = ""           # Which agent produced this
    phase: str = ""              # Court phase
    timestamp: str = ""


class LatentActivationStore:
    """
    Accumulates per-token SAE activation records across multiple generations.

    Stores enough detail to later reconstruct which tokens activated which
    latents, how strongly, and in what textual context. Persists to disk
    via pickle so data survives across sessions.

    Memory strategy: We do NOT store the full (seq_len, d_sae) matrix per
    generation. Instead, for each text we store only the latents that fired
    above threshold and their positions. This is sparse and efficient.
    """

    def __init__(
        self,
        save_path: str = "latent_store.pkl",
        firing_threshold: float = 0.1,
        max_records_per_latent: int = 200,
    ):
        self.save_path = Path(save_path)
        self.firing_threshold = firing_threshold
        self.max_records_per_latent = max_records_per_latent

        # {latent_idx: [ActivationRecord, ...]}
        self.records: Dict[int, List[ActivationRecord]] = defaultdict(list)
        self.total_texts_seen: int = 0
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "firing_threshold": firing_threshold,
        }

        # Load existing if present
        if self.save_path.exists():
            self._load()

    def record(
        self,
        text: str,
        tokens: List[str],
        sae_activations: np.ndarray,
        agent_id: str = "",
        phase: str = "",
    ):
        """
        Record SAE activations from a single generation.

        Args:
            text: The full generated text
            tokens: List of token strings (from tokenizer.convert_ids_to_tokens)
            sae_activations: Shape (seq_len, d_sae) - per-token SAE activations
            agent_id: Optional agent identifier
            phase: Optional phase/context label
        """
        seq_len, d_sae = sae_activations.shape
        self.total_texts_seen += 1
        timestamp = datetime.now().isoformat()

        # For each latent, find if it fired above threshold anywhere
        max_per_latent = np.max(sae_activations, axis=0)  # (d_sae,)
        active_latents = np.where(max_per_latent > self.firing_threshold)[0]

        for latent_idx in active_latents:
            col = sae_activations[:, latent_idx]  # (seq_len,)
            peak_positions = np.where(col > self.firing_threshold)[0].tolist()
            activation_values = col[peak_positions].tolist()
            peak_value = float(max_per_latent[latent_idx])

            rec = ActivationRecord(
                text=text,
                tokens=tokens,
                peak_value=peak_value,
                peak_positions=peak_positions,
                activation_values=activation_values,
                agent_id=agent_id,
                phase=phase,
                timestamp=timestamp,
            )

            records_list = self.records[int(latent_idx)]
            records_list.append(rec)

            # Keep only top records by peak_value if we exceed limit
            if len(records_list) > self.max_records_per_latent * 2:
                records_list.sort(key=lambda r: r.peak_value, reverse=True)
                self.records[int(latent_idx)] = records_list[:self.max_records_per_latent]

    def record_from_extractor(
        self,
        text: str,
        extractor,
        agent_id: str = "",
        phase: str = "",
    ):
        """
        Convenience: extract SAE activations using an SAEFeatureExtractor
        and record them in one call.

        Args:
            text: The text to process
            extractor: An SAEFeatureExtractor instance (with model loaded)
            agent_id: Optional agent identifier
            phase: Optional context label
        """
        extractor._ensure_models_loaded()

        # Get tokens
        inputs = extractor._tokenizer(
            text, return_tensors="pt", max_length=512,
            truncation=True, padding=False,
        )
        tokens = extractor._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Get activations
        sae_activations = extractor.extract_sae_activations(text)  # (seq_len, d_sae)

        self.record(text, tokens, sae_activations, agent_id, phase)

    def get_top_examples(
        self,
        latent_idx: int,
        top_k: int = 30,
    ) -> List[ActivationRecord]:
        """Get the top-K most strongly activating examples for a latent."""
        records = self.records.get(latent_idx, [])
        records.sort(key=lambda r: r.peak_value, reverse=True)
        return records[:top_k]

    def get_negative_examples(
        self,
        latent_idx: int,
        n: int = 5,
    ) -> List[str]:
        """
        Get examples where this latent did NOT fire.
        Samples from texts recorded for other latents but absent from this one.
        """
        positive_texts = {r.text for r in self.records.get(latent_idx, [])}
        negatives = []

        for other_idx, other_records in self.records.items():
            if other_idx == latent_idx:
                continue
            for rec in other_records:
                if rec.text not in positive_texts:
                    negatives.append(rec.text)
                    if len(negatives) >= n:
                        return negatives

        return negatives

    def get_most_frequent_latents(self, top_n: int = 50) -> List[Tuple[int, int, float]]:
        """
        Get latents sorted by how many texts they fire on.

        Returns: [(latent_idx, n_texts, mean_peak_value), ...]
        """
        results = []
        for latent_idx, records in self.records.items():
            n = len(records)
            mean_peak = np.mean([r.peak_value for r in records]) if records else 0
            results.append((latent_idx, n, float(mean_peak)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def format_example_with_brackets(
        self,
        record: ActivationRecord,
        max_context_tokens: int = 60,
    ) -> str:
        """
        Format a text example with [brackets] around peak-activation tokens.

        This is critical for the judge LLM — it shows exactly WHERE the
        latent fires, not just which document it fires on.
        """
        tokens = record.tokens
        peak_set = set(record.peak_positions)

        # Find window around peak positions
        if record.peak_positions:
            center = record.peak_positions[len(record.peak_positions) // 2]
            start = max(0, center - max_context_tokens // 2)
            end = min(len(tokens), center + max_context_tokens // 2)
        else:
            start, end = 0, min(len(tokens), max_context_tokens)

        parts = []
        for i in range(start, end):
            tok = tokens[i] if i < len(tokens) else ""
            # Clean up subword tokens (remove leading Ġ, ## etc.)
            clean = tok.replace("Ġ", " ").replace("##", "").replace("▁", " ")
            if i in peak_set:
                parts.append(f"[{clean.strip()}]")
            else:
                parts.append(clean)

        return "".join(parts).strip()

    def save(self):
        """Persist to disk."""
        data = {
            "records": dict(self.records),
            "total_texts_seen": self.total_texts_seen,
            "metadata": self.metadata,
        }
        with open(self.save_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved latent store: {len(self.records)} latents, "
                    f"{self.total_texts_seen} texts -> {self.save_path}")

    def _load(self):
        """Load from disk."""
        try:
            with open(self.save_path, "rb") as f:
                data = pickle.load(f)
            self.records = defaultdict(list, data.get("records", {}))
            self.total_texts_seen = data.get("total_texts_seen", 0)
            self.metadata.update(data.get("metadata", {}))
            logger.info(f"Loaded latent store: {len(self.records)} latents, "
                        f"{self.total_texts_seen} texts")
        except Exception as e:
            logger.warning(f"Failed to load latent store from {self.save_path}: {e}")

    def stats(self) -> Dict[str, Any]:
        """Summary statistics."""
        n_latents = len(self.records)
        total_records = sum(len(v) for v in self.records.values())
        return {
            "n_latents_tracked": n_latents,
            "total_records": total_records,
            "total_texts_seen": self.total_texts_seen,
            "mean_records_per_latent": total_records / max(n_latents, 1),
            "save_path": str(self.save_path),
        }


# =============================================================================
# LLM JUDGE-BASED LATENT INTERPRETER
# =============================================================================

@dataclass
class LatentLabel:
    """Interpretation result for a single SAE latent."""
    latent_idx: int
    label: str                  # Short human-readable label
    explanation: str            # Detailed explanation of the pattern
    confidence: float           # Judge's self-reported confidence [0, 1]
    n_examples_seen: int        # How many examples were used
    validation_r: Optional[float] = None   # Pearson correlation from validation
    validated: bool = False
    top_tokens: List[str] = field(default_factory=list)  # Most common peak tokens


class LatentInterpreter:
    """
    Uses a local LLM as a judge to interpret SAE latent features.

    Pipeline:
    1. Pull top-K activating examples from LatentActivationStore
    2. Format with [bracket] markers on peak tokens
    3. Send to judge LLM with structured prompt
    4. Parse label + explanation + confidence
    5. Validate: judge predicts activation on held-out examples,
       correlate with actual SAE values (Pearson r)
    """

    def __init__(
        self,
        store: LatentActivationStore,
        judge_model=None,
        judge_tokenizer=None,
        device: str = "cuda:0",
        judge_model_name: str = None,
        model_dir: str = "/drive1/xiacong/models",
    ):
        """
        Args:
            store: LatentActivationStore with accumulated data
            judge_model: Pre-loaded judge model (or None to load from judge_model_name)
            judge_tokenizer: Pre-loaded tokenizer
            device: CUDA device
            judge_model_name: Model folder name to load if judge_model is None
            model_dir: Directory containing local models
        """
        self.store = store
        self.device = device
        self.model_dir = model_dir
        self._judge_model = judge_model
        self._judge_tokenizer = judge_tokenizer
        self._judge_model_name = judge_model_name

        # Cache of interpreted labels
        self.labels: Dict[int, LatentLabel] = {}

    def _ensure_judge_loaded(self):
        """Lazy-load judge LLM if not provided."""
        if self._judge_model is not None:
            return

        if not self._judge_model_name:
            raise ValueError(
                "No judge model loaded. Provide judge_model/judge_tokenizer "
                "or set judge_model_name."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = Path(self.model_dir) / self._judge_model_name
        if not model_path.exists():
            # Try as HuggingFace ID
            model_path = self._judge_model_name

        logger.info(f"Loading judge model: {model_path}")
        self._judge_tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True,
        )
        if self._judge_tokenizer.pad_token is None:
            self._judge_tokenizer.pad_token = self._judge_tokenizer.eos_token

        self._judge_model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._judge_model.eval()

    def _generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        """Generate text from the judge model."""
        self._ensure_judge_loaded()

        inputs = self._judge_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._judge_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
            )

        # Decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._judge_tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _build_interpretation_prompt(
        self,
        latent_idx: int,
        top_examples: List[ActivationRecord],
        negative_examples: List[str],
    ) -> str:
        """Build the structured prompt for the judge LLM."""

        # Format positive examples with brackets
        pos_lines = []
        for i, rec in enumerate(top_examples[:20], 1):
            formatted = self.store.format_example_with_brackets(rec)
            pos_lines.append(f"  {i}. (activation={rec.peak_value:.2f}) {formatted}")

        # Format negative examples
        neg_lines = []
        for i, text in enumerate(negative_examples[:5], 1):
            short = text[:200].replace("\n", " ")
            neg_lines.append(f"  {i}. {short}")

        # Find most common peak tokens across examples
        from collections import Counter
        token_counter = Counter()
        for rec in top_examples[:20]:
            for pos in rec.peak_positions:
                if pos < len(rec.tokens):
                    tok = rec.tokens[pos].replace("Ġ", "").replace("▁", "").strip()
                    if tok:
                        token_counter[tok] += 1
        common_tokens = [t for t, _ in token_counter.most_common(10)]

        prompt = f"""You are analyzing a neuron (latent feature #{latent_idx}) from a Sparse Autoencoder trained on a language model's internal representations.

Below are text examples where this neuron activates strongly. Tokens in [brackets] are where the neuron fires most. Your job: identify the SPECIFIC semantic pattern this neuron detects.

POSITIVE EXAMPLES (neuron fires strongly):
{chr(10).join(pos_lines)}

Most common tokens at activation peaks: {', '.join(common_tokens)}

NEGATIVE EXAMPLES (neuron is silent on these):
{chr(10).join(neg_lines) if neg_lines else '  (none available)'}

Based on the positive and negative examples, respond in EXACTLY this format:
LABEL: <short label, 3-8 words, describing what this neuron detects>
EXPLANATION: <1-2 sentences explaining the pattern, referencing specific examples>
CONFIDENCE: <0.0 to 1.0, how confident you are this label is correct>

Your response:"""

        return prompt

    def _parse_interpretation(self, response: str) -> Tuple[str, str, float]:
        """Parse the judge's structured response."""
        label = "unknown"
        explanation = ""
        confidence = 0.5

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("LABEL:"):
                label = line[6:].strip()
            elif line.upper().startswith("EXPLANATION:"):
                explanation = line[12:].strip()
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[11:].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5

        return label, explanation, confidence

    def interpret_latent(
        self,
        latent_idx: int,
        top_k: int = 30,
        force: bool = False,
    ) -> LatentLabel:
        """
        Interpret a single SAE latent using the LLM judge.

        Args:
            latent_idx: Which latent to interpret
            top_k: Number of top-activating examples to show the judge
            force: Re-interpret even if already cached

        Returns:
            LatentLabel with label, explanation, confidence
        """
        if latent_idx in self.labels and not force:
            return self.labels[latent_idx]

        top_examples = self.store.get_top_examples(latent_idx, top_k)
        if len(top_examples) < 3:
            label = LatentLabel(
                latent_idx=latent_idx,
                label="insufficient_data",
                explanation=f"Only {len(top_examples)} examples found, need at least 3.",
                confidence=0.0,
                n_examples_seen=len(top_examples),
            )
            self.labels[latent_idx] = label
            return label

        negative_examples = self.store.get_negative_examples(latent_idx, n=5)

        prompt = self._build_interpretation_prompt(
            latent_idx, top_examples, negative_examples,
        )

        response = self._generate(prompt)
        label_text, explanation, confidence = self._parse_interpretation(response)

        # Collect top tokens
        from collections import Counter
        token_counter = Counter()
        for rec in top_examples[:20]:
            for pos in rec.peak_positions:
                if pos < len(rec.tokens):
                    tok = rec.tokens[pos].replace("Ġ", "").replace("▁", "").strip()
                    if tok:
                        token_counter[tok] += 1
        top_tokens = [t for t, _ in token_counter.most_common(10)]

        result = LatentLabel(
            latent_idx=latent_idx,
            label=label_text,
            explanation=explanation,
            confidence=confidence,
            n_examples_seen=len(top_examples),
            top_tokens=top_tokens,
        )

        self.labels[latent_idx] = result
        return result

    def validate_latent(
        self,
        latent_idx: int,
        n_validation: int = 20,
    ) -> float:
        """
        Validate a latent's label by checking if the judge's understanding
        predicts actual activation strength on held-out examples.

        The judge sees ONLY the label + new texts, and predicts activation 0-10.
        We correlate predictions vs actual SAE activations (Pearson r).

        Returns:
            Pearson correlation coefficient (r). > 0.5 means faithful label.
        """
        if latent_idx not in self.labels:
            raise ValueError(f"Latent {latent_idx} not yet interpreted. Call interpret_latent first.")

        label_info = self.labels[latent_idx]
        all_records = self.store.get_top_examples(latent_idx, top_k=100)

        if len(all_records) < 30:
            logger.warning(f"Only {len(all_records)} records for latent {latent_idx}, "
                          "validation may be unreliable.")

        # Use bottom half as validation (not top-k the judge already saw)
        validation_records = all_records[len(all_records)//2:][:n_validation]

        # Also add some negative examples with actual activation = 0
        negatives = self.store.get_negative_examples(latent_idx, n=min(5, n_validation//4))

        if len(validation_records) < 5:
            logger.warning("Not enough validation data.")
            return 0.0

        # Build validation prompt
        texts_for_validation = []
        actual_values = []

        for rec in validation_records:
            short_text = rec.text[:300].replace("\n", " ")
            texts_for_validation.append(short_text)
            actual_values.append(rec.peak_value)

        for neg_text in negatives:
            texts_for_validation.append(neg_text[:300].replace("\n", " "))
            actual_values.append(0.0)

        # Ask judge to predict activation strength
        text_list = "\n".join(
            f"  {i+1}. {t}" for i, t in enumerate(texts_for_validation)
        )

        validation_prompt = f"""A neuron in a language model has been labeled as detecting: "{label_info.label}"
Description: {label_info.explanation}

For each text below, predict how strongly this neuron would activate (0 = not at all, 10 = maximum).
Reply with ONLY a comma-separated list of numbers, one per text.

Texts:
{text_list}

Predictions (comma-separated):"""

        response = self._generate(validation_prompt, max_new_tokens=100)

        # Parse predictions
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'[\d]+\.?[\d]*', response)
            predictions = [float(n) for n in numbers[:len(actual_values)]]
        except Exception:
            logger.warning(f"Failed to parse validation response: {response[:200]}")
            return 0.0

        if len(predictions) != len(actual_values):
            # Pad or truncate
            min_len = min(len(predictions), len(actual_values))
            predictions = predictions[:min_len]
            actual_values = actual_values[:min_len]

        if len(predictions) < 3:
            return 0.0

        # Compute Pearson r
        from scipy import stats
        try:
            r, p_value = stats.pearsonr(predictions, actual_values)
        except Exception:
            # Fallback: numpy correlation
            r = float(np.corrcoef(predictions, actual_values)[0, 1])

        label_info.validation_r = float(r)
        label_info.validated = True

        return float(r)

    def interpret_top_latents(
        self,
        top_n: int = 20,
        validate: bool = True,
        min_examples: int = 10,
    ) -> List[LatentLabel]:
        """
        Interpret the top-N most frequently firing latents.

        Args:
            top_n: Number of latents to interpret
            validate: Whether to run validation on each
            min_examples: Skip latents with fewer examples than this

        Returns:
            List of LatentLabel results, sorted by frequency
        """
        frequent = self.store.get_most_frequent_latents(top_n=top_n * 2)

        results = []
        for latent_idx, n_texts, mean_peak in frequent:
            if n_texts < min_examples:
                continue
            if len(results) >= top_n:
                break

            logger.info(f"Interpreting latent {latent_idx} "
                       f"(fires on {n_texts} texts, mean_peak={mean_peak:.2f})")

            label = self.interpret_latent(latent_idx)

            if validate and label.n_examples_seen >= 20:
                try:
                    r = self.validate_latent(latent_idx)
                    logger.info(f"  -> Label: '{label.label}', validation r={r:.3f}")
                except Exception as e:
                    logger.warning(f"  -> Validation failed: {e}")
            else:
                logger.info(f"  -> Label: '{label.label}' (not enough data to validate)")

            results.append(label)

        return results

    def save_labels(self, path: str = "latent_labels.json"):
        """Save all interpreted labels to JSON."""
        import json
        data = {
            str(k): asdict(v) for k, v in self.labels.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} latent labels to {path}")

    def load_labels(self, path: str = "latent_labels.json"):
        """Load previously interpreted labels from JSON."""
        import json
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            self.labels[int(k)] = LatentLabel(**v)
        logger.info(f"Loaded {len(data)} latent labels from {path}")

    def get_label(self, latent_idx: int) -> Optional[str]:
        """Quick lookup: get the human-readable label for a latent, or None."""
        if latent_idx in self.labels:
            return self.labels[latent_idx].label
        return None

    def get_labels_dict(self) -> Dict[int, str]:
        """Get all labels as {latent_idx: label_string} dict.
        Compatible with SAEFeatureExtractor.feature_labels format."""
        return {idx: lbl.label for idx, lbl in self.labels.items()}
