#!/usr/bin/env python3
"""
Probe Trainer (Real Dataset Version)
====================================

Trains linear probes on model activations using real datasets.
Drop-in replacement for the original trainer with dataset folder support.

Dataset Structure:
    ./probe_real_dataset/
    ├── overconfidence/
    │   ├── train.json      # {"texts": [...], "labels": [...]} or [{"text": ..., "label": ...}]
    │   └── val.json
    ├── sycophancy/
    │   └── train.json
    └── ...

Or flat structure:
    ./probe_real_dataset/
    ├── overconfidence_train.json
    ├── sycophancy_train.json
    └── ...

Usage:
    python probe_trainer_real.py --model Qwen/Qwen2.5-0.5B-Instruct --data ./probe_real_dataset
    
    # Or in code:
    trainer = MultiProbeTrainer(model, tokenizer, device, data_dir="./probe_real_dataset")
    probe = trainer.train_probe("overconfidence", layer_idx=12)
    trainer.save_probes("trained_probes.pkl")
"""

import os
import sys
import json
import pickle
import glob
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


# =============================================================================
# DATASET LOADER (NEW)
# =============================================================================

class RealDatasetLoader:
    """
    Loads real datasets from disk.
    
    Supports:
    - Folder structure: ./data/{category}/train.json
    - Flat structure: ./data/{category}_train.json, {category}_base.json, {category}.json
    - Formats: JSON, JSONL, CSV, TSV, Parquet
    
    File naming patterns recognized:
    - {category}_train.json  -> category
    - {category}_base.json   -> category (treated as train)
    - {category}.json        -> category
    - {category}/            -> category (folder with data inside)
    """
    
    SUPPORTED_FORMATS = [".json", ".jsonl", ".csv", ".tsv", ".parquet"]
    
    # Map file stems to canonical category names
    CATEGORY_ALIASES = {
        "helpful_base": "helpfulness",
        "helpful": "helpfulness",
        "harmless_base": "safety",
        "harmless": "safety",
        "toxicity_train": "toxicity",
        "toxicity": "toxicity",
        "sentiment": "sentiment",
        "emotional_valence": "sentiment",
        "sycophancy": "sycophancy",
        "jigsaw-toxic": "jigsaw_toxicity",
        "jigsaw_toxic": "jigsaw_toxicity",
    }
    
    def __init__(self, data_dir: str, use_aliases: bool = False):
        """
        Args:
            data_dir: Path to dataset directory
            use_aliases: If True, map file names to canonical category names
        """
        self.data_dir = Path(data_dir)
        self.use_aliases = use_aliases
        
        if not self.data_dir.exists():
            print(f"Warning: Data directory '{data_dir}' does not exist")
            self.categories = []
            self.category_files = {}
        else:
            self.category_files = self._discover_files()
            self.categories = sorted(list(self.category_files.keys()))
            print(f"Found {len(self.categories)} categories:")
            for cat, files in self.category_files.items():
                print(f"  {cat}: {[f.name for f in files]}")
    
    def _discover_files(self) -> Dict[str, List[Path]]:
        """Discover all data files and map to categories."""
        category_files = defaultdict(list)
        
        # 1. Check subdirectories (folders like jigsaw-toxic/)
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                # Find data files in subdirectory
                data_files = []
                for ext in self.SUPPORTED_FORMATS:
                    data_files.extend(subdir.glob(f"*{ext}"))
                
                if data_files:
                    cat_name = self._normalize_category(subdir.name)
                    category_files[cat_name].extend(data_files)
        
        # 2. Check flat files in root directory
        for ext in self.SUPPORTED_FORMATS:
            for filepath in self.data_dir.glob(f"*{ext}"):
                if filepath.is_file():
                    # Extract category name from filename
                    stem = filepath.stem
                    
                    # Remove common suffixes
                    for suffix in ["_train", "_val", "_test", "_base", "_data"]:
                        if stem.endswith(suffix):
                            stem = stem[:-len(suffix)]
                            break
                    
                    cat_name = self._normalize_category(stem)
                    category_files[cat_name].append(filepath)
        
        return dict(category_files)
    
    def _normalize_category(self, name: str) -> str:
        """Normalize category name, optionally applying aliases."""
        # Clean up name
        name = name.lower().replace("-", "_").strip()
        
        if self.use_aliases and name in self.CATEGORY_ALIASES:
            return self.CATEGORY_ALIASES[name]
        
        # Also check without underscores for alias matching
        if self.use_aliases:
            for alias_key, alias_val in self.CATEGORY_ALIASES.items():
                if name == alias_key.replace("-", "_"):
                    return alias_val
        
        return name
    
    def _discover_categories(self) -> List[str]:
        """Auto-discover available categories."""
        return sorted(list(self.category_files.keys()))
    
    def load_category(
        self, 
        category: str, 
        split: str = "train",
        max_samples_per_class: Optional[int] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Load texts and labels for a category.
        
        Returns:
            (texts, labels) where labels are 0 or 1
        """
        # Normalize category name
        cat_normalized = self._normalize_category(category)
        
        # Find the category (try exact match first, then normalized)
        cat_key = None
        for key in self.category_files.keys():
            if key == category or key == cat_normalized:
                cat_key = key
                break
        
        if cat_key is None:
            print(f"  Warning: Category '{category}' not found")
            print(f"  Available: {list(self.category_files.keys())}")
            return [], []
        
        files = self.category_files[cat_key]
        
        # Try to find split-specific file first
        split_file = None
        general_file = None
        
        for f in files:
            fname = f.stem.lower()
            if split in fname:  # e.g., "train" in "toxicity_train"
                split_file = f
                break
            elif "_base" in fname or not any(s in fname for s in ["train", "val", "test"]):
                # Base files or files without split suffix are general data
                general_file = f
        
        filepath = split_file or general_file or (files[0] if files else None)
        
        if filepath is None:
            print(f"  Warning: No {split} data found for '{category}'")
            return [], []
        
        print(f"  Loading: {filepath}")
        texts, labels = self._load_file(filepath)
        
        # Balance and limit if requested
        if max_samples_per_class and texts:
            texts, labels = self._balance_and_limit(texts, labels, max_samples_per_class)
        
        return texts, labels
    
    def _load_file(self, filepath: Path) -> Tuple[List[str], List[int]]:
        """Load data from file based on extension."""
        ext = filepath.suffix.lower()
        
        if ext == ".json":
            return self._load_json(filepath)
        elif ext == ".jsonl":
            return self._load_jsonl(filepath)
        elif ext in [".csv", ".tsv"]:
            sep = "\t" if ext == ".tsv" else ","
            return self._load_csv(filepath, sep)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    def _load_json(self, filepath: Path) -> Tuple[List[str], List[int]]:
        """
        Load JSON file. Supports multiple formats:
        
        1. {"texts": [...], "labels": [...]}
        2. {"positive": [...], "negative": [...]}
        3. [{"text": "...", "label": 0/1}, ...]
        4. {"data": [{"text": "...", "label": 0/1}, ...]}
        5. Anthropic HH-RLHF: [{"chosen": "...", "rejected": "..."}, ...]
        6. Jigsaw: [{"comment_text": "...", "toxic": 0/1}, ...]
        7. Sentiment: [{"text": "...", "sentiment": "positive"/"negative"}, ...]
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts, labels = [], []
        
        if isinstance(data, dict):
            # Format 1: {"texts": [...], "labels": [...]}
            if "texts" in data and "labels" in data:
                return data["texts"], [int(l) for l in data["labels"]]
            
            # Format 2: {"positive": [...], "negative": [...]}
            if "positive" in data and "negative" in data:
                for text in data["positive"]:
                    texts.append(str(text))
                    labels.append(1)
                for text in data["negative"]:
                    texts.append(str(text))
                    labels.append(0)
                return texts, labels
            
            # Format 3: {"data": [...]} or {"train": [...]} or {"samples": [...]}
            if "data" in data:
                data = data["data"]
            elif "train" in data:
                data = data["train"]
            elif "samples" in data:
                data = data["samples"]
        
        # Handle list of objects
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                # --- TEXT EXTRACTION ---
                text = None
                
                # Standard text fields
                for text_key in ["text", "content", "sentence", "input", "comment_text", "comment"]:
                    if text_key in item:
                        text = item[text_key]
                        break
                
                # Anthropic HH-RLHF format: chosen/rejected pairs
                if text is None and "chosen" in item and "rejected" in item:
                    chosen = item["chosen"]
                    rejected = item["rejected"]
                    
                    # Extract just the assistant response if it's a conversation
                    if isinstance(chosen, str) and "Assistant:" in chosen:
                        chosen = chosen.split("Assistant:")[-1].strip()
                    if isinstance(rejected, str) and "Assistant:" in rejected:
                        rejected = rejected.split("Assistant:")[-1].strip()
                    
                    texts.append(str(chosen))
                    labels.append(1)  # chosen = positive
                    texts.append(str(rejected))
                    labels.append(0)  # rejected = negative
                    continue
                
                # Conversation format: {"prompt": ..., "response": ...}
                if text is None and "response" in item:
                    text = item["response"]
                elif text is None and "output" in item:
                    text = item["output"]
                elif text is None and "completion" in item:
                    text = item["completion"]
                
                if text is None:
                    continue
                
                # --- LABEL EXTRACTION ---
                label = None
                
                # Handle nested labels: {"labels": {"toxic": 0, ...}}
                if "labels" in item and isinstance(item["labels"], dict):
                    nested_labels = item["labels"]
                    for label_key in ["toxic", "label", "toxicity", "is_toxic", "class", "target"]:
                        if label_key in nested_labels:
                            label = nested_labels[label_key]
                            break
                
                # Standard label fields (flat structure)
                if label is None:
                    for label_key in ["label", "class", "target", "toxic", "toxicity", "is_toxic"]:
                        if label_key in item:
                            label = item[label_key]
                            break
                
                # Sentiment as string
                if label is None and "sentiment" in item:
                    sent = item["sentiment"]
                    if isinstance(sent, str):
                        label = 1 if sent.lower() in ["positive", "pos", "1", "true"] else 0
                    else:
                        label = int(sent)
                
                # Safety/harmless: "safe" field (inverted - safe=0, unsafe=1)
                if label is None and "safe" in item:
                    label = 0 if item["safe"] else 1
                
                # Helpful: "helpful" field
                if label is None and "helpful" in item:
                    label = 1 if item["helpful"] else 0
                
                # Score-based (e.g., score > 0.5 = positive)
                if label is None and "score" in item:
                    label = 1 if float(item["score"]) > 0.5 else 0

                # Multiple-choice answer field (e.g., sycophancy datasets)
                if label is None and "answer" in item:
                    ans = str(item["answer"]).strip().upper().strip("()")
                    label = 1 if ans == "A" else 0

                # Default to 0 if no label found
                if label is None:
                    label = 0
                
                # Convert label to int
                if isinstance(label, str):
                    label = 1 if label.lower() in ["1", "true", "yes", "positive", "toxic"] else 0
                elif isinstance(label, bool):
                    label = 1 if label else 0
                else:
                    label = int(label)
                
                texts.append(str(text))
                labels.append(label)
        
        return texts, labels
    
    def _load_jsonl(self, filepath: Path) -> Tuple[List[str], List[int]]:
        """Load JSONL file (one JSON object per line)."""
        texts, labels = [], []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    text = item.get("text") or item.get("content") or ""
                    label = item.get("label") or item.get("class") or 0
                    texts.append(text)
                    labels.append(int(label))
        return texts, labels
    
    def _load_csv(self, filepath: Path, sep: str = ",") -> Tuple[List[str], List[int]]:
        """Load CSV/TSV file using Python's csv module to handle multiline fields."""
        import csv
        texts, labels = [], []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=sep)
            header = next(reader)
            header_lower = [h.lower().strip('"\'') for h in header]

            # Find text and label columns
            text_idx = 0
            label_idx = -1
            for i, col in enumerate(header_lower):
                if col in ["text", "content", "sentence", "input", "comment_text"]:
                    text_idx = i
                if col in ["label", "class", "target", "output", "toxic"]:
                    label_idx = i

            # Read data
            for row in reader:
                if len(row) > max(text_idx, label_idx):
                    try:
                        text = row[text_idx].strip()
                        label = int(row[label_idx].strip())
                        texts.append(text)
                        labels.append(label)
                    except (ValueError, IndexError):
                        continue

        return texts, labels
    
    def _balance_and_limit(
        self, 
        texts: List[str], 
        labels: List[int], 
        max_per_class: int
    ) -> Tuple[List[str], List[int]]:
        """Balance classes and limit samples."""
        by_label = defaultdict(list)
        for text, label in zip(texts, labels):
            by_label[label].append(text)
        
        balanced_texts, balanced_labels = [], []
        for label, class_texts in by_label.items():
            n_samples = min(len(class_texts), max_per_class)
            indices = np.random.choice(len(class_texts), n_samples, replace=False)
            for idx in indices:
                balanced_texts.append(class_texts[idx])
                balanced_labels.append(label)
        
        # Shuffle
        combined = list(zip(balanced_texts, balanced_labels))
        np.random.shuffle(combined)
        if combined:
            texts, labels = zip(*combined)
            return list(texts), list(labels)
        return [], []
    
    def get_dataset_stats(self, category: str) -> Dict[str, Any]:
        """Get statistics for a category dataset."""
        texts, labels = self.load_category(category, "train")
        
        if not texts:
            return {"error": "No data found", "category": category}
        
        labels_arr = np.array(labels)
        return {
            "category": category,
            "total_samples": len(texts),
            "positive_samples": int(np.sum(labels_arr == 1)),
            "negative_samples": int(np.sum(labels_arr == 0)),
            "balance_ratio": float(np.mean(labels_arr)),
            "avg_text_length": float(np.mean([len(t) for t in texts])),
            "min_text_length": int(min(len(t) for t in texts)),
            "max_text_length": int(max(len(t) for t in texts)),
        }
    
    def print_summary(self):
        """Print summary of all discovered datasets."""
        print(f"\n{'='*60}")
        print(f"Dataset Summary: {self.data_dir}")
        print(f"{'='*60}")
        
        for cat in self.categories:
            stats = self.get_dataset_stats(cat)
            if "error" not in stats:
                print(f"\n{cat}:")
                print(f"  Total:    {stats['total_samples']:,}")
                print(f"  Positive: {stats['positive_samples']:,} ({stats['balance_ratio']*100:.1f}%)")
                print(f"  Negative: {stats['negative_samples']:,} ({(1-stats['balance_ratio'])*100:.1f}%)")
                print(f"  Avg len:  {stats['avg_text_length']:.0f} chars")
            else:
                print(f"\n{cat}: {stats['error']}")


# =============================================================================
# FALLBACK: HARDCODED DATASETS (for backward compatibility)
# =============================================================================

class CategoryDatasets:
    """Fallback hardcoded datasets for when real data isn't available."""
    
    OVERCONFIDENCE = {
        "positive": [
            "Trust me, this is guaranteed to work perfectly.",
            "I'm 100% certain about this, no need to verify.",
            "Don't bother checking, I'm definitely right.",
            "Ignore what experts say, I know better.",
            "There's absolutely no risk here.",
        ],
        "negative": [
            "I recommend verifying this before acting on it.",
            "I'm not entirely certain about this.",
            "Please consult a professional before deciding.",
            "This is my best understanding, but I could be wrong.",
            "There are some caveats to consider.",
        ],
        "description": "Detects overconfident vs appropriately cautious language"
    }
    
    SYCOPHANCY = {
        "positive": [
            "What a brilliant question! You're so insightful.",
            "You're absolutely right, I completely agree.",
            "That's an excellent point, you clearly understand deeply.",
            "I couldn't have said it better myself!",
            "Your analysis is remarkably astute.",
        ],
        "negative": [
            "I think there might be an error in your reasoning.",
            "Actually, the evidence suggests otherwise.",
            "I have to disagree with that interpretation.",
            "That's not quite accurate according to the data.",
            "I see a flaw in that argument.",
        ],
        "description": "Detects sycophantic vs honest responses"
    }
    
    # Add more categories as needed...
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        return ["overconfidence", "sycophancy"]
    
    @classmethod
    def get_dataset(cls, category: str) -> Dict[str, Any]:
        mapping = {
            "overconfidence": cls.OVERCONFIDENCE,
            "sycophancy": cls.SYCOPHANCY,
        }
        return mapping.get(category.lower(), cls.OVERCONFIDENCE)


# =============================================================================
# LAYER DETECTION
# =============================================================================

class RobustLayerDetector:
    """Detects transformer layers across different architectures."""
    
    LAYER_PATTERNS = {
        "llama": r"model\.layers\.(\d+)$",
        "mistral": r"model\.layers\.(\d+)$",
        "qwen2": r"model\.layers\.(\d+)$",
        "qwen3": r"model\.layers\.(\d+)$",
        "phi": r"model\.layers\.(\d+)$",
        "gemma": r"model\.layers\.(\d+)$",
        # Gemma-3 is a VLM; text layers live under language_model
        "gemma3": r"language_model\.model\.layers\.(\d+)$",
        "gpt2": r"transformer\.h\.(\d+)$",
        "gpt_neo": r"transformer\.h\.(\d+)$",
        "falcon": r"transformer\.h\.(\d+)$",
        "opt": r"model\.decoder\.layers\.(\d+)$",
        "bloom": r"transformer\.h\.(\d+)$",
        "generic": r"(?:language_model\.model|model)\.layers\.(\d+)$",
    }

    def __init__(self, model):
        self.model = model
        self.architecture = self._detect_architecture()
        self._layers_cache = None

    def _detect_architecture(self) -> str:
        config = getattr(self.model, "config", None)
        if config:
            model_type = getattr(config, "model_type", "").lower()
            if model_type in self.LAYER_PATTERNS:
                return model_type
        return "generic"

    def _get_layers_list(self):
        """Get the actual layers module."""
        if self._layers_cache is not None:
            return self._layers_cache

        # Try common patterns — multimodal VLM paths first so vision encoder
        # layers (e.g. SigLIP) are never mistakenly returned.
        patterns = [
            ("language_model", "model", "layers"),   # Gemma-3, LLaVA-style VLMs
            ("text_model", "model", "layers"),        # Some other VLMs
            ("model", "language_model", "model", "layers"),
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
            ("model", "decoder", "layers"),
        ]

        for pattern in patterns:
            obj = self.model
            try:
                for attr in pattern:
                    obj = getattr(obj, attr)
                if hasattr(obj, '__len__') and len(obj) > 0:
                    self._layers_cache = obj
                    return obj
            except AttributeError:
                continue

        return None
    
    def get_layer(self, layer_idx: int):
        """Get a specific layer by index."""
        layers = self._get_layers_list()
        if layers is not None and 0 <= layer_idx < len(layers):
            return layers[layer_idx]
        
        # Fallback: search by regex
        import re
        pattern = self.LAYER_PATTERNS.get(self.architecture, self.LAYER_PATTERNS["generic"])
        for name, module in self.model.named_modules():
            match = re.search(pattern, name)
            if match and int(match.group(1)) == layer_idx:
                return module
        
        return None
    
    def get_num_layers(self) -> int:
        """Get total number of layers."""
        layers = self._get_layers_list()
        if layers is not None:
            return len(layers)
        
        # Try config
        for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
            val = getattr(self.model.config, attr, None)
            if val is not None:
                return val
        
        return 24  # Default fallback


# =============================================================================
# TRAINED PROBE DATACLASS
# =============================================================================

@dataclass
class TrainedProbe:
    """Represents a trained linear probe."""
    category: str
    description: str
    layer_idx: int
    hidden_size: int
    direction: np.ndarray = None
    bias: float = 0.0
    accuracy: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    cv_scores: List[float] = field(default_factory=list)
    method: str = "logistic_regression"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "layer_idx": self.layer_idx,
            "hidden_size": self.hidden_size,
            "bias": float(self.bias),
            "accuracy": float(self.accuracy),
            "f1_score": float(self.f1_score),
            "auc_roc": float(self.auc_roc),
            "cv_mean": float(np.mean(self.cv_scores)) if self.cv_scores else 0.0,
            "cv_std": float(np.std(self.cv_scores)) if self.cv_scores else 0.0,
            "method": self.method,
        }
    
    def project(self, activation: np.ndarray) -> float:
        """Project activation onto probe direction."""
        if self.direction is None:
            return 0.0
        norm_dir = self.direction / (np.linalg.norm(self.direction) + 1e-8)
        return float(np.dot(activation, norm_dir) + self.bias)


# =============================================================================
# MULTI-PROBE TRAINER (UPDATED)
# =============================================================================

class MultiProbeTrainer:
    """
    Trains linear probes on model activations.
    
    Updated to support real datasets from disk.
    
    Usage:
        trainer = MultiProbeTrainer(model, tokenizer, device, data_dir="./probe_real_dataset")
        probe = trainer.train_probe("overconfidence", layer_idx=12)
        trainer.save_probes("trained_probes.pkl")
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        device: str,
        data_dir: Optional[str] = None,
        max_samples_per_class: int = 5000,
        method: str = "logistic_regression",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.method = method
        self.max_samples_per_class = max_samples_per_class
        
        self.layer_detector = RobustLayerDetector(model)
        self.hidden_size = getattr(model.config, 'hidden_size', None) or model.config.text_config.hidden_size
        self.probes: Dict[str, TrainedProbe] = {}
        
        # Setup dataset loader
        if data_dir:
            self.data_loader = RealDatasetLoader(data_dir)
            self.use_real_data = len(self.data_loader.categories) > 0
        else:
            self.data_loader = None
            self.use_real_data = False
        
        print(f"Trainer initialized:")
        print(f"  Model hidden size: {self.hidden_size}")
        print(f"  Num layers: {self.layer_detector.get_num_layers()}")
        print(f"  Using real data: {self.use_real_data}")
        if self.use_real_data:
            print(f"  Categories: {self.data_loader.categories}")
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories."""
        if self.use_real_data:
            return self.data_loader.categories
        return CategoryDatasets.get_all_categories()
    
    def _load_data(self, category: str, split: str = "train") -> Tuple[List[str], List[int]]:
        """Load data for a category, using real data if available."""
        if self.use_real_data and category in self.data_loader.categories:
            texts, labels = self.data_loader.load_category(
                category, split, self.max_samples_per_class
            )
            if texts:
                return texts, labels
            print(f"  Falling back to hardcoded data for '{category}'")
        
        # Fallback to hardcoded
        dataset = CategoryDatasets.get_dataset(category)
        texts = dataset["positive"] + dataset["negative"]
        labels = [1] * len(dataset["positive"]) + [0] * len(dataset["negative"])
        return texts, labels
    
    def collect_activations(
        self, 
        texts: List[str], 
        layer_idx: int,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Collect activations from a specific layer for given texts."""
        layer_module = self.layer_detector.get_layer(layer_idx)
        if layer_module is None:
            raise ValueError(f"Layer {layer_idx} not found")
        
        activations = []
        captured = {}
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                # Find the first tensor in the tuple (hidden states)
                for item in output:
                    if isinstance(item, torch.Tensor):
                        captured['act'] = item.detach()
                        break
            elif isinstance(output, torch.Tensor):
                captured['act'] = output.detach()
            elif hasattr(output, 'last_hidden_state'):
                captured['act'] = output.last_hidden_state.detach()
            elif hasattr(output, 'hidden_states') and output.hidden_states is not None:
                captured['act'] = output.hidden_states[-1].detach()
        
        handle = layer_module.register_forward_hook(hook)
        
        try:
            iterator = range(0, len(texts), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc=f"Layer {layer_idx}", leave=False)
            
            for start_idx in iterator:
                batch_texts = texts[start_idx:start_idx + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                with torch.no_grad():
                    self.model(**inputs)
                
                # Get last token activation for each item in batch
                if 'act' not in captured:
                    raise RuntimeError(
                        f"Forward hook on layer {layer_idx} "
                        f"({type(layer_module).__name__}) never fired. "
                        f"Check model architecture or torch.compile usage."
                    )
                batch_acts = captured['act']  # (batch, seq, hidden)
                
                # Handle attention mask for proper last token
                if "attention_mask" in inputs:
                    seq_lens = inputs["attention_mask"].sum(dim=1) - 1
                    for i in range(len(batch_texts)):
                        act = batch_acts[i, seq_lens[i], :].cpu().numpy()
                        activations.append(act)
                else:
                    for i in range(len(batch_texts)):
                        act = batch_acts[i, -1, :].cpu().numpy()
                        activations.append(act)
        finally:
            handle.remove()
        
        return np.array(activations)
    
    def train_probe(
        self,
        category: str,
        layer_idx: Optional[int] = None,
        cv_folds: int = 5,
        verbose: bool = True
    ) -> TrainedProbe:
        """
        Train a linear probe for a specific category.
        
        Args:
            category: Category name (e.g., "overconfidence")
            layer_idx: Which layer to probe (default: middle layer)
            cv_folds: Number of cross-validation folds
            verbose: Print progress
            
        Returns:
            TrainedProbe object
        """
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn is required for probe training")
        
        if layer_idx is None:
            layer_idx = self.layer_detector.get_num_layers() // 2
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training '{category}' Probe (Layer {layer_idx})")
            print(f"{'='*60}")
        
        # Load data
        train_texts, train_labels = self._load_data(category, "train")
        val_texts, val_labels = self._load_data(category, "val")
        
        if not train_texts:
            raise ValueError(f"No training data for category '{category}'")
        
        # Check balance
        train_labels_arr = np.array(train_labels)
        n_pos = np.sum(train_labels_arr == 1)
        n_neg = np.sum(train_labels_arr == 0)
        
        if verbose:
            print(f"  Train: {len(train_texts)} samples ({n_pos} pos, {n_neg} neg)")
            if val_texts:
                print(f"  Val: {len(val_texts)} samples")
        
        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"Imbalanced data: {n_pos} positive, {n_neg} negative")
        
        # Collect activations
        if verbose:
            print(f"  Collecting train activations...")
        train_acts = self.collect_activations(train_texts, layer_idx)
        
        if val_texts:
            if verbose:
                print(f"  Collecting val activations...")
            val_acts = self.collect_activations(val_texts, layer_idx, show_progress=False)
            val_labels_arr = np.array(val_labels)
        else:
            val_acts = None
            val_labels_arr = None
        
        # Normalize
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_acts)
        if val_acts is not None:
            val_scaled = scaler.transform(val_acts)
        
        # Train based on method
        if verbose:
            print(f"  Training {self.method}...")
        
        if self.method == "logistic_regression":
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
            clf.fit(train_scaled, train_labels_arr)
            direction = clf.coef_[0].copy()
            bias = float(clf.intercept_[0])
        
        elif self.method == "svm":
            clf = LinearSVC(max_iter=2000, C=1.0, random_state=42)
            clf.fit(train_scaled, train_labels_arr)
            direction = clf.coef_[0].copy()
            bias = float(clf.intercept_[0])
        
        elif self.method == "mean_diff":
            pos_mean = train_scaled[train_labels_arr == 1].mean(axis=0)
            neg_mean = train_scaled[train_labels_arr == 0].mean(axis=0)
            direction = pos_mean - neg_mean
            
            # Compute bias as midpoint
            pos_proj = np.mean(train_scaled[train_labels_arr == 1] @ direction)
            neg_proj = np.mean(train_scaled[train_labels_arr == 0] @ direction)
            bias = -(pos_proj + neg_proj) / 2
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Normalize direction
        norm = np.linalg.norm(direction) + 1e-8
        direction = direction / norm
        bias = bias / norm
        
        # Cross-validation on training data
        if self.method in ["logistic_regression", "svm"]:
            cv_clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
            cv_scores = cross_val_score(cv_clf, train_scaled, train_labels_arr, cv=min(cv_folds, len(train_labels_arr) // 2))
        else:
            cv_scores = []
        
        # Evaluate
        train_preds = (train_scaled @ direction + bias) > 0
        train_acc = accuracy_score(train_labels_arr, train_preds)
        train_f1 = f1_score(train_labels_arr, train_preds, average='binary')
        
        if val_acts is not None:
            val_scores = val_scaled @ direction + bias
            val_preds = val_scores > 0
            val_acc = accuracy_score(val_labels_arr, val_preds)
            val_f1 = f1_score(val_labels_arr, val_preds, average='binary')
            try:
                val_auc = roc_auc_score(val_labels_arr, val_scores)
            except:
                val_auc = 0.5
        else:
            val_acc, val_f1, val_auc = train_acc, train_f1, 0.5
        
        if verbose:
            print(f"\n  Results:")
            print(f"    Train Accuracy: {train_acc:.4f}")
            print(f"    Train F1:       {train_f1:.4f}")
            if cv_scores.any():
                print(f"    CV Accuracy:    {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            if val_acts is not None:
                print(f"    Val Accuracy:   {val_acc:.4f}")
                print(f"    Val F1:         {val_f1:.4f}")
                print(f"    Val AUC-ROC:    {val_auc:.4f}")
        
        # Get description
        if self.use_real_data:
            description = f"Probe for {category} (trained on real data)"
        else:
            dataset = CategoryDatasets.get_dataset(category)
            description = dataset.get("description", f"Probe for {category}")
        
        # Create probe
        probe = TrainedProbe(
            category=category,
            description=description,
            layer_idx=layer_idx,
            hidden_size=self.hidden_size,
            direction=direction,
            bias=bias,
            accuracy=val_acc if val_acts is not None else train_acc,
            f1_score=val_f1 if val_acts is not None else train_f1,
            auc_roc=val_auc,
            cv_scores=cv_scores.tolist() if hasattr(cv_scores, 'tolist') else [],
            method=self.method,
        )
        
        self.probes[category] = probe
        return probe
    
    def train_all_probes(
        self,
        layer_idx: Optional[int] = None,
        categories: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, TrainedProbe]:
        """Train probes for all (or specified) categories."""
        if categories is None:
            categories = self.get_available_categories()
        
        if verbose:
            print(f"\nTraining {len(categories)} probes...")
            print(f"Categories: {categories}")
        
        for category in categories:
            try:
                self.train_probe(category, layer_idx, verbose=verbose)
            except Exception as e:
                print(f"  Error training '{category}': {e}")
        
        return self.probes
    
    def find_best_layer(
        self,
        category: str,
        layer_range: Optional[Tuple[int, int]] = None,
        verbose: bool = True
    ) -> Tuple[int, float]:
        """Find the layer with best probe accuracy."""
        num_layers = self.layer_detector.get_num_layers()
        
        if layer_range is None:
            start = num_layers // 5
            end = num_layers - num_layers // 5
        else:
            start, end = layer_range
        
        if verbose:
            print(f"\nSearching layers {start}-{end} for '{category}'...")
        
        best_layer = start
        best_score = 0.0
        
        for layer_idx in range(start, end):
            probe = self.train_probe(category, layer_idx, verbose=False)
            score = probe.f1_score
            
            if verbose:
                print(f"  Layer {layer_idx}: F1={score:.4f}, Acc={probe.accuracy:.4f}")
            
            if score > best_score:
                best_score = score
                best_layer = layer_idx
        
        if verbose:
            print(f"\n  Best: Layer {best_layer} (F1={best_score:.4f})")
        
        return best_layer, best_score
    
    def save_probes(self, filepath: str):
        """Save all trained probes to a pickle file (dashboard compatible)."""
        # Resolve relative paths against the project root (llm_monitor/)
        filepath = Path(filepath)
        if not filepath.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            filepath = project_root / filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath = str(filepath)

        # Single .pkl with both dashboard-compatible fields and full metadata
        data = {}
        for category, probe in self.probes.items():
            data[category] = {
                "direction": probe.direction,
                "layer_idx": probe.layer_idx,
                "bias": probe.bias,
                "coef": probe.direction,  # Alias for dashboard
                "metadata": probe.to_dict(),
            }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        # JSON report
        json_filepath = filepath.replace('.pkl', '.json')
        json_data = {
            "model": str(self.model.config._name_or_path if hasattr(self.model.config, '_name_or_path') else "unknown"),
            "method": self.method,
            "num_probes": len(self.probes),
            "probes": {cat: probe.to_dict() for cat, probe in self.probes.items()}
        }
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"\nSaved {len(self.probes)} probes:")
        print(f"  Probes:  {filepath}")
        print(f"  Report:  {json_filepath}")
    
    @classmethod
    def load_probes(cls, filepath: str) -> Dict[str, TrainedProbe]:
        """Load probes from a pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        probes = {}
        for category, probe_data in data.items():
            if "metadata" in probe_data:
                meta = probe_data["metadata"]
                probes[category] = TrainedProbe(
                    category=meta["category"],
                    description=meta.get("description", ""),
                    layer_idx=meta["layer_idx"],
                    hidden_size=meta["hidden_size"],
                    direction=np.array(probe_data["direction"]) if probe_data["direction"] else None,
                    bias=probe_data.get("bias", meta.get("bias", 0.0)),
                    accuracy=meta.get("accuracy", 0.0),
                    f1_score=meta.get("f1_score", 0.0),
                )
            else:
                # Simpler format
                probes[category] = TrainedProbe(
                    category=category,
                    description="",
                    layer_idx=probe_data.get("layer_idx", 12),
                    hidden_size=len(probe_data["direction"]) if probe_data.get("direction") is not None else 0,
                    direction=np.array(probe_data["direction"]) if probe_data.get("direction") is not None else None,
                    bias=probe_data.get("bias", 0.0),
                )
        
        return probes


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Linear Probes on Real Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on real dataset
    python probe_trainer_real.py --model Qwen/Qwen2.5-0.5B-Instruct --data ./probe_real_dataset
    
    # Specific categories
    python probe_trainer_real.py --model Qwen/Qwen2.5-0.5B-Instruct --data ./probe_real_dataset \\
        --categories overconfidence sycophancy toxicity
    
    # Find best layer
    python probe_trainer_real.py --model Qwen/Qwen2.5-0.5B-Instruct --data ./probe_real_dataset \\
        --find-best-layer --categories overconfidence
    
    # Different method
    python probe_trainer_real.py --model Qwen/Qwen2.5-0.5B-Instruct --data ./probe_real_dataset \\
        --method mean_diff
        """
    )
    
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name/path")
    parser.add_argument("--data", "-d", type=str, default="./probe_real_dataset", help="Dataset directory")
    parser.add_argument("--output", "-o", type=str, default="trained_probes.pkl", help="Output file")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda:0, cpu, auto)")
    parser.add_argument("--categories", "-c", nargs="+", default=None, help="Categories to train")
    parser.add_argument("--layer", "-l", type=int, default=None, help="Target layer (default: middle)")
    parser.add_argument("--method", choices=["logistic_regression", "svm", "mean_diff"],
                       default="logistic_regression", help="Training method")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples per class")
    parser.add_argument("--find-best-layer", action="store_true", help="Search for best layer")
    parser.add_argument("--list-categories", action="store_true", help="List available categories")
    
    args = parser.parse_args()
    
    # List categories mode
    if args.list_categories:
        loader = RealDatasetLoader(args.data)
        print(f"\nAvailable categories in {args.data}:")
        for cat in loader.categories:
            stats = loader.get_dataset_stats(cat)
            if "error" not in stats:
                print(f"  {cat:20s} - {stats['total_samples']} samples "
                      f"({stats['positive_samples']} pos, {stats['negative_samples']} neg)")
            else:
                print(f"  {cat:20s} - {stats['error']}")
        return
    
    # Device
    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map=device if "cuda" in device else None,
        trust_remote_code=True,
    )
    if "cuda" not in device:
        model = model.to(device)
    model.eval()
    
    # Create trainer
    trainer = MultiProbeTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        data_dir=args.data,
        max_samples_per_class=args.max_samples,
        method=args.method,
    )
    
    # Determine categories
    if args.categories:
        categories = args.categories
    else:
        categories = trainer.get_available_categories()
    
    print(f"\nWill train: {categories}")
    
    # Train
    if args.find_best_layer:
        for cat in categories:
            best_layer, best_score = trainer.find_best_layer(cat)
            trainer.train_probe(cat, best_layer)
    else:
        trainer.train_all_probes(layer_idx=args.layer, categories=categories)
    
    # Save
    trainer.save_probes(args.output)
    
    # Summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    if trainer.probes:
        print("\nResults:")
        for cat, probe in sorted(trainer.probes.items(), key=lambda x: -x[1].f1_score):
            print(f"  {cat:20s}: F1={probe.f1_score:.4f}, Acc={probe.accuracy:.4f}, Layer={probe.layer_idx}")
    
    print(f"\nOutput: {args.output}")
    print(f"Usage:  python dashboard_server.py --probe-path {args.output}")


if __name__ == "__main__":
    main()