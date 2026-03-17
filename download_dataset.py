#!/usr/bin/env python3
"""
Minimal Dataset Downloader - Run this directly on your remote server
"""
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import os
import pandas as pd

OUTPUT_DIR = Path("./probe_real_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

def download_sycophancy():
    """Download Anthropic sycophancy data (~30k samples)"""
    print("\n" + "="*50)
    print("Downloading: Sycophancy")
    print("="*50)
    
    samples = []
    files = [
        "sycophancy/sycophancy_on_nlp_survey.jsonl",
        "sycophancy/sycophancy_on_philpapers2020.jsonl",
        "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
    ]
    
    for f in files:
        print(f"  Loading {f}...")
        try:
            ds = load_dataset("Anthropic/model-written-evals", data_files=f, split="train")
            for item in tqdm(ds):
                samples.append({
                    "text": item.get("question", ""),
                    "answer": item.get("answer_matching_behavior", ""),
                })
        except Exception as e:
            print(f"    Error: {e}")
    
    with open(OUTPUT_DIR / "sycophancy.json", 'w') as f:
        json.dump(samples, f)
    print(f"  Saved {len(samples)} samples")


def download_sentiment():
    """Download IMDB sentiment data (~50k samples)"""
    print("\n" + "="*50)
    print("Downloading: Sentiment (IMDB)")
    print("="*50)
    
    ds = load_dataset("stanfordnlp/imdb")
    
    samples = []
    for split in ["train", "test"]:
        for item in tqdm(ds[split], desc=split):
            samples.append({
                "text": item["text"],
                "label": item["label"],  # 0=negative, 1=positive
            })
    
    with open(OUTPUT_DIR / "sentiment.json", 'w') as f:
        json.dump(samples, f)
    print(f"  Saved {len(samples)} samples")

def convert_jigsaw_to_json(data_dir, output_dir):
    """
    Reads Jigsaw CSVs, merges the separated test labels, and saves as JSONL.
    """
    # Define file paths
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    test_labels_path = os.path.join(data_dir, "test_labels.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Train data...")
    df_train = pd.read_csv(train_path)
    # Save Train
    train_out = os.path.join(output_dir, "train.jsonl")
    df_train.to_json(train_out, orient="records", lines=True)
    print(f"Saved {len(df_train)} training records to {train_out}")

    print("Loading and Merging Test data...")
    df_test = pd.read_csv(test_path)
    df_labels = pd.read_csv(test_labels_path)

    # CRITICAL STEP: Merge test text with test labels on 'id'
    # The Jigsaw dataset separates these, so this step makes the test set usable.
    df_test_combined = pd.merge(df_test, df_labels, on='id')

    # Optional: Filter out rows with -1 labels (which Jigsaw marks as 'ignored' for scoring)
    # df_test_combined = df_test_combined[df_test_combined['toxic'] != -1]

    # Save Test
    test_out = os.path.join(output_dir, "test.jsonl")
    df_test_combined.to_json(test_out, orient="records", lines=True)
    print(f"Saved {len(df_test_combined)} test records to {test_out}")

def download_toxicity():
    """
    Process Jigsaw Toxicity data into clean JSONL for Probe Training.
    - Filters out 'ignored' (-1) labels from the test set.
    - Keeps train and test separate for proper validation.
    """
    print("\n" + "="*50)
    print("Downloading: Toxicity (Jigsaw)")
    print("="*50)

    # Directory where you manually placed the CSVs
    source_dir = Path("./probe_real_dataset/jigsaw-toxic")
    
    # 1. Define the files and how to handle them
    tasks = [
        {"subset": "train", "file": "train.csv", "merge_labels": None},
        {"subset": "test", "file": "test.csv", "merge_labels": "test_labels.csv"}
    ]

    for task in tasks:
        subset = task["subset"]
        print(f"\n  Processing {subset} set...")
        
        csv_path = source_dir / task["file"]
        if not csv_path.exists():
            print(f"  [ERROR] File not found: {csv_path}")
            continue

        # Load CSV
        df = pd.read_csv(csv_path)

        # Special handling for Test split: Merge labels & Filter bad rows
        if task["merge_labels"]:
            label_path = source_dir / task["merge_labels"]
            if label_path.exists():
                print(f"  Merging labels from {task['merge_labels']}...")
                df_labels = pd.read_csv(label_path)
                df = pd.merge(df, df_labels, on='id')
                
                # CRITICAL FOR PROBES: Remove rows with label -1 (disputed/ignored)
                before_len = len(df)
                df = df[df["toxic"] != -1]
                print(f"  Filtered out {before_len - len(df)} rows with label '-1'")

        samples = []
        # The specific toxicity attributes we care about
        label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {subset}"):
            # Create a clean dictionary object
            item = {
                "text": row.get("comment_text", ""),
                "id": row.get("id", ""),
                "labels": {col: int(row[col]) for col in label_cols if col in row}
            }
            # For a simple probe, you might just want the binary "is_toxic" flag
            # item["is_toxic"] = 1 if item["labels"]["toxic"] == 1 else 0
            
            samples.append(item)
        
        # Save as JSON List (standard format)
        filename = f"toxicity_{subset}.json"
        out_path = OUTPUT_DIR / filename
        with open(out_path, 'w') as f:
            json.dump(samples, f)
            
        print(f"  Saved {len(samples)} clean samples to {filename}")

def download_hh_rlhf():
    """Download Anthropic HH-RLHF data (~170k samples)"""
    print("\n" + "="*50)
    print("Downloading: HH-RLHF (Safety & Helpfulness)")
    print("="*50)
    
    for subset in ["harmless-base", "helpful-base"]:
        print(f"\n  Loading {subset}...")
        ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="train")
        
        samples = []
        for item in tqdm(ds, desc=subset):
            samples.append({
                "chosen": item.get("chosen", ""),
                "rejected": item.get("rejected", ""),
            })
        
        filename = f"{subset.replace('-', '_')}.json"
        with open(OUTPUT_DIR / filename, 'w') as f:
            json.dump(samples, f)
        print(f"  Saved {len(samples)} samples to {filename}")

if __name__ == "__main__":
    print("="*60)
    print("DOWNLOADING REAL DATASETS FROM HUGGINGFACE")
    print("="*60)
    
    #download_sycophancy()
    #download_sentiment()
    #download_hh_rlhf()
    data_dir = Path("./probe_real_dataset/jigsaw-toxic")
    output_dir = Path("./probe_real_dataset/jigsaw-toxic")
    download_toxicity()

    
    print("\n" + "="*60)
    print("COMPLETE! Files saved to:", OUTPUT_DIR)
    print("="*60)