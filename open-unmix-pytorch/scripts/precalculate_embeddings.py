import os
import ast
import json
import argparse
import tqdm
import torch
import pandas as pd
import numpy as np
import Customized_NetSV as M 
from typing import Tuple
import librosa as lb
from pathlib import Path

SEGMENT_LENGTH = 3.0

# ---------------------------------------------------------------------
# Load singer embedding model (NetSV)
# ---------------------------------------------------------------------
def load_singer_embedding_model(device, model_path: str):
    """Load the singer embedding model and move to device."""
    hidden_size = 32
    num_layers = 2
    model = M.NetSV(hidden_size, num_layers)
    print(f"✅ Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------
# Load segment and randomly pick a 3s slice
# ---------------------------------------------------------------------
def load_segment(file_path: str, segment: Tuple[float, float]) -> Tuple[torch.Tensor, Tuple[float, float]]:
    audio_data, sr = lb.load(file_path, sr=None, mono=True)
    start_time, end_time = segment
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    segment_length = end_sample - start_sample
    if segment_length < int(SEGMENT_LENGTH * sr):
        raise ValueError(f"Segment too short in {file_path}: {segment}")

    max_start = segment_length - int(SEGMENT_LENGTH * sr)
    random_start = np.random.randint(0, max_start + 1)

    segment_start_time = (start_sample + random_start) / sr
    segment_end_time = segment_start_time + SEGMENT_LENGTH

    segment_data = audio_data[
        start_sample + random_start : start_sample + random_start + int(SEGMENT_LENGTH * sr)
    ]
    segment_data = torch.Tensor(segment_data / np.max(np.abs(segment_data)))
    return segment_data, (segment_start_time, segment_end_time)


# ---------------------------------------------------------------------
# Load precomputed segment ranges (from your precalculated_segments_*.csv)
# ---------------------------------------------------------------------
def load_segments_from_csv(filename):
    df = pd.read_csv(filename)
    df["segments"] = df["segments"].apply(ast.literal_eval)
    return dict(zip(df["file_path"], df["segments"]))


# ---------------------------------------------------------------------
# Helper for relative path storage
# ---------------------------------------------------------------------
def extract_relative_path(file_path):
    parts = file_path.split("/")
    for i, part in enumerate(parts):
        if part in ["train", "valid", "test"]:
            return "/".join(parts[i+1:-1])
    return None


# ---------------------------------------------------------------------
# Precalculate embeddings and save + record used segments
# ---------------------------------------------------------------------
def precalculate_and_save_embeddings(split_name, file_paths, embedding_model, precalculated_segments, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    embeddings_dict = {}

    print(f"\n🚀 Starting embedding generation for {split_name} ({len(file_paths)} files)")

    for file_path in tqdm.tqdm(file_paths, desc=f"{split_name} embeddings"):
        all_segments = precalculated_segments.get(file_path)
        if not all_segments:
            continue
        
        # Randomly pick one segment from list
        if isinstance(all_segments[0], tuple) :
            segment = all_segments[np.random.randint(0, len(all_segments))]
        else:
            segment = all_segments
            
        try:
            segment_data, used_segment_time = load_segment(file_path, segment)
        except Exception as e:
            print(f"⚠️ Skipping {file_path}: {e}")
            continue

        segment_data = segment_data.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = embedding_model.embedding(segment_data)

        relative_path = extract_relative_path(file_path)
        embedding_dir = os.path.join(save_dir, relative_path)
        os.makedirs(embedding_dir, exist_ok=True)

        embedding_path = os.path.join(embedding_dir, "vocal.pt")
        torch.save(embedding.cpu(), embedding_path)

        embeddings_dict[file_path] = {
            "embedding_path": embedding_path,
            "used_segment_time": used_segment_time
        }

    # Save mapping JSON
    json_path = os.path.join(save_dir, f"embeddings_mapping_{split_name}.json")
    with open(json_path, "w") as f:
        json.dump(embeddings_dict, f, indent=2)
    print(f"✅ Saved {len(embeddings_dict)} embeddings for {split_name} → {json_path}")
    return embeddings_dict


# ---------------------------------------------------------------------
# Run for train, valid, and test
# ---------------------------------------------------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Precompute DUET singer embeddings.")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=script_dir,
        help="Directory containing DUET_*_segments.csv files."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to singer embedding checkpoint (.pth)."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir / "embeddings" / "DUET_clean_embeddings",
        help="Directory where precomputed embeddings are written."
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = load_singer_embedding_model(device, str(args.model_path))

    base_dir = args.base_dir
    segment_files = {
        "train": base_dir / "DUET_train_segments.csv",
        "valid": base_dir / "DUET_valid_segments.csv",
        "test": base_dir / "DUET_test_segments.csv",
    }

    output_root = args.output_root

    for split, csv_path in segment_files.items():
        precalculated_segments = load_segments_from_csv(csv_path)
        df = pd.read_csv(csv_path)
        file_paths = df["file_path"].to_list()
        precalculate_and_save_embeddings(
            split, file_paths, embedding_model, precalculated_segments, device, os.path.join(str(output_root), split)
        )
