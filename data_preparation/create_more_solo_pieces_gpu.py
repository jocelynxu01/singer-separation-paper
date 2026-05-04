#!/usr/bin/env python3

import os
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

# -------------------- CONFIG --------------------

BASE_DIR = "/mnt/data/damp-vsep"   # root where your dataset lives
OUTPUT_ROOT = "/home/yuex7/openunmix_solo_generated"
SAMPLE_RATE = 44100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------


# ---------- helpers ----------

def load_audio_tensor(path):
    waveform, sr = torchaudio.load(path)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    # convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.to(DEVICE)


def normalize_tensor(waveform, target_db=-20.0, offset_db=0.0):
    rms = torch.sqrt(torch.mean(waveform ** 2))
    current_db = 20 * torch.log10(rms + 1e-9)
    gain_db = target_db - current_db + offset_db
    factor = 10 ** (gain_db / 20)
    return waveform * factor


def pad_to_same_length(a, b):
    max_len = max(a.shape[1], b.shape[1])
    a = torch.nn.functional.pad(a, (0, max_len - a.shape[1]))
    b = torch.nn.functional.pad(b, (0, max_len - b.shape[1]))
    return a, b


def save_tensor(waveform, path):
    torchaudio.save(path, waveform.cpu(), SAMPLE_RATE)


# ---------- main pipeline ----------

def generate_solo_dataset(csv_path):
    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating solo mixtures"):
        relative_path = row["file_path"]

        vocal_path = os.path.join(BASE_DIR, relative_path)
        folder = os.path.dirname(vocal_path)

        # assuming background is in same folder
        background_path = os.path.join(folder, "background.m4a")

        if not os.path.exists(background_path):
            print(f"Background not found for {vocal_path}")
            continue

        # Create output folder
        sample_name = relative_path.replace("/", "-").replace(".ogg", "")
        out_dir = os.path.join(OUTPUT_ROOT, sample_name)
        os.makedirs(out_dir, exist_ok=True)

        mixture_out = os.path.join(out_dir, "mixture.wav")
        if os.path.exists(mixture_out):
            continue  # skip if already processed

        # Load audio
        vocal = load_audio_tensor(vocal_path)
        background = load_audio_tensor(background_path)

        # Normalize
        vocal = normalize_tensor(vocal, target_db=-20)
        background = normalize_tensor(background, target_db=-20, offset_db=-6)

        # Pad and mix
        vocal, background = pad_to_same_length(vocal, background)
        mixture = torch.clamp(vocal + background, -1.0, 1.0)

        # Save
        save_tensor(vocal, os.path.join(out_dir, "vocal1.wav"))
        save_tensor(background, os.path.join(out_dir, "background.wav"))
        save_tensor(mixture, mixture_out)

# ---------- run ----------

def main():
    csv_path = "/home/yuex7/research/audio_segments.csv"
    generate_solo_dataset(csv_path)


if __name__ == "__main__":
    main()