import os
import random
import pandas as pd
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

# ---------- helpers ----------

def convert_to_wav(src_path, dst_path):
    """Convert any audio file to wav format, 44.1kHz mono."""
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_frame_rate(44100).set_channels(1)
    audio.export(dst_path, format="wav")
    return dst_path

def normalize_with_offset(audio, target_level=-20.0, offset_db=0.0):
    """
    Normalize audio to target dBFS, then apply an optional offset.
    """
    change_in_dBFS = target_level - audio.dBFS
    return audio.apply_gain(change_in_dBFS + offset_db)

def mix_tracks(vocal1_audio, vocal2_audio, bg_audio, target_level=-20.0, bg_offset=-6.0):
    """Mix three tracks with balanced volumes."""
    # Normalize vocals
    v1 = normalize_with_offset(vocal1_audio, target_level)
    v2 = normalize_with_offset(vocal2_audio, target_level)

    # Normalize background, then make it quieter than vocals
    bg = normalize_with_offset(bg_audio, target_level, offset_db=bg_offset)

    # Pad to max length
    max_len = max(len(bg), len(v1), len(v2))
    def pad(audio): return audio + AudioSegment.silent(duration=max_len - len(audio))
    bg, v1, v2 = map(pad, (bg, v1, v2))

    return bg.overlay(v1).overlay(v2)

# ---------- main pipeline ----------

def generate_mixtures(csv_path, output_root, num_random=9, seed=42):
    random.seed(seed)
    df = pd.read_csv(csv_path)

    all_vocal2s = df["vocal2_path"].dropna().tolist()
    base_dir = "/mnt/data/damp-vsep"
    
    # for _, row in df.iterrows():
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing vocal1 anchors"):
        vocal1_path = row["vocal1_path"]
        vocal2_path = row["vocal2_path"]

        vocal1_full = os.path.join(base_dir, vocal1_path)
        # vocal2_full = os.path.join(base_dir, vocal2_path)
        bg_path = os.path.join(os.path.dirname(vocal1_full), "background.m4a")

        # ---- anchor folder ----
        v1_name = vocal1_path.replace("/", "-").replace(".ogg", "")
        anchor_dir = os.path.join(output_root, v1_name)
        os.makedirs(anchor_dir, exist_ok=True)

        # Convert and load vocal1 + background
        v1_wav = os.path.join(anchor_dir, "vocal1.wav")
        bg_wav = os.path.join(anchor_dir, "background.wav")
        if not os.path.exists(v1_wav):
            convert_to_wav(vocal1_full, v1_wav)
        if not os.path.exists(bg_wav):
            convert_to_wav(bg_path, bg_wav)
        # convert_to_wav(vocal1_full, v1_wav)
        # convert_to_wav(bg_path, bg_wav)

        v1_audio = AudioSegment.from_file(v1_wav)
        bg_audio = AudioSegment.from_file(bg_wav)

        # ---- choose partners ----
        partners = [vocal2_path]  # include true partner
        candidates = [p for p in all_vocal2s if p not in (vocal2_path, vocal1_path)]
        random_partners = random.sample(candidates, min(num_random, len(candidates)))
        partners.extend(random_partners)

        # ---- generate mixtures ----
        for partner in partners:
            partner_full = os.path.join(base_dir, partner)
            partner_name = partner.replace("/", "-").replace(".ogg", "")
            partner_dir = os.path.join(anchor_dir, partner_name)
            os.makedirs(partner_dir, exist_ok=True)

            mix_path = os.path.join(partner_dir, "mixture.wav")

            if os.path.exists(mix_path):
                # print(f"Skipping {mix_path}, already exists.")
                continue

            v2_wav = os.path.join(partner_dir, "vocal2.wav")
            if not os.path.exists(v2_wav):
                convert_to_wav(partner_full, v2_wav)
            v2_audio = AudioSegment.from_file(v2_wav)

            mixture = mix_tracks(v1_audio, v2_audio, bg_audio)
            mixture.export(mix_path, format="wav")

        # break  # remove this break to process all rows
    
def main():
    csv_path = "/home/yuex7/research/audio_segments_DUET.csv"
    output_root = "/home/yuex7/openunmix_duet_extra"
    generate_mixtures(csv_path, output_root, num_random=9)

if __name__ == "__main__":
    main()