import os
import random
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

# ---------- helpers ----------

def load_audio_tensor(path, sample_rate=44100, device='cuda'):
    """Load audio as tensor [1, num_samples], resample to 44.1kHz mono."""
    waveform, sr = torchaudio.load(path)  # [channels, samples]
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # convert to mono
    return waveform.to(device)

def normalize_tensor(waveform, target_db=-20.0, offset_db=0.0):
    """Normalize waveform to target dBFS with optional offset."""
    rms = torch.sqrt(torch.mean(waveform ** 2))
    current_db = 20 * torch.log10(rms + 1e-9)
    gain_db = target_db - current_db + offset_db
    factor = 10 ** (gain_db / 20)
    return waveform * factor

def pad_to_max_length(tensors):
    """Pad all tensors to max length along time axis."""
    max_len = max(t.shape[1] for t in tensors)
    padded = [torch.nn.functional.pad(t, (0, max_len - t.shape[1])) for t in tensors]
    return padded

def save_tensor(waveform, path, sample_rate=44100):
    """Save tensor waveform as WAV."""
    torchaudio.save(path, waveform.cpu(), sample_rate)

def mix_tensors(v1, v2, bg, target_db=-20.0, bg_offset=-6.0):
    """Mix vocal1, vocal2, and background tensors on GPU."""
    v1 = normalize_tensor(v1, target_db)
    v2 = normalize_tensor(v2, target_db)
    bg = normalize_tensor(bg, target_db, offset_db=bg_offset)

    v1, v2, bg = pad_to_max_length([v1, v2, bg])
    mixture = bg + v1 + v2
    mixture = torch.clamp(mixture, -1.0, 1.0)  # prevent clipping
    return mixture

# ---------- main pipeline ----------

def generate_mixtures(csv_path, output_root, num_random=9, seed=42, device='cuda'):
    random.seed(seed)
    df = pd.read_csv(csv_path)
    all_vocal2s = df["vocal2_path"].dropna().tolist()
    base_dir = "/mnt/data/damp-vsep"

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing vocal1 anchors"):
        vocal1_path = row["vocal1_path"]
        vocal2_path = row["vocal2_path"]

        vocal1_full = os.path.join(base_dir, vocal1_path)
        bg_path = os.path.join(os.path.dirname(vocal1_full), "background.m4a")

        # Anchor folder
        v1_name = vocal1_path.replace("/", "-").replace(".ogg", "")
        anchor_dir = os.path.join(output_root, v1_name)
        os.makedirs(anchor_dir, exist_ok=True)

        # Load audio tensors
        v1_tensor = load_audio_tensor(vocal1_full, device=device)
        bg_tensor = load_audio_tensor(bg_path, device=device)
        
        # Save normalized individual files
        v1_save_path = os.path.join(anchor_dir, "vocal1.wav")
        bg_save_path = os.path.join(anchor_dir, "background.wav")
        if not os.path.exists(v1_save_path):
            save_tensor(normalize_tensor(v1_tensor), v1_save_path)
        if not os.path.exists(bg_save_path):
            save_tensor(normalize_tensor(bg_tensor, offset_db=-6.0), bg_save_path)

        # Partners
        partners = [vocal2_path]
        candidates = [p for p in all_vocal2s if p not in (vocal2_path, vocal1_path)]
        random_partners = random.sample(candidates, min(num_random, len(candidates)))
        partners.extend(random_partners)

        for partner in partners:
            partner_full = os.path.join(base_dir, partner)
            partner_name = partner.replace("/", "-").replace(".ogg", "")
            partner_dir = os.path.join(anchor_dir, partner_name)
            os.makedirs(partner_dir, exist_ok=True)

            mix_path = os.path.join(partner_dir, "mixture.wav")
            if os.path.exists(mix_path):
                continue  # skip already done

            v2_tensor = load_audio_tensor(partner_full, device=device)
            v2_save_path = os.path.join(partner_dir, "vocal2.wav")
            if not os.path.exists(v2_save_path):
                save_tensor(normalize_tensor(v2_tensor), v2_save_path)
            mixture_tensor = mix_tensors(v1_tensor, v2_tensor, bg_tensor)

            save_tensor(mixture_tensor, mix_path)

# ---------- run ----------

def main():
    csv_path = "/home/yuex7/research/audio_segments_DUET.csv"
    output_root = "/home/yuex7/openunmix_duet_extra"
    generate_mixtures(csv_path, output_root, num_random=9, device='cuda')

if __name__ == "__main__":
    main()