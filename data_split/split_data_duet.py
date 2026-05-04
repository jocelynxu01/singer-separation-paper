import pandas as pd
import ast
from pathlib import Path

# Input: your big CSV with vocal1_path and vocal1_segments
csv_path = "audio_segments_DUET.csv"

# Base duet directory (already has train, valid, test subfolders)
root_duet = Path("openunmix_duet_extra_split")

# Output CSVs
output_train = root_duet / "DUET_train_segments.csv"
output_valid = root_duet / "DUET_valid_segments.csv"
output_test = root_duet / "DUET_test_segments.csv"

# Load the big CSV
df = pd.read_csv(csv_path)

records = []

for _, row in df.iterrows():
    vocal1_rel = str(row["vocal1_path"]).strip()
    seg_str = row["vocal1_segments"]

    try:
        segments = ast.literal_eval(seg_str)
    except Exception as e:
        print(f"⚠️ Skipping malformed row: {vocal1_rel} ({e})")
        continue

    # Extract country/language/id pattern to match folder names
    # e.g., 1/BR/ko/330735398_2737582511/vocal.ogg → 1-BR-ko-330735398_2737582511-vocal
    folder_name = (
        vocal1_rel.replace("/", "-")
        .replace(".ogg", "")
        .replace("-vocal", "_vocal")
        .replace("_vocal", "-vocal")
    )

    # Search for where this folder actually exists
    found_path = None
    for split in ["train", "valid", "test"]:
        candidate = root_duet / split / folder_name / "vocal1.wav"
        if candidate.exists():
            found_path = candidate
            break

    if not found_path:
        print(f"⚠️ Could not match {folder_name}, skipping.")
        continue

    # Format segments
    if isinstance(segments, list) and len(segments) > 0:
        seg_formatted = ", ".join([f"({round(s[0], 2)}, {round(s[1], 2)})" for s in segments])
        records.append({
            "file_path": str(found_path),
            "segments": seg_formatted
        })

# Convert to DataFrame
df_out = pd.DataFrame(records)

# Split by folder (since we detected actual split)
train_df = df_out[df_out["file_path"].str.contains("/train/")]
valid_df = df_out[df_out["file_path"].str.contains("/valid/")]
test_df = df_out[df_out["file_path"].str.contains("/test/")]

# Save them
train_df.to_csv(output_train, index=False)
valid_df.to_csv(output_valid, index=False)
test_df.to_csv(output_test, index=False)

print(f"✅ Saved {len(train_df)} rows to {output_train}")
print(f"✅ Saved {len(valid_df)} rows to {output_valid}")
print(f"✅ Saved {len(test_df)} rows to {output_test}")