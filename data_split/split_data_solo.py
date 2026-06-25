#!/usr/bin/env python3

import ast
from pathlib import Path
import pandas as pd

# ========== CONFIG ==========
meta_csv = Path("audio_segments.csv") 
root_solo = Path("openunmix_solo_generated")
# ============================

out_train = root_solo / "SOLO_train_segments.csv"
out_valid = root_solo / "SOLO_valid_segments.csv"
out_test  = root_solo / "SOLO_test_segments.csv"


def folder_name_from_rel_path(rel_path: str) -> str:
    """
    Convert: 1/GB/ro/.../vocal.ogg  ->  1-GB-ro-...-vocal
    Must match how you named output folders when generating the solo dataset.
    """
    return rel_path.strip().replace("/", "-").replace(".ogg", "")


def format_segments(seg_list):
    # seg_list is like [(1.35, 4.61), (10.68, 15.29)]
    return ", ".join([f"({s}, {e})" for s, e in seg_list])


def main():
    df = pd.read_csv(meta_csv)

    records = []
    for _, row in df.iterrows():
        rel = str(row["file_path"]).strip()
        seg_str = row["segments"]

        try:
            seg_list = ast.literal_eval(seg_str)
        except Exception as e:
            print(f"⚠️ Skipping malformed segments for {rel}: {e}")
            continue

        if not isinstance(seg_list, list) or len(seg_list) == 0:
            continue

        folder = folder_name_from_rel_path(rel)

        found = None
        for split in ["train", "valid", "test"]:
            candidate = root_solo / split / folder / "vocal1.wav"
            if candidate.exists():
                found = candidate
                break

        if not found:
            print(f"⚠️ Could not find {folder} under train/valid/test, skipping.")
            continue

        records.append({
            "file_path": str(found),
            "segments": format_segments(seg_list),
        })

    df_out = pd.DataFrame(records)

    train_df = df_out[df_out["file_path"].str.contains("/train/")]
    valid_df = df_out[df_out["file_path"].str.contains("/valid/")]
    test_df  = df_out[df_out["file_path"].str.contains("/test/")]

    train_df.to_csv(out_train, index=False)
    valid_df.to_csv(out_valid, index=False)
    test_df.to_csv(out_test, index=False)

    print(f"✅ Saved {len(train_df)} rows to {out_train}")
    print(f"✅ Saved {len(valid_df)} rows to {out_valid}")
    print(f"✅ Saved {len(test_df)} rows to {out_test}")


if __name__ == "__main__":
    main()