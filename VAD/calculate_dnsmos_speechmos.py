from joblib import Parallel, delayed
import pandas as pd
import librosa as lb
from tqdm import tqdm
from speechmos import dnsmos
import numpy as np
import os

EXPECTED_KEYS_1 = ["vocal1_ovrl_mos","vocal1_sig_mos","vocal1_bak_mos","vocal1_p808_mos"]
EXPECTED_KEYS_2 = ["vocal2_ovrl_mos","vocal2_sig_mos","vocal2_bak_mos","vocal2_p808_mos"]

def normalize_audio(y):
    """Normalize audio data to the range [-1, 1]."""
    return y / np.max(np.abs(y))

def process_audio(row):
    vocal1, vocal2 = row.vocal1_path, row.vocal2_path
    
    y_1, sr = lb.load(f"/mnt/data/damp-vsep/{vocal1}", sr=16000)
    y_2, sr = lb.load(f"/mnt/data/damp-vsep/{vocal2}", sr=16000)
    
    if np.all(y_1 == 0):
        voice_quality_1 = {key: 0 for key in EXPECTED_KEYS_1}
    else:
        if np.max(np.abs(y_1)) > 1.0:
            y_1 = normalize_audio(y_1)
        voice_quality_1 = dnsmos.run(y_1, sr=16000)

    if np.all(y_2 == 0):
        voice_quality_2 = {key: 0 for key in EXPECTED_KEYS_2
                           }
    else:
        if np.max(np.abs(y_2)) > 1.0:
            y_2 = normalize_audio(y_2)
        voice_quality_2 = dnsmos.run(y_2, sr=16000)

    result = {
        "vocal1_path": vocal1,
        "vocal2_path": vocal2,
    }

    for key, value in voice_quality_1.items():
        result[f"vocal1_{key}"] = value
    for key, value in voice_quality_2.items():
        result[f"vocal2_{key}"] = value

    return result

def write_result_to_csv(result, output_csv_path):
    """Write a single result to the CSV file."""
    result_df = pd.DataFrame([result])
    result_df.to_csv(output_csv_path, mode='a', header=False, index=False)

def voice_quality_estimation(audio_files, output_csv_path, n_jobs=-1):
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(output_csv_path)
    print('Directory created')
    
    # Write the header before starting parallel processing
    if not os.path.isfile(output_csv_path):
        header_df = pd.DataFrame(columns=[
            "vocal1_path", "vocal2_path",
            "vocal1_ovrl_mos", "vocal1_sig_mos", "vocal1_bak_mos", "vocal1_p808_mos",
            "vocal2_ovrl_mos", "vocal2_sig_mos", "vocal2_bak_mos", "vocal2_p808_mos"
        ])
        header_df.to_csv(output_csv_path, index=False)
        print("Header written to CSV file.")
    
    # Use a callback function to write results incrementally
    def callback(result):
        write_result_to_csv(result, output_csv_path)
    
    # Process audio files in parallel
    Parallel(n_jobs=n_jobs)(
        delayed(lambda row: callback(process_audio(row)))(row)
        for row in tqdm(audio_files.itertuples(index=False), desc="Processing Duet Files")
    )

# this does take a while - only cpu with parallel
if __name__ == "__main__":
    cleaned_metadata = pd.read_csv('cleaned_metadata.csv')
    all_duets = cleaned_metadata[cleaned_metadata['ensemble'] == 'DUET'][['vocal1_path', 'vocal2_path']]
    
    voice_quality_estimation(all_duets, 'DNSMOS_results_DUET.csv', n_jobs=-1)