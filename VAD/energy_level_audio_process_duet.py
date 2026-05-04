import librosa as lb
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os

def load_audio(file_path, sr=16000):
    signal, sampling_rate = lb.load(file_path, sr=sr)
    return signal, sampling_rate


def calculate_energy(signal, sr, frame_duration=0.025, hop_duration=0.010):
    frame_length = int(frame_duration * sr)  # Convert ms to samples
    hop_length = int(hop_duration * sr)
    energy = np.array([np.sum(np.abs(signal[i:i+frame_length]**2)) 
                       for i in range(0, len(signal), hop_length)])
    energy /= np.max(energy)  # Normalize energy
    return energy, hop_length


def detect_audio_segments(energy, hop_length, sr, threshold=0.001, max_gap=0.05):
    voice_frames = energy > threshold
    frame_duration = hop_length / sr
    speech_segments = []
    starting_segment = -1

    for i, is_voice in enumerate(voice_frames):
        time_stamp = i * frame_duration
        if is_voice:
            if starting_segment < 0:  # Track the beginning of speech
                starting_segment = time_stamp
        elif starting_segment >= 0 and (time_stamp - starting_segment) <= max_gap:
            continue
        elif starting_segment >= 0:
            speech_segments.append((starting_segment, time_stamp))
            starting_segment = -1

    return speech_segments


def merge_segments(speech_segments, merge_threshold=0.15):
    merged_segments = []
    if speech_segments:
        current_segment = speech_segments[0]
        for next_segment in speech_segments[1:]:
            if next_segment[0] - current_segment[1] <= merge_threshold:
                current_segment = (current_segment[0], next_segment[1])
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment
        merged_segments.append(current_segment)
    return merged_segments


def filter_final_segments(merged_segments, min_speech_duration=3.0):
    final_segments = [(start, end) for start, end in merged_segments 
                      if (end - start) >= min_speech_duration]
    return final_segments

def process_single_audio_file(vocal1_path, vocal2_path, sr, threshold, max_gap, min_speech_duration, merge_threshold):
    # Process vocal1 to calculate segments
    signal1, sr1 = load_audio(f'/mnt/data/damp-vsep/{vocal1_path}', sr)
    energy1, hop_length1 = calculate_energy(signal1, sr1)
    speech_segments1 = detect_audio_segments(energy1, hop_length1, sr1, threshold, max_gap)
    merged_segments1 = merge_segments(speech_segments1, merge_threshold)
    final_segments1 = filter_final_segments(merged_segments1, min_speech_duration)

    # Return both vocal1_path and vocal2_path along with the segments
    return vocal1_path, final_segments1, vocal2_path

def write_result_to_csv(result, csv_file):
    """
    Writes a single result to the CSV file.
    """
    vocal1_path, final_segments1, vocal2_path = result
    if final_segments1:
        temp_df = pd.DataFrame({
            'vocal1_path': [vocal1_path],
            'vocal1_segments': [final_segments1],
            'vocal2_path': [vocal2_path]
        })
        temp_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

def process_audio_files(file_paths, sr=16000, threshold=0.001, max_gap=0.05, min_speech_duration=3.0, merge_threshold=0.15, csv_file='/home/yuex7/research/audio_segments.csv', n_jobs=-1):
    output_dir = os.path.dirname(csv_file)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.isfile(csv_file):
        header_df = pd.DataFrame(columns=['vocal1_path', 'vocal1_segments', 'vocal2_path'])
        header_df.to_csv(csv_file, index=False)
        print("Header written to CSV file.")
    
    def callback(result):
        write_result_to_csv(result, csv_file)
    
    # Process vocal1_path to calculate segments, but save both vocal1_path and vocal2_path
    Parallel(n_jobs=n_jobs)(
        delayed(lambda vocal1_path, vocal2_path: callback(process_single_audio_file(vocal1_path, vocal2_path, sr, threshold, max_gap, min_speech_duration, merge_threshold)))(
            row['vocal1_path'], row['vocal2_path']
        )
        for _, row in tqdm(file_paths.iterrows(), desc="Processing Audio Files")
    )
    
    
if __name__ == "__main__":
    DNSMOS_duet = pd.read_csv('DNSMOS_results_DUET.csv')
    DNSMOS_duet_clean = DNSMOS_duet[(DNSMOS_duet['vocal1_bak_mos'] > 3) & (DNSMOS_duet['vocal2_bak_mos'] > 3)]
    
    process_audio_files(DNSMOS_duet_clean)