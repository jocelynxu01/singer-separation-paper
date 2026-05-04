import os
import pandas as pd
import subprocess
from tqdm import tqdm

def convert_m4a_to_wav(input_path, output_path, sample_rate=44100):
    """Convert background.m4a to WAV with PCM_S16LE encoding."""
    if os.path.exists(output_path):
        print(f'Background WAV already exists: {output_path}. Skipping...')
        return  # Skip if already converted

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        "ffmpeg", "-i", input_path, "-ac", "2", "-ar", str(sample_rate), 
        "-acodec", "pcm_s16le", output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # print(f'Converted {input_path} → {output_path}')
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e}")

def process_audio_files(audio_files):
    """Convert all background.m4a files to background.wav."""
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        base_path = f'/mnt/data/damp-vsep/{audio_file[10:-10]}'
        
        background_m4a = os.path.join(base_path, 'background.m4a')
        background_wav = os.path.join(base_path, 'background.wav')

        # Convert only if the input file exists
        if os.path.exists(background_m4a):
            convert_m4a_to_wav(background_m4a, background_wav)
        else:
            print(f"File not found: {background_m4a}")

if __name__ == "__main__":
    audio_files = pd.read_csv('/home/yuex7/research/audio_segments.csv')['file_path']
    
    # Test with a single file first
    # audio_files = ['RESAMPLED/1/GB/ro/1143747760_2807752881/vocal.ogg']
    
    process_audio_files(audio_files)