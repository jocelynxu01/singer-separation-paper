import os
import subprocess
import pandas as pd
from tqdm import tqdm

def convert_ogg_to_pcm(input_path, output_path):
    """
    Convert an OGG file to PCM encoding, convert mono to stereo (double channels),
    and change the sample rate to 44100 Hz.
    
    Args:
    - input_path (str): Path to the input OGG file.
    - output_path (str): Path to save the output file in PCM encoding.
    """
    
    if os.path.exists(output_path):
        print(f"Skipping {input_path}, already converted.")
        return
    
    command = [
        "ffmpeg", "-i", input_path,  # Input file
        "-ac", "1",                  # Convert to stereo (2 channels)
        "-ar", "44100",              # Set sample rate to 44100 Hz
        "-acodec", "pcm_s16le",      # Set codec to PCM signed 16-bit little-endian
        output_path                  # Output file path
    ]
    
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_audio_files(csv_file):
    """
    Process all audio files listed in a CSV file:
    - Convert each OGG file to PCM encoding with stereo channels and 44100 Hz sample rate.
    
    Args:
    - csv_file (str): Path to the CSV file containing audio file paths.
    """
    df = pd.read_csv(csv_file)  # Load CSV file
    audio_files = df['file_path'].tolist()  # Extract file paths

    for file_path in tqdm(audio_files, desc="Processing Audio Files"):
        file_path = f"/mnt/data/damp-vsep/{file_path[10:]}"
        
        if file_path.endswith(".ogg"):
            output_path = file_path.replace(".ogg", "_pcm.wav")  # Save as WAV with '_pcm' suffix
            convert_ogg_to_pcm(file_path, output_path)

# Example usage

if __name__ == "__main__":
    csv_file = "/home/yuex7/research/audio_segments.csv"  # Path to the CSV file
    process_audio_files(csv_file)