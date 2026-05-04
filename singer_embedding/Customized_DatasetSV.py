import torch
import numpy as np
from scipy.io import wavfile
import librosa as lb
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetCustomSV(torch.utils.data.IterableDataset):

    def __init__(
            self,
            audio_files: List[str],
            segments_dict: dict,
            utterance_duration: int = 3.0,
    ):
        super().__init__()
        self.audio_files = audio_files
        self.segments_dict = segments_dict  # Dictionary with file paths and their segments
        self.utterance_duration = utterance_duration #int(utterance_duration * 16000)
        self.rng = np.random.default_rng(0)
        
        self.positive_count = 0
        self.negative_count = 0
        
    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # # Randomly decide whether to create a positive or negative pair
        # is_same_speaker = bool(self.rng.integers(0, 2))
        if self.positive_count < self.negative_count:
            positive_pair = True
        elif self.negative_count < self.positive_count:
            positive_pair = False
        else:
            positive_pair = bool(self.rng.integers(0, 2))

        if positive_pair:
            audio_file = self.rng.choice(self.audio_files)
            segments = self.segments_dict[audio_file]
            
            while len(segments) < 2:
                audio_file = self.rng.choice(self.audio_files)
                segments = self.segments_dict[audio_file]

            # if len(segments) >= 2:
                # Pick two random segments from the same audio file
            segment_indices = self.rng.choice(len(segments), size=2, replace=False)
            seg_1 = segments[segment_indices[0]]
            seg_2 = segments[segment_indices[1]]
            # else:
            #     seg = segments[0]
            #     mid_point = (seg[1] - seg[0]) / 2 + seg[0]  # Calculate midpoint of the segment
            #     seg_1 = (seg[0], mid_point)
            #     seg_2 = (mid_point, seg[1])

            # s_1 = self.load_segment(audio_file, seg_1)
            # s_2 = self.load_segment(audio_file, seg_2)
            # else:
            #     # If only one segment is available, create a negative pair - BAD
            #     seg_1 = segments[0]
            #     # Pick a different file for the second segment
            #     audio_file_2 = self.rng.choice([file for file in self.audio_files if file != audio_file])
            #     seg_2 = self.rng.choice(self.segments_dict[audio_file_2])

            # Generate clean audio segments
            s_1 = self.load_segment(audio_file, seg_1)
            s_2 = self.load_segment(audio_file, seg_2)
            
            self.positive_count += 1
        else:
            audio_file_1, audio_file_2 = self.rng.choice(self.audio_files, size=2, replace=False)
            seg_1 = self.rng.choice(self.segments_dict[audio_file_1])
            seg_2 = self.rng.choice(self.segments_dict[audio_file_2])
            
            s_1 = self.load_segment(audio_file_1, seg_1)
            s_2 = self.load_segment(audio_file_2, seg_2)
            
            self.negative_count += 1

        # Create the output tensor (1 if same speaker, 0 if different)
        label = torch.Tensor([1.0 if positive_pair else 0.0])

        return s_1, s_2, label

    def load_segment(self, file_path: str, segment: Tuple[float, float]) -> torch.Tensor:
        audio_data, sr = lb.load(file_path, sr=16000)
        start_time, end_time = segment
        
        utterance_duration_samples = int(self.utterance_duration * sr)
        
        if end_time > 30:
            end_time = 30
            start_time = max(30 - self.utterance_duration, 0)
            
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        segment_length = end_sample - start_sample
        max_start = segment_length - utterance_duration_samples
        random_start = self.rng.integers(0, max_start + 1)
        
        # print(file_path, ((start_sample + random_start)/16000, (start_sample + random_start + self.utterance_duration)/16000))        
        segment_data = audio_data[start_sample + random_start:start_sample + random_start + utterance_duration_samples]
        
        # if len(segment_data) < utterance_duration_samples:
        #     print(file_path)
        #     print(file_path)
        #     print(file_path)
        #     print(file_path)
        #     print(file_path)
        #     padding = utterance_duration_samples - len(segment_data)
        #     segment_data = np.pad(segment_data, (0, padding), mode='constant')

        return torch.Tensor(segment_data / np.max(np.abs(segment_data))) 
    
    
    
# speaker_ids_tr
# segments_dict_tr
# speaker_ids_vl
# segments_dict_vl
# 80/20

csv_file = 'audio_segments.csv'
df = pd.read_csv(csv_file)

df['segments'] = df['segments'].apply(lambda x: eval(x))
# df['speaker_id'] = df['file_path'].apply(lambda x: f'/mnt/data/damp-vsep/{x}')
# noisy speaker_id
df['speaker_id'] = df['file_path'].apply(lambda x: f'/mnt/data/damp-vsep/{x[10:-10]}/background+vocal_NEW.wav')

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

def create_segments_dict(dataframe):
    segments_dict = {row['speaker_id']: row['segments'] for _, row in dataframe.iterrows()}
    return segments_dict

speaker_ids_tr = train_df['speaker_id'].tolist()
segments_dict_tr = create_segments_dict(train_df)

speaker_ids_vl = val_df['speaker_id'].tolist()
segments_dict_vl = create_segments_dict(val_df)