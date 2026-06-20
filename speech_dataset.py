import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from typing import Tuple, List
from torch import Tensor
import torchaudio
import scipy.signal as signal
from einops import rearrange
import pickle
from tqdm import tqdm

# カーネルが生きている限り保持されるグローバルキャッシュ
_GLOBAL_DATA_CACHE = {}


def speaker_to_student_label(speaker_name: str) -> str:
    speaker_name = str(speaker_name)
    base_name = os.path.basename(os.path.normpath(speaker_name))
    match = re.search(r'(\d{8,})', base_name)
    if match:
        return match.group(1)
    return base_name


def build_speaker2idx(df: pd.DataFrame) -> dict:
    speakers = sorted(
        df['speaker'].dropna().unique(),
        key=lambda spk: (speaker_to_student_label(spk), str(spk)),
    )
    return {spk: idx for idx, spk in enumerate(speakers)}


def save_speaker2idx(speaker2idx: dict, save_path: str) -> None:
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(speaker2idx, f)

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str, sample_rate=16000, speaker2idx=None, save_path=None, valid=False) -> None:
        super().__init__()

        data_type = 'valid' if valid is True else 'train'
        self.df = (
            pd.read_csv(csv_path)
            .query('data_type==@data_type')
            .assign(_student_label=lambda df: df['speaker'].map(speaker_to_student_label))
            .sort_values(['_student_label', 'speaker', 'path'])
            .drop(columns=['_student_label'])
            .reset_index(drop=True)
        )
        self.sample_rate = sample_rate

        if speaker2idx is not None:
            unknown_speakers = sorted(set(self.df['speaker']) - set(speaker2idx))
            if unknown_speakers:
                print(
                    f"Skip {len(unknown_speakers)} unknown speakers in {data_type}: "
                    + ', '.join(unknown_speakers)
                )
                self.df = (
                    self.df.query('speaker in @speaker2idx')
                    .reset_index(drop=True)
                )
            self.speaker2idx = speaker2idx
        else:
            self.speaker2idx = build_speaker2idx(self.df)
        self.idx2speaker = {v: k for k, v in self.speaker2idx.items()}

        if speaker2idx is None and save_path is not None:
            if os.path.isfile(save_path):
                with open(save_path, 'rb') as f:
                    saved_speaker2idx = pickle.load(f)
                if saved_speaker2idx != self.speaker2idx:
                    print(f"speaker2idx changed. Overwrite {save_path}.")
            save_speaker2idx(self.speaker2idx, save_path)

        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=80)

        # キャッシュキーの作成（話者ID対応が変わった場合は別データとして扱う）
        speaker_key = tuple(sorted(self.speaker2idx.items()))
        cache_key = (csv_path, data_type, speaker_key)
        
        global _GLOBAL_DATA_CACHE
        if cache_key in _GLOBAL_DATA_CACHE:
            print(f"Using cached {data_type} dataset ({len(self.df)} files). No Drive access needed.")
            self.cached_data = _GLOBAL_DATA_CACHE[cache_key]
        else:
            self.cached_data = []
            print(f"Loading {data_type} dataset ({len(self.df)} files) into memory for the first time...")
            for idx in tqdm(range(len(self.df)), desc=f"Loading {data_type}"):
                spec, speaker = self._load_and_process(idx)
                self.cached_data.append((spec, speaker))
            # キャッシュに保存
            _GLOBAL_DATA_CACHE[cache_key] = self.cached_data

    def _load_and_process(self, idx: int) -> Tuple[Tensor, int]:
        row = self.df.iloc[idx]
        path = row['path']

        try:
            wave, sr = torchaudio.load(path)
            wave = wave[0, :].unsqueeze(dim=0)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate, dtype=wave.dtype)
                wave = resampler(wave)
            
            std, mean = torch.std_mean(wave, dim=-1)
            wave = (wave - mean) / (std + 1e-9)
            
            spec = torch.log(self.transform(wave) + 1.e-9)
            speaker = self.speaker2idx[row['speaker']]
            return spec, speaker
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise RuntimeError('file error', path)

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, idx:int) -> Tuple[Tensor, int]:
        return self.cached_data[idx]
    
    def _speaker2idx(self):
        return self.speaker2idx

'''
    バッチデータの作成
'''
def data_processing(data:Tuple[Tensor,int]) -> Tuple[Tensor, Tensor]:
    specs = []
    speakers = []

    for spec, speaker in data:
        c, _, _ = spec.shape
        spec = rearrange(spec, 'c f t -> t (c f)')
        specs.append(spec)
        speakers.append(speaker)

    specs = nn.utils.rnn.pad_sequence(specs, batch_first=True)
    specs = rearrange(specs, 'b t (c f) -> b c f t', c=c)
    speakers = torch.from_numpy(np.array(speakers)).clone()

    return specs, speakers
