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

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str, sample_rate=16000, speaker2idx=None, save_path=None, valid=False) -> None:
        super().__init__()

        data_type = 'valid' if valid is True else 'train'
        self.df = pd.read_csv(csv_path).query('data_type==@data_type')
        self.sample_rate = sample_rate

        if speaker2idx is not None:
            self.speaker2idx = speaker2idx
        else:
            self.speaker2idx = {}
            for idx, spk in enumerate(self.df['speaker'].unique()):
                self.speaker2idx[spk] = idx
        self.idx2speaker = {v: k for k, v in self.speaker2idx.items()}

        if speaker2idx is None and save_path is not None:
            if os.path.isfile(save_path) == False:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.speaker2idx, f)

        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=80)

        # キャッシュキーの作成（CSVパスとデータ種別で一意に特定）
        cache_key = (csv_path, data_type)
        
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
