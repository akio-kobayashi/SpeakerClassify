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

        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate)

    def __len__(self) -> int:
        return len(self.df)

    '''
        データフレームからidx番目のサンプルを抽出する
    '''
    def __getitem__(self, idx:int) -> Tuple[Tensor, int]:
        row = self.df.iloc[idx]
        # audio path
        path = row['path']


        try:
            # torchaudioで読み込んだ場合，音声データはFloatTensorで（チャンネル，サンプル数）
            wave, sr = torchaudio.load(path)
            # 平均をゼロ，分散を1に正規化
            std, mean = torch.std_mean(wave, dim=-1)
            wave = (wave - mean)/std
            spec = self.transform(wave)
        except:
            raise RuntimeError('file open error')
        
        speaker = self.speaker2idx[row['speaker']]
        
        return spec, speaker
    
    def _speaker2idx(self):
        return self.speaker2idx

'''
    バッチデータの作成
'''
def data_processing(data:Tuple[Tensor,int]) -> Tuple[Tensor, Tensor]:
    specs = []
    speakers = []

    for spec, speaker in data:
        # w/ channel
        c, _, _ = spec.shape
        spec = rearrange(spec, 'c f t -> t (c f)')
        specs.append(spec)
        speakers.append(speaker)

    # データはサンプル数（長さ）が異なるので，長さを揃える
    # 一番長いサンプルよりも短いサンプルに対してゼロ詰めで長さをあわせる
    # バッチはFloatTensorで（バッチサイズ，チャンネル，サンプル数）
    spees = nn.utils.rnn.pad_sequence(specs, batch_first=True)
    specs = rearrange(specs, 'b t (c f) -> b c f t', c=c)
    # 話者のインデックスを配列（Tensor）に変換
    speakers = torch.from_numpy(np.array(speakers)).clone()

    return specs, speakers
