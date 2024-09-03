import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()

        # 入力特徴量は（バッチサイズ，チャンネル，サンプル数）
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,
                      kernel_size, stride, padding,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.block(x)
        return y

'''
    話者識別のベースライン（基本）モデル
    話者の識別が可能だが，性能が高いわけではない
'''    
class Baseline(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()

        # 畳み込みニューラルネットワークは講義で学習すること
        # 入力特徴量は（バッチサイズ，チャンネル，特徴量，サンプル数）
        # このうちチャンネルは，まず1チャンネル（モノラル）から始まり，
        # self.block出力で512チャンネルになる
        self.block=nn.Sequential(
            ConvBlock(1, 64, kernel_size=16, stride=(2, 2)),
            ConvBlock(64, 512, kernel_size=8, stride=(2, 2)),
            ConvBlock(512, 512, kernel_size=8, stride=(1, 2)),
            ConvBlock(512, 512, kernel_size=4, stride=(1, 1))
        )

        n_mels=80
        # サンプル数の軸にそって平均を計算した後，
        # （バッチサイズ，チャンネル）データに対してアフィン変換を適用する
        self.feedforward = nn.Sequential(
            nn.Linear(512*3, 256),
            nn.ReLU(),
            nn.Linear(256, num_speakers)
        )

    def forward(self, x):
        y = self.block(x)
        y = rearrange(y, 'b c f t -> b (c f) t')
        y = torch.mean(y, axis=-1)
        y = self.feedforward(y)

        return y
    

