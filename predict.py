import matplotlib.pyplot as plt
import os, sys
import torch
import torchaudio
import pytorch_lightning as pl
import numpy as np
from solver import LightningSolver
from argparse import ArgumentParser
import pandas as pd
import yaml
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pprint
import warnings
from einops import rearrange
warnings.filterwarnings('ignore')

def predict(config:dict, data_type="eval", sample_rate=16000):

    lite = LightningSolver.load_from_checkpoint(config['checkpoint_path'], strict=False, config=config).cuda()
    lite.eval()

    speaker2idx={}
    with open(config['speakers']['save_path'], 'rb') as f:
        speaker2idx = pickle.load(f)
    idx2speaker = {v: k for k, v in speaker2idx.items()}

    transform = torchaudio.transforms.MelSpectrogram(sample_rate)
    predicts, targets = [], []
    with torch.no_grad():
        df = pd.read_csv(config['csv'])
        for idx, row in df.query('data_type==@data_type').iterrows():
            wave, sr = torchaudio.load(row['path'])
            spec = transform(wave)
            spec = rearrange(spec, '(b c) f t -> b c f t', b=1)
            logits = lite.forward(spec.cuda())
            predicts.append(torch.argmax(logits, axis=-1).item())
            targets.append(speaker2idx[row['speaker']])

    # 全体の正解率など
    df = pd.DataFrame(classification_report(targets, predicts, target_names = speaker2idx.keys(), output_dict=True))
    print(df)
    df.to_csv(config['report']['path'])
    
    # 混同行列
    target_labels = [ idx2speaker[id] for id in targets ]
    predict_labels = [ idx2speaker[id] for id in predicts ]
    cm = confusion_matrix(target_labels, predict_labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(config['report']['confusion_matrix'])
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
        
    predict(config) 
