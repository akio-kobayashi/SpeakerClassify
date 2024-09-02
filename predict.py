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
warnings.filterwarnings('ignore')

def predict(config:dict):

    lite = LightningSolver.load_from_checkpoint(config['checkpoint_path'], strict=False, config=config).cuda()
    lite.eval()

    speaker2idx={}
    with open(config['speakers']['save_path'], 'rb') as f:
        speaker2idx = pickle.load(f)
    idx2speaker = {v: k for k, v in speaker2idx.items()}

    predicts, targets = [], []
    with torch.no_grad():
        df = pd.read_csv(config['csv'])
        for idx, row in df.query('data_type=="eval"').iterrows():
            wave, sr = torchaudio.load(row['path'])
            wave = wave.unsqueeze(0)
            logits = lite.forward(wave.cuda())
            predicts.append(torch.argmax(logits, axis=-1).item())
            targets.append(speaker2idx[row['speaker']])

    # 全体の正解率など
    df = pd.DataFrame(classification_report(targets, predicts, target_names = speaker2idx.keys(), output_dict=True))
    print(df)
    #df.to_csv(os.path.join(config['logger']['save_dir'], config['report']['path']))
    df.to_csv(config['report']['path'])
    
    # 混同行列
    cm = confusion_matrix(targets, predicts, normalize='true', labels = sorted(speaker2idx.items(), key=lambda x:x[1]))
    disp = ConfusionMatrixDisplay(consusion_matrix=cm, display_labels=sorted(speaker2idx.items(), key=lambda x:x[1]))
    disp.plot()
    #plt.save_fig(os.path.join(config['logger']['save_dir'], config['report']['confusion_matrix']))
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
