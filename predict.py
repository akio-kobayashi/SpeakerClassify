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
    transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=80)
    predicts, targets = [], []
    files, spk_targets, spk_predicts = [], [], [] 
    corrects=samples=0
    with torch.no_grad():
        df = pd.read_csv(config['csv'])
        for idx, row in df.query('data_type==@data_type').iterrows():
            wave, sr = torchaudio.load(row['path'])
            std, mean = torch.std_mean(wave, dim=-1)
            wave = (wave - mean)/std
            spec = torch.log10(transform(wave) + 1.e-9)
            #std, mean = torch.std_mean(spec)
            #spec = (spec - mean)/std
            spec = rearrange(spec, '(b c) f t -> b c f t', b=1)
            logits = lite.forward(spec.cuda())
            predicts.append(torch.argmax(logits, axis=-1).item())
            targets.append(speaker2idx[row['speaker']])
            tgt = speaker2idx[row['speaker']]
            prd = torch.argmax(logits, axis=-1).item()
            if tgt==prd:
                corrects += 1
            samples += 1

            files.append(row['path'])
            spk_targets.append(row['speaker'])
            spk_predicts.append(idx2speaker[prd])
            
    #print(f'{corrects} {samples}')
    # 全体の正解率など
    df = pd.DataFrame(classification_report(targets, predicts, target_names = speaker2idx.keys(), output_dict=True))
    print(df)
    df.to_csv(config['report']['path'])

    detail = pd.DataFrame.from_dict({'file': files, 'correct': spk_targets, 'predict': spk_predicts })
    detail.to_csv(config['report']['detail'])
    
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
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--gpu',  type=int, default=0)
    args=parser.parse_args()

    #torch.set_default_device("cuda:"+str(args.gpu))
    
    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    config['checkpoint_path'] = args.ckpt
    predict(config) 
