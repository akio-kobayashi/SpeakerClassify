import matplotlib.pyplot as plt
import os, sys
import re
import torch
import torchaudio
import lightning.pytorch as pl
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


def speaker_to_student_label(speaker_name: str) -> str:
    speaker_name = str(speaker_name)
    base_name = os.path.basename(os.path.normpath(speaker_name))
    match = re.search(r'(\d{8,})', base_name)
    if match:
        return match.group(1)
    return base_name


def get_label_map_path(config: dict) -> str:
    label_map_path = config['report'].get('label_map')
    if label_map_path:
        return label_map_path
    base, _ = os.path.splitext(config['report']['confusion_matrix'])
    return base + '_label_map.csv'

def predict(config:dict, model, data_type="eval", sample_rate=16000):

    ckpt = torch.load(config['checkpoint_path'], weights_only=False)
    lite = LightningSolver(config=config, model=model)
    lite.load_state_dict(ckpt['state_dict'], strict=False)
    lite = lite.cuda()
    #lite = LightningSolver.load_from_checkpoint(config['checkpoint_path'], strict=False, config=config).cuda()
    lite.eval()

    speaker2idx={}
    with open(config['speakers']['save_path'], 'rb') as f:
        speaker2idx = pickle.load(f)
    idx2speaker = {v: k for k, v in speaker2idx.items()}
    idx2student = {idx: speaker_to_student_label(spk) for idx, spk in idx2speaker.items()}
    transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=80)
    predicts, targets = [], []
    files, spk_targets, spk_predicts = [], [], [] 
    corrects=samples=0
    skipped_unknown = []
    with torch.no_grad():
        df = pd.read_csv(config['csv'])
        unknown_speakers = sorted(set(df.query('data_type==@data_type')['speaker']) - set(speaker2idx))
        if unknown_speakers:
            print(
                f"Skip {len(unknown_speakers)} unknown speakers in {data_type}: "
                + ', '.join(unknown_speakers)
            )
        for idx, row in df.query('data_type==@data_type').iterrows():
            if row['speaker'] not in speaker2idx:
                skipped_unknown.append(row['path'])
                continue
            wave, sr = torchaudio.load(row['path'])
            wave = wave[0, :].unsqueeze(dim=0)
            if sr != 16_000:
                resampler = torchaudio.transforms.Resample(sr, 16_000, dtype=wave.dtype)
                wave = resampler(wave)
            std, mean = torch.std_mean(wave, dim=-1)
            wave = (wave - mean)/std
            spec = torch.log(transform(wave) + 1.e-9)
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
            spk_targets.append(speaker_to_student_label(row['speaker']))
            spk_predicts.append(idx2student.get(prd, f'<unknown:{prd}>'))

    if not samples:
        raise RuntimeError(
            'No evaluable samples remain after speaker filtering. '
            'Check speakers.pkl and the CSV speaker labels.'
        )
            
    # speaker2idx の key（話者名）を value順（インデックス順）にソートして取得
    sorted_speakers = sorted(speaker2idx.items(), key=lambda x: x[1])
    id_to_student_id = {idx: speaker_to_student_label(name) for name, idx in sorted_speakers}
    label_map = pd.DataFrame(
        {
            'label': [idx for _, idx in sorted_speakers],
            'student_id': [speaker_to_student_label(name) for name, _ in sorted_speakers],
            'speaker': [name for name, _ in sorted_speakers],
        }
    )
    label_map_path = get_label_map_path(config)
    label_map.to_csv(label_map_path, index=False)

    # 出現するクラスインデックス（int型）を取得
    unique_labels = sorted(set(targets) | set(predicts))

    # 出現クラスに対応する名前だけ抽出
    target_names_used = [id_to_student_id.get(id, f'<unknown:{id}>') for id in unique_labels]
    print(f'{corrects} {samples} {corrects/samples}')

    # 全体の正解率など
    #df = pd.DataFrame(classification_report(targets, predicts, target_names = speaker2idx.keys(), output_dict=True))
    df = pd.DataFrame(classification_report(targets, predicts, labels=unique_labels, target_names = target_names_used, output_dict=True))
    
    print(df)
    df.to_csv(config['report']['path'])
    print(label_map)
    print(f'label map: {label_map_path}')

    detail = pd.DataFrame.from_dict({'file': files, 'correct': spk_targets, 'predict': spk_predicts })
    detail.to_csv(config['report']['detail'])
    
    # 混同行列の計算は内部IDで統一し、表示は縦軸を学生ID、横軸を内部IDにする。
    target_labels = targets
    predict_labels = predicts
    cm_labels = [idx for _, idx in sorted_speakers]
    cm_true_display_labels = [id_to_student_id[idx] for idx in cm_labels]
    cm_pred_display_labels = cm_labels
    cm = confusion_matrix(target_labels, predict_labels, labels=cm_labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_pred_display_labels)
    disp.plot(xticks_rotation=0)
    disp.ax_.set_yticks(range(len(cm_true_display_labels)))
    disp.ax_.set_yticklabels(cm_true_display_labels)
    disp.ax_.set_xlabel('Predicted label (internal ID)')
    disp.ax_.set_ylabel('True label (student ID)')
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
