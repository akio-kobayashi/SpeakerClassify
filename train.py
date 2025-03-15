import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from solver import LightningSolver
from speech_dataset import SpeechDataset
from speech_dataset import data_processing
from argparse import ArgumentParser
import yaml
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)  # Python の乱数を固定
    np.random.seed(seed)  # NumPy の乱数を固定
    torch.manual_seed(seed)  # PyTorch の CPU 乱数を固定
    torch.cuda.manual_seed(seed)  # CUDA (GPU) の乱数を固定
    torch.cuda.manual_seed_all(seed)  # 複数 GPU の場合にも適用
    torch.backends.cudnn.deterministic = True  # cuDNN を決定的動作に
    torch.backends.cudnn.benchmark = False  # 最適化をオフ（再現性を優先）

def train(config:dict, model):

    set_seed(42)
    lite = LightningSolver(config, model)
       
    train_dataset = SpeechDataset(csv_path=config['csv'], save_path=config['speakers']['save_path'], valid=False)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=config['batch_size'],
                                   num_workers=1,
                                   pin_memory=True,
                                   shuffle=True, 
                                   collate_fn=lambda x: data_processing(x))
    valid_dataset = SpeechDataset(csv_path=config['csv'], speaker2idx=train_dataset._speaker2idx(), valid=True)
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   batch_size=config['batch_size'],
                                   num_workers=1,
                                   pin_memory=True,
                                   shuffle=False, 
                                   collate_fn=lambda x: data_processing(x))
           
    callbacks = [
        pl.callbacks.ModelCheckpoint( **config['checkpoint'])
    ]
    logger = TensorBoardLogger(**config['logger'])

    trainer = pl.Trainer( callbacks=callbacks,
                          logger=logger,
                          **config['trainer'] )
    trainer.fit(model=lite, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args=parser.parse_args()

    #torch.set_default_device("cuda:"+str(args.gpu))
    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    if 'config' in config.keys():
        config = config['config']
    config['trainer'].pop('profiler')
    train(config, args.checkpoint)
