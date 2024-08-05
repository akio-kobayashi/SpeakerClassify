import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple
from einops import rearrange
from model import Baseline

class LightningSolver(pl.LightningModule):
    def __init__(self, config:dict, model=None) -> None:
        super().__init__()

        self.optim_config=config['optimizer']

        self.model = model
        if model is None:
            self.model = Baseline(config['num_speakers'])

        self.loss = nn.CrossEntropyLoss(reduction='sum')

        self.num_corrects = self.num_samples = 0
        self.save_hyperparameters()

    def forward(self, wave:Tensor) -> Tensor:
        return self.model(wave)

    def compute_loss(self, estimates, targets, valid=False):
        d={}
        _loss = self.loss(estimates, targets)
        if valid:
            d['valid_loss'] = _loss
        else:
            d['train_loss'] = _loss
       
        self.log_dict(d)

        return _loss

    def compute_correct(self, estimates, targets, valid=False):
        prediction = torch.argmax(estimates, dim=-1)
        self.num_corrects = torch.sum((prediction == targets).long())
        self.num_samples += targets[0]

    def training_step(self, batch, batch_idx:int) -> Tensor:
        waves, targets = batch
        estimates = self.forward(waves)
        _loss = self.compute_loss(estimates, targets, valid=False)

        return _loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        waves, targets = batch
        estimates = self.forward(waves)
        _loss = self.compute_loss(estimates, targets, valid=True)
        self.compute_correct(estimates, targets)

        return _loss

    def on_validation_epoch_end(self):
        corr = self.num_corrects.item() / self.num_samples
        self.log_dict({'valid_corr': corr})
        self.num_corrects = self.num_samples = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.optim_config)
        return optimizer
