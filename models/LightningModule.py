from typing import Any
from torch.optim import Adam
from torchmetrics import RelativeSquaredError, MeanSquaredError
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts




class LitEstimator(L.LightningModule):
    def __init__(self, learning_rate, model):
        super().__init__()
        self.lr = learning_rate
        self.model = model
        self.metric = RelativeSquaredError()
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log('loss', loss, logger=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True, logger=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        # nans = torch.isnan(x)
        # bnans = torch.all(nans, dim=0)
        # import pdb
        # pdb.set_trace()
        # x= x[~nans]
        # y= y[~nans[:,0,:,:]]
        y_hat = self.forward(x)
        # import pdb
        # pdb.set_trace()
        y_flat = y.reshape(-1)
        # nans = torch.isnan(y_flat)
        y_hat_flat = y_hat.view(-1)
        # y_flat = y_flat[nans]
        # nans = torch.isnan(y_hat).view(-1)
        loss = self.metric(y_hat_flat, y_flat)
        
        return loss, y_hat, y

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1,)

        return {'optimizer':optimizer, 'lr_scheduler':scheduler}
