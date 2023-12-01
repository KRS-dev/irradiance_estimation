from typing import Any
from torch.optim import Adam
from torchmetrics import RelativeSquaredError, MeanSquaredError
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L



class LitEstimator(L.LightningModule):
    def __init__(self, learning_rate, model):
        super().__init__()
        self.lr = learning_rate
        
        self.model = model
        self.metric = RelativeSquaredError
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        print('forward')
        return self.model(x)

    def training_step(self, batch, batch_idx):
        print('training_step')
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log('loss', loss, logger=True, on_step=True, prog_bar=True)
        print(batch_idx, loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        print('validation')
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log('val_loss', on_epoch=True, logger=True)
        print(loss, batch_idx)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def _shared_eval_step(self, batch, batch_idx):
        print('shared eval')
        x, y = batch
        y_hat = self.forward(x)
        loss = self.metric(y_hat, y)
        return loss, y_hat, y

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
