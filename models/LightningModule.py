from typing import Any
from torch.optim import Adam
from torchmetrics import Metric, RelativeSquaredError, MeanSquaredError, R2Score, MeanAbsoluteError, MetricCollection
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.aggregation import MeanMetric
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.plotting import plot_patches, prediction_error_plot
import wandb

class MyRelativeMeanSquaredError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds, target) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        # parse inputs
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        # some intermediate computation...
        
        
        # finalize the computations
        loss = torch.nanmean( (target-preds)**2/target**2)
        return loss

def mse_loss_with_nans(input, target):

    # Missing data are nan's
    mask = torch.isnan(target)

    # Missing data are 0's
    # mask = target == 0

    out = (input[~mask]-target[~mask])**2
    loss = out.mean()

    return loss



class LitEstimator(L.LightningModule):
    def __init__(self, learning_rate, model, dm):
        super().__init__()
        self.lr = learning_rate
        self.model = model
        self.metric = RelativeSquaredError()
        self.dm = dm
        self.save_hyperparameters(ignore=["dm"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log("loss", loss, logger=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, logger=True, prog_bar=True)
        if loss > self.val_loss_worst:
            self.val_batch_worst = batch
            self.val_loss_worst = loss
        if loss < self.val_loss_best:
            self.val_batch_best = batch
            self.val_loss_best = loss
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "val_loss_worst": self.val_loss_worst,
                "val_loss_best": self.val_loss_best,
            }
        )
        _, y_hat, y = self._shared_eval_step(self.val_batch_worst, self.val_idx_worst)
       
        error = torch.mean(torch.sqrt((y_hat - y) ** 2), dim=[1, 2])
        _, ind = torch.topk(error, k=4)
        fig = plot_patches(y[ind].cpu(), y_hat[ind].cpu(), n_patches=4)
        fig.savefig("test_worstpatchers.png")
        self.logger.log_image("Worst Patches", images=[wandb.Image(fig)])

        _, y_hat, y = self._shared_eval_step(self.val_batch_best, self.val_idx_best)
        error = torch.mean(torch.sqrt((y_hat - y) ** 2), dim=[1, 2])
        _, ind = torch.topk(error, k=4)
        fig = plot_patches(y[ind].cpu(), y_hat[ind].cpu(), n_patches=4)
        fig.savefig("test_bestpatches.png")
        self.logger.log_image("Best Patches", images=[wandb.Image(fig)])

    def on_validation_epoch_start(self):
        self.val_loss_worst = 0
        self.val_loss_best = 1000
        self.val_idx_best = None
        self.val_idx_worst = None

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        # import pdb
        # pdb.set_trace()
        y_hat = self.forward(x)
        y_flat = y.reshape(-1)
        y_hat_flat = y_hat.view(-1)
        loss = self.metric(y_hat_flat, y_flat)
        return loss, y_hat.squeeze(), y.squeeze()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class LitEstimatorPoint(L.LightningModule):
    def __init__(self, learning_rate, model, dm):
        super().__init__()
        self.save_hyperparameters(ignore=["dm","y","y_hat"])
        self.lr = learning_rate
        self.model = model
        self.metric = MeanSquaredError()
        self.other_metrics = MetricCollection([RelativeSquaredError(), MeanAbsoluteError(), R2Score()])
        self.dm = dm
        self.y = []
        self.y_hat =[]

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log("loss", loss, logger=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, logger=True, prog_bar=True)
        self.y.append(y)
        self.y_hat.append(y_hat)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx, ):
        X, x, y = batch
        y_hat = self.forward(X.float(), x.float())
        return y_hat, x

    def on_validation_epoch_end(self):

        y = torch.vstack(self.y)
        y_hat = torch.vstack(self.y_hat)

        if self.dm.target_transform:
            y_hat = self.dm.target_transform.inverse(y_hat, self.dm.y_vars)
            y = self.dm.target_transform.inverse(y, self.dm.y_vars)

        losses = self.other_metrics(y_hat.reshape(-1), y.reshape(-1))
        self.log_dict(losses, logger=True, sync_dist=True)


        fig = prediction_error_plot(y.cpu(), y_hat.cpu())
        self.logger.log_image(key="Prediction error", images=[fig])
    
    def on_validation_epoch_start(self):
        self.y = []
        self.y_hat = []

    def forward(self, X, x_attrs):
        return self.model(X,x_attrs)
    
    def _shared_eval_step(self, batch, batch_idx):
        X, x, y = batch
        y_hat = self.forward(X.float(), x.float())
        y_flat = y.reshape(-1)
        y_hat_flat = y_hat.view(-1)
        loss = self.metric(y_hat_flat, y_flat)
        return loss, y_hat.squeeze(), y.squeeze()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
