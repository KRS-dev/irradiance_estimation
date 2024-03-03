from typing import Any
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d
import numpy as np
from torch.optim import Adam
from torchmetrics import Metric, RelativeSquaredError, MeanSquaredError, R2Score, MeanAbsoluteError, MetricCollection, MeanMetric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.aggregation import MeanMetric
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.plotting import plot_patches, prediction_error_plot
import wandb

class LitEstimator(L.LightningModule):
    def __init__(self, learning_rate, model, dm):
        super().__init__()
        self.lr = learning_rate
        self.model = model
        self.metric = RelativeSquaredError()
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
    def __init__(self, learning_rate, model, config):
        super().__init__()
        self.save_hyperparameters(ignore=["y","y_hat"])
        self.lr = learning_rate
        self.model = model
        self.transform = config.transform
        self.y_vars = config.y_vars
        self.x_vars = config.x_vars
        self.metric = MeanSquaredError()
        self.other_metrics = MetricCollection([RelativeSquaredError(), MeanAbsoluteError(), MeanMetric(), R2Score()])
        self.y = []
        self.y_hat =[]
        self.x_attr = []

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log("loss", loss, logger=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, logger=True, prog_bar=True)
        self.y.append(y)
        self.y_hat.append(y_hat)
        self.x_attr.append(batch[1])
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, y_hat, y = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True, logger=True)
        self.y.append(y)
        self.y_hat.append(y_hat)
        self.x_attr.append(batch[1])
        return loss
    
    def on_test_epoch_start(self):
        self.y = []
        self.y_hat = []
        self.x_attr = []

    def on_test_epoch_end(self):
        y = torch.vstack(self.y)
        y_hat = torch.vstack(self.y_hat)
        x_attr = torch.vstack(self.x_attr)

        if self.transform:
            y_hat = self.transform.inverse(y_hat.cpu(), self.y_vars)
            y = self.transform.inverse(y.cpu(), self.y_vars)
            x_attr = self.transform.inverse(x_attr.cpu(), self.x_vars)

        error = y_hat - y
        SIS_error = error[:, 0]
        if 'SZA' in self.x_vars and 'dayofyear' in self.x_vars:
            idx = self.xvars.index('SZA')
            SZA = x_attr[:, idx]
            idx = self.xvars.index('dayofyear')
            dayofyear = x_attr[:, idx]

            bins =  np.arange(0, np.pi, np.pi/8)
            sza_bins_labels = np.rad2deg(bins)
            bin_indices = np.digitize(SZA.cpu(),bins)
            SZAs_errors = [SIS_error[bin_indices == i] for i in range(len(bins))]

            bins = np.arange(0, 365, 7)
            dayofyear_bins_labels = np.arange(0,52)
            bin_indices = np.digitize(dayofyear, bins)
            dayofyears_errors = [SIS_error[bin_indices == i] for i in range(len(bins))]

            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
            dayofyearboxplot = ax1.boxplot(
                dayofyears_errors,
                vert=True,
                sym=None,
                notch=True,
                patch_artist=True,
                labels=dayofyear_bins_labels)
            SZAboxplot = ax2.boxplot(
                SZAs_errors, 
                vert=True,
                sym=None,
                notch=True,
                patch_artist=True, 
                labels=sza_bins_labels)
            
            ax1.set_title('Error distribution within the year')
            ax1.set_ylabel('SIS error [w/m^2]')
            ax1.set_xlabel('Week of the year')
            ax2.set_title('SZA')
            ax2.set_xlabel('Solar Zenith Angle (degrees)')
            self.logger.log_image(key="Error SZA and dayofyear", images=[fig])
        
        if 'lat' in self.x_vars and 'lon' in self.xvars:
            idx = self.xvars.index('lat')
            lat = x_attr[:, idx]
            idx = self.xvars.index('lon')
            lon = x_attr[:, idx]

            lat_bins =  np.arange(np.floor(np.min(lat)), np.max(lat) + 1, 1)
            lon_bins = np.arange(np.floor(np.min(lon)), np.max(lon) + 1, 1)
            mean_error, x_edge, y_edge, _ = binned_statistic_2d(lat, lon, SIS_error, bins = [lat_bins, lon_bins])
            
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            ax.colormesh(C=mean_error, X=x_edge, Y=y_edge)
            ax.set_title('Error distribution Location')
            ax.set_ylabel('Longitude')
            ax.set_xlabel('Latitude')
            self.logger.log_image(key="Error Location", images=[fig])

        
        losses = self.other_metrics(y_hat.reshape(-1), y.reshape(-1))
        self.log_dict(losses, logger=True, sync_dist=True)

        figs = []
        for i in range(y.shape[1]):
            fig = prediction_error_plot(y[:, i].cpu(), y_hat[:, i].cpu(), self.y_vars[i])
            figs.append(fig)
        self.logger.log_image(key="Prediction error groundstations", images=figs)
    
    def predict_step(self, batch, batch_idx, ):
        X, x, y = batch
        y_hat = self.forward(X.float(), x.float())
        return y_hat, x

    def on_validation_epoch_end(self):

        y = torch.vstack(self.y)
        y_hat = torch.vstack(self.y_hat)

        if self.transform:
            y_hat = self.transform.inverse(y_hat.cpu(), self.y_vars)
            y = self.transform.inverse(y.cpu(), self.y_vars)

        losses = self.other_metrics(y_hat.reshape(-1), y.reshape(-1))
        self.log_dict(losses, logger=True, sync_dist=True)

        figs = []
        for i in range(y.shape[1]):
            fig = prediction_error_plot(y[:, i].cpu(), y_hat[:, i].cpu(), self.y_vars[i])
            figs.append(fig)
        self.logger.log_image(key="Prediction error SARAH3", images=figs)
    
    def on_validation_epoch_start(self):
        self.y = []
        self.y_hat = []
        self.x_attr = []    

    def forward(self, X, x_attrs):
        return self.model(X,x_attrs)
    
    def _shared_eval_step(self, batch, batch_idx):
        X, x, y = batch
        y_hat = self.forward(X.float(), x.float())
        loss = self.metric(y_hat, y)
        return loss, y_hat.squeeze(), y.squeeze()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
