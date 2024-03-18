from typing import Any
from matplotlib import colors, pyplot as plt
import matplotlib
from scipy.stats import binned_statistic_2d
import numpy as np
from torch.optim import Adam
from torchmetrics import Metric, RelativeSquaredError, MeanSquaredError, R2Score, MeanAbsoluteError, MetricCollection, MeanMetric
from torchmetrics.aggregation import MeanMetric
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.plotting import latlon_error_plot, plot_patches, prediction_error_plot
import wandb
from utils.plotting import SZA_error_plot, dayofyear_error_plot
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class LitEstimator(L.LightningModule):
    def __init__(self, learning_rate, model, dm):
        super().__init__()
        self.lr = learning_rate
        self.model = model
        self.metric = RelativeSquaredError()
        self.save_hyperparameters()
    
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
        plt.close()

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
        self.save_hyperparameters(ignore=["y","y_hat", "x_attr", "model"])
        self.lr = learning_rate
        self.model = model
        self.transform = config.transform
        self.y_vars = config.y_vars
        self.x_features = config.x_features
        self.metric = MeanSquaredError()
        self.other_metrics = MetricCollection([RelativeSquaredError(), MeanAbsoluteError(), R2Score()])
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
        y = torch.cat(self.y)
        y_hat = torch.cat(self.y_hat)
        x_attr = torch.cat(self.x_attr)


        if self.transform:
            y_hat = self.transform.inverse(y_hat.cpu(), self.y_vars)
            y = self.transform.inverse(y.cpu(), self.y_vars)
            x_attr = self.transform.inverse(x_attr.cpu(), self.x_features)
        

        error = y_hat - y
        if len(self.y_vars)>1:
            SIS_error = error[:, self.y_vars.index('SIS')]
            SIS_idx = self.y_vars.index('SIS')
            losses = self.other_metrics(y_hat[:, SIS_idx].flatten().cuda(), y[:, SIS_idx].flatten().cuda())
        else:
            SIS_error = error
            losses = self.other_metrics(y_hat.flatten().cuda(), y.flatten().cuda())

        self.log_dict(losses, logger=True, sync_dist=True)

        if 'SZA' in self.x_features and 'dayofyear' in self.x_features:
            SZA = x_attr[:, self.x_features.index('SZA')]
            dayofyear = x_attr[:, self.x_features.index('dayofyear')]

            SZA_fig, _ = SZA_error_plot(SZA, SIS_error)
            dayofyear_fig, _ = dayofyear_error_plot(dayofyear, SIS_error)

            self.logger.log_image(key="y_hat - y, distribution SZA and dayofyear groundstations", images=[SZA_fig, dayofyear_fig])
       
        figs = []
        if len(self.y_vars) > 1:
            for i in range(y.shape[1]):
                fig = prediction_error_plot(y[:, i].cpu(), y_hat[:, i].cpu(), self.y_vars[i])
                figs.append(fig)
        else:
            fig = prediction_error_plot(y.cpu(), y_hat.cpu(), self.y_vars[0])
            figs.append(fig)
        self.logger.log_image(key="Prediction error groundstations", images=figs)
        plt.close()
    
    def predict_step(self, batch, batch_idx):
        X, x, y = batch
        y_hat = self.forward(X, x)
        return y_hat, y, x

    def on_validation_epoch_end(self):
        y = torch.cat(self.y)
        y_hat = torch.cat(self.y_hat)
        x_attr = torch.cat(self.x_attr)

        if self.transform:
            y_hat = self.transform.inverse(y_hat.cpu(), self.y_vars)
            y = self.transform.inverse(y.cpu(), self.y_vars)
            x_attr = self.transform.inverse(x_attr.cpu(), self.x_features)

        error = y_hat - y
        if len(self.y_vars)>1:
            SIS_error = error[:, self.y_vars.index('SIS')]
            SIS_idx = self.y_vars.index('SIS')
            losses = self.other_metrics(y_hat[:, SIS_idx].double().cuda(), y[:, SIS_idx].double().cuda())
        else:
            SIS_error = error
            losses = self.other_metrics(y_hat.flatten().double().cuda(), y.flatten().double().cuda())

        self.log_dict(losses, logger=True, sync_dist=True)
        
        if 'SZA' in self.x_features and 'dayofyear' in self.x_features:
            SZA = x_attr[:, self.x_features.index('SZA')]
            dayofyear = x_attr[:, self.x_features.index('dayofyear')]

            SZA_fig, _ = SZA_error_plot(SZA, SIS_error)
            dayofyear_fig, _ = dayofyear_error_plot(dayofyear, SIS_error)
            self.logger.log_image(key="y_hat - y, distribution SZA and dayofyear", images=[SZA_fig, dayofyear_fig])

        
        if 'lat' in self.x_features and 'lon' in self.x_features:
            lat = x_attr[:, self.x_features.index('lat')]
            lon = x_attr[:, self.x_features.index('lon')]

            fig, _ = latlon_error_plot(lat, lon, SIS_error)
            self.logger.log_image(key="Error Location", images=[fig])
            plt.close()

        figs = []
        if len(self.y_vars) > 1:
            for i in range(y.shape[1]):
                fig = prediction_error_plot(y[:, i].cpu(), y_hat[:, i].cpu(), self.y_vars[i])
                figs.append(fig)
        else:
            fig = prediction_error_plot(y.cpu(), y_hat.cpu(), self.y_vars[0])
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
        y_hat = self.forward(X, x)
        loss = self.metric(y_hat, y)
        return loss, y_hat.squeeze(), y.squeeze()

    # def configure_optimizers(self):
    #     optimizer = Adam(self.parameters(), lr=self.lr)
    #     scheduler = CosineAnnealingLR(optimizer, T_max=1)
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.lr,
                                    #   betas=(0.5, 0.9), 
                                      weight_decay=1e-2)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }