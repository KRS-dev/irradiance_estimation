from typing import Any
from matplotlib import colors, pyplot as plt
import matplotlib
from models.ConvResNet_Jiang import ConvResNet_batchnormMLP
from models.ConvResNet_short import ConvResNet_short
from scipy.stats import binned_statistic_2d
import numpy as np
from torch.optim import Adam
from torchmetrics import Metric, RelativeSquaredError, MeanSquaredError, R2Score, MeanAbsoluteError, MetricCollection, MeanMetric
from torchmetrics.aggregation import MeanMetric
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.plotting import latlon_error_plot, plot_patches, prediction_error_plot
import wandb
from utils.plotting import SZA_error_plot, dayofyear_error_plot
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class LitEstimatorPoint(L.LightningModule):
    def __init__(self, learning_rate, config, metric = MeanSquaredError(), 
                 parameter_loss=False, alpha=0.1, zero_loss = None):
        super().__init__()
        self.lr = learning_rate

        self.model = ConvResNet_short(
            num_attr=len(config.x_features),
            input_channels=len(config.x_vars),
            output_channels=len(config.y_vars),
        )

        self.transform = config.transform
        self.y_vars = config.y_vars
        self.x_features = config.x_features
        self.train_metric = MeanSquaredError()
        self.valid_metric = MeanSquaredError()
        self.other_metrics = MetricCollection([MeanMetric(), MeanAbsoluteError(), R2Score()])
        self.y = []
        self.y_hat =[]
        self.x_attr = []

        self.parameter_loss = parameter_loss
        self.parameter_metric = MeanSquaredError().to(self.device)
        self.alpha = alpha

        self.zero_loss = zero_loss
        self.save_hyperparameters(ignore=["y","y_hat", "x_attr"])

    def set_reference_parameters(self, reference_parameters):
        self.reference_parameters = [par.clone().detach().to(self.device) for par in reference_parameters]

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        X, x, y = batch
        y_hat = self.forward(X, x)
        loss = self.train_metric(y_hat, y)

        if self.parameter_loss:
            if self.reference_parameters is None:
                raise ValueError("No reference state dict provided to create parameter loss")
            par_loss = 0

            par1_ls, par2_ls = [], []
            for par1, par2 in zip(self.model.parameters(), self.reference_parameters):
                if par1.requires_grad:
                    par1_ls.append(par1.flatten())
                    par2_ls.append(par2.flatten())
            par1_flat = torch.cat(par1_ls)
            par2_flat = torch.cat(par2_ls)
            par_loss = self.alpha*self.parameter_metric(par1_flat.to(self.device), par2_flat.to(self.device))
            loss += par_loss
            self.log('par_loss', self.parameter_metric, logger=True, on_epoch=True, prog_bar=True, on_step=True)

        if self.zero_loss is not None:
            zero_idx = y_hat == -1
            a = y_hat.clone()
            b = y.clone()
            # a[~zero_idx] = 0
            # b[~zero_idx] = 0
            zero_loss = self.zero_loss(a[zero_idx],b[zero_idx])
            self.log('zero_loss', self.zero_loss, logger=True, on_step=True)
            loss += zero_loss

        self.log("loss", self.train_metric, logger=True, on_epoch=True, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        X, x, y = batch
        y_hat = self.forward(X, x)
        y_hat[y_hat < -1] = -1
        loss = self.valid_metric(y_hat, y)
        self.log("val_loss", self.valid_metric, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.y.append(y)
        self.y_hat.append(y_hat)
        self.x_attr.append(batch[1])
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        X, x, y = batch
        y_hat = self.forward(X, x)
        y_hat[y_hat < -1] = -1
        loss = self.valid_metric(y_hat, y)
        self.log("test_loss", self.valid_metric, on_epoch=True, logger=True, sync_dist=True)
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
            self.other_metrics(y_hat[:, SIS_idx].flatten().cuda(), y[:, SIS_idx].flatten().cuda())
        else:
            SIS_error = error
            self.other_metrics(y_hat.flatten().cuda(), y.flatten().cuda())

        self.log_dict(self.other_metrics, logger=True, sync_dist=True)

        world_size = self.trainer.world_size
        x_attr_all = self.all_gather(x_attr).view(world_size*x_attr.shape[0], *x_attr.shape[1:])
        y_all = self.all_gather(y).view(world_size*y.shape[0], *y.shape[1:])
        y_hat_all = self.all_gather(y_hat).view(world_size*y_hat.shape[0], *y_hat.shape[1:])
        SIS_error_all = self.all_gather(SIS_error).view(world_size*SIS_error.shape[0], *SIS_error.shape[1:])

        if self.trainer.is_global_zero:
            
            if 'SZA' in self.x_features and 'dayofyear' in self.x_features:
                SZA = x_attr_all[:, self.x_features.index('SZA')]
                dayofyear = x_attr_all[:, self.x_features.index('dayofyear')]

                SZA_fig, _ = SZA_error_plot(SZA, SIS_error_all)
                dayofyear_fig, _ = dayofyear_error_plot(dayofyear, SIS_error_all)
                if isinstance(self.logger, WandbLogger):
                    self.logger.log_image(key="y_hat - y, distribution SZA and dayofyear groundstations", images=[SZA_fig, dayofyear_fig])
        
            figs = []
            if len(self.y_vars) > 1:
                for i in range(y_all.shape[1]):
                    fig = prediction_error_plot(y_all[:, i].cpu(), y_hat_all[:, i].cpu(), self.y_vars[i])
                    figs.append(fig)
            else:
                fig = prediction_error_plot(y_all.cpu(), y_hat_all.cpu(), self.y_vars[0])
                figs.append(fig)
            
            if isinstance(self.logger, WandbLogger):
                self.logger.log_image(key="Prediction error groundstations", images=figs)
            plt.close()
    
    def predict_step(self, batch, batch_idx=0):
        X, x, y = batch
        y_hat = self.forward(X, x)
        y_hat[y_hat < -1] = -1

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
            self.other_metrics(y_hat[:, SIS_idx].cuda(), y[:, SIS_idx].cuda())
        else:
            SIS_error = error
            self.other_metrics(y_hat.flatten().cuda(), y.flatten().cuda())

        self.log_dict(self.other_metrics, logger=True, sync_dist=True)
        
        world_size = self.trainer.world_size
        x_attr_all = self.all_gather(x_attr).view(world_size*x_attr.shape[0], *x_attr.shape[1:]).cpu().squeeze()
        y_all = self.all_gather(y).view(world_size*y.shape[0], *y.shape[1:]).cpu().squeeze()
        y_hat_all = self.all_gather(y_hat).view(world_size*y_hat.shape[0], *y_hat.shape[1:]).cpu().squeeze()
        SIS_error_all = self.all_gather(SIS_error).view(world_size*SIS_error.shape[0], *SIS_error.shape[1:]).cpu().squeeze()
        
        if self.trainer.is_global_zero: 
            

            if 'SZA' in self.x_features and 'dayofyear' in self.x_features:
                SZA = x_attr_all[:, self.x_features.index('SZA')]
                dayofyear = x_attr_all[:, self.x_features.index('dayofyear')]

                SZA_fig, _ = SZA_error_plot(SZA, SIS_error_all)
                dayofyear_fig, _ = dayofyear_error_plot(dayofyear, SIS_error_all)
                if isinstance(self.logger, WandbLogger):
                    self.logger.log_image(key="y_hat - y, distribution SZA and dayofyear", images=[SZA_fig, dayofyear_fig])

            if 'lat' in self.x_features and 'lon' in self.x_features:
                lat = x_attr_all[:, self.x_features.index('lat')]
                lon = x_attr_all[:, self.x_features.index('lon')]

                fig, _ = latlon_error_plot(lat.cpu(), lon.cpu(), SIS_error_all)
                if isinstance(self.logger, WandbLogger):
                    self.logger.log_image(key="Error Location", images=[fig])
                plt.close()

            figs = []
            if len(self.y_vars) > 1:
                for i in range(y_all.shape[1]):
                    fig = prediction_error_plot(y_all[:, i].cpu(), y_hat_all[:, i].cpu(), self.y_vars[i])
                    figs.append(fig)
            else:
                fig = prediction_error_plot(y_all.cpu(), y_hat_all.cpu(), self.y_vars[0])
                figs.append(fig)

            if isinstance(self.logger, WandbLogger):
                self.logger.log_image(key="Prediction error SARAH3", images=figs)
        
            plt.close()
        
    
    def on_validation_epoch_start(self):
        self.y = []
        self.y_hat = []
        self.x_attr = []    

    def forward(self, X, x_attrs):
        return self.model(X.float(),x_attrs.float())
  

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.lr,
                                    #   betas=(0.5, 0.9), 
                                      weight_decay=0.1)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=0, factor=0.1, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_loss/dataloader_idx_0",
                "interval": "step",
                "frequency": 0.051*2067,
            },
        }
        
        
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": reduce_lr,
        #         "monitor": "val_loss/dataloader_idx_0",
        #         "frequency": 1,
        #     },
        # }