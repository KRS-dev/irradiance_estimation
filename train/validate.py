import itertools
from multiprocessing import Pool
import os
import traceback
from dataset.station_dataset import GroundstationDataset, GroundstationDataset2
from lightning import LightningDataModule
import matplotlib.pyplot as plt
from models.FCN import residual_FCN
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from utils.etc import benchmark
import wandb
import xarray
from dataset.dataset import ImageDataset, SeviriDataset, valid_test_split, pickle_read
from dataset.normalization import MinMax, ZeroMinMax
from train import get_dataloaders
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from models.ConvResNet_Jiang import ConvResNet, ConvResNet_batchnormMLP, ConvResNet_dropout, ConvResNet_inputCdropout, ConvResNet_BNdropout
from models.LightningModule import LitEstimator, LitEstimatorPoint
from tqdm import tqdm

# from pytorch_lightning.pytorch.callbacks import DeviceStatsMonitor
from types import SimpleNamespace

class DataModule(LightningDataModule):

    def __init__(self, train_dataset, val_dataset,  batch_size = 2):

        super(DataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)

def create_gsdataset(args):
    nm, config = args
    return GroundstationDataset2(f'../../ZARR/DWD/DWD_SOLAR_{str(nm).zfill(5)}.zarr', config.y_vars, config.x_vars, config.x_features, config.patch_size['x'], config.transform, config.target_transform)


def main():

    config = {
        "batch_size": 2048,
        "patch_size": {
            "x": 15,
            "y": 15,
            "stride_x": 1,
            "stride_y": 1,
        },
        "x_vars": [
            "channel_1",
            "channel_2",
            "channel_3",
            "channel_4",
            "channel_5",
            "channel_6",
            "channel_7",
            "channel_8",
            "channel_9",
            "channel_10",
            "channel_11",
            "DEM",
        ],
        "y_vars": ["SIS",],
        "x_features": ["dayofyear", "lat", "lon", 'SZA', "AZI",],
        "transform": ZeroMinMax(),
        "target_transform": ZeroMinMax(),
        'max_epochs': 20,
        # Compute related
        'num_workers': 24,
        'ACCELERATOR': "gpu",
        'DEVICES': -1,
        'NUM_NODES': 2,
        'STRATEGY': "ddp",
        'PRECISION': "32",
        'EarlyStopping': {'patience':3},
        'ModelCheckpoint':{'every_n_epochs':1, 'save_top_k':1},
    }
    config = SimpleNamespace(**config)

       
    model = ConvResNet_batchnormMLP(
        num_attr=len(config.x_features),
        input_channels=len(config.x_vars),
        output_channels=len(config.y_vars),
    )

    resume_ckpt_fn = '/scratch/snx3000/kschuurm/irradiance_estimation/train/SIS_point_estimation_groundstation/groundstations_only/checkpoints/epoch=10-val_loss=0.01703.ckpt'

    config.model = type(model).__name__

    wandb_logger = WandbLogger(project="SIS_point_estimation", id='groundstations_only',resume=True, log_model=False)

    mc_sarah = ModelCheckpoint(
        monitor='val_loss', 
        every_n_epochs=config.ModelCheckpoint['every_n_epochs'], 
        save_top_k = config.ModelCheckpoint['save_top_k'],
        filename='{epoch}-{val_loss:.5f}'
    ) 
    # early_stopping = EarlyStopping(monitor='val_loss', patience=config.EarlyStopping['patience'])

    trainer =  Trainer(
        logger=wandb_logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.max_epochs,
        precision=config.PRECISION,
        log_every_n_steps=500,
        strategy=config.STRATEGY,
        num_nodes=config.NUM_NODES,
        callbacks=[mc_sarah],
        max_time="00:02:00:00"
    )
    
    model = ConvResNet_batchnormMLP(
        num_attr=len(config.x_features),
        input_channels=len(config.x_vars),
        output_channels=len(config.y_vars),
    )
    estimator = LitEstimatorPoint(
        model=model,
        learning_rate=0.00001,
        config=config,
        metric=MeanSquaredError(),
    )
    ch = torch.load(resume_ckpt_fn, map_location=torch.device('cuda'))
    estimator.load_state_dict(ch['state_dict'])
    
    val_dataloader_sarah = get_dataloaders(config)
    trainer.validate(estimator, val_dataloader_sarah, ckpt_path=resume_ckpt_fn)

if __name__ == "__main__":
    
    main()


