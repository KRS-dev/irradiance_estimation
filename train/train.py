import os
import traceback
from dataset.station_dataset import GroundstationDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import wandb
import xarray
from dataset.dataset import ImageDataset, SeviriDataset, valid_test_split, pickle_read
from dataset.normalization import ZeroMinMax
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from models.LightningModule import LitEstimatorPoint
from tqdm import tqdm

from types import SimpleNamespace

os.environ["WANDB__SERVICE_WAIT"] = "30"

def get_dataloaders(config):
    

    train_dataset = SeviriDataset(
        x_vars=config.x_vars,
        y_vars=config.y_vars,
        x_features=config.x_features,
        patch_size=config.patch_size,
        transform=config.transform,
        target_transform=config.target_transform,
        patches_per_image=config.batch_size,
    )
    valid_dataset = SeviriDataset(
        x_vars=config.x_vars,
        y_vars=config.y_vars,
        x_features=config.x_features,
        patch_size=config.patch_size,
        transform=config.transform,
        target_transform=config.target_transform,
        patches_per_image=2048,
        validation=True,
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=None, num_workers=config.num_workers)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=None, num_workers=config.num_workers)

    return train_dataloader, valid_dataloader

def get_testdataloader(config):
    stations = ['CAB', 'CAR', 'CEN' ,'MIL', 'NOR', 'PAL', 'PAY', 'TAB', 'TOR', 'VIS']

    test_datasets = [GroundstationDataset(nm, 
                                    config.y_vars, 
                                    config.x_vars, 
                                    config.x_features, 
                                    patch_size=config.patch_size['x'],
                                    transform=config.transform,
                                    target_transform=config.target_transform) 
                for nm in stations] 

    test_dataloaders = {nm: DataLoader(ds, batch_size=10000, shuffle=False) for nm, ds in zip(stations, test_datasets)}
    return test_dataloaders


def main():

    config = {
        "batch_size": 128,
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
        'max_epochs': 5,
        # Compute related
        'num_workers': 24,
        'ACCELERATOR': "gpu",   
        'DEVICES': -1,
        'NUM_NODES': 16,
        'STRATEGY': "ddp",
        'PRECISION': "32",
        'EarlyStopping': {'patience':3},
        'ModelCheckpoint':{'every_n_epochs':1, 'save_top_k':3},
        # 'ckpt_fn': '/scratch/snx3000/kschuurm/irradiance_estimation/train/SIS_point_estimation/m4ms61hn/checkpoints/last.ckpt',
    }
    config = SimpleNamespace(**config)


    estimator = LitEstimatorPoint(
        learning_rate=0.001,
        config=config,
        metric=MeanSquaredError(),
    )

    config.model = type(estimator.model).__name__

    # wandb_logger = WandbLogger(name='Emulator', project="SIS_point_estimation", log_model=True)
    wandb_logger = WandbLogger(name='Emulator 3', project="SIS_point_estimation",)# id='m4ms61hn', resume='must')


    # if rank_zero_only.rank == 0:  # only update the wandb.config on the rank 0 process
    #     wandb_logger.experiment.config.update(vars(config))

    mc_sarah = ModelCheckpoint(
        monitor='val_loss', 
        every_n_epochs=config.ModelCheckpoint['every_n_epochs'], 
        save_top_k = config.ModelCheckpoint['save_top_k'],
        filename='{epoch}-{val_loss:.5f}',
        save_last=True,
    ) 
    # early_stopping = EarlyStopping(monitor='val_loss', 
    #                                patience=config.EarlyStopping['patience'],
    #                                verbose=True,
    #                                min_delta=0,
    #                                log_rank_zero_only=True,
    #                                check_finite=True,)

    trainer =  Trainer(
        # fast_dev_run=True,
        logger=wandb_logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        max_epochs=config.max_epochs,
        precision=config.PRECISION,
        log_every_n_steps=500,
        strategy=config.STRATEGY,
        num_nodes=config.NUM_NODES,
        callbacks=[ mc_sarah],
        max_time="00:02:00:00",
    )


    train_dataloader, val_dataloader = get_dataloaders(config)

    trainer.fit(
        estimator, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
        # ckpt_path=config.ckpt_fn,
    )


    if rank_zero_only.rank == 0:
        print('Best model:', mc_sarah.best_model_path)
        wandb_logger.experiment.finish()
    

if __name__ == "__main__":
    
    main()


