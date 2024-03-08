import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import wandb
import xarray
from dataset.dataset import ImageDataset, valid_test_split
from dataset.normalization import MinMax, ZeroMinMax
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities import rank_zero_only
from models.ConvResNet_Jiang import ConvResNet, ConvResNet_dropout
from models.LightningModule import LitEstimator, LitEstimatorPoint
from tqdm import tqdm

# from pytorch_lightning.pytorch.callbacks import DeviceStatsMonitor
from types import SimpleNamespace

config = {
    "batch_size": 2048,
    "patch_size": {
        "x": 15,
        "y": 15,
        "stride_x": 4,
        "stride_y": 4,
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
    "y_vars": ["SIS"],
    "x_features": ["dayofyear", "lat", "lon", 'SZA', "AZI", "DEM"],
    "transform": ZeroMinMax(),
    "target_transform": ZeroMinMax(),
    # Compute related
    'ACCELERATOR': "gpu",
    'DEVICES': -1,
    'NUM_NODES': 1,
    'STRATEGY': "ddp",
    'PRECISION': "32",
}
config = SimpleNamespace(**config)


def main():

    sarah_bnds = xarray.open_zarr('/scratch/snx3000/kschuurm/ZARR/SARAH3_bnds.zarr').load()
    sarah_bnds = sarah_bnds.isel(time = sarah_bnds.pixel_count != -1)
    seviri = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/SEVIRI_new.zarr")
    seviri_time = pd.DatetimeIndex(seviri.time)
    timeindex= pd.DatetimeIndex(sarah_bnds.time)
    timeindex = timeindex.intersection(seviri_time)
    timeindex = timeindex[(timeindex.hour >10) & (timeindex.hour <17)]

    _, traintimeindex = valid_test_split(timeindex[(timeindex.year == 2016)])
    _, validtimeindex = valid_test_split(timeindex[(timeindex.year == 2017)])

    train_dataset = ImageDataset(
        x_vars=config.x_vars,
        y_vars=config.y_vars,
        x_features=config.x_features,
        patch_size=config.patch_size,
        transform=config.transform,
        target_transform=config.target_transform,
        timeindices=traintimeindex,
        # random_sample=1,
        batch_in_time=10,
    )
    valid_dataset = ImageDataset(
        x_vars=config.x_vars,
        y_vars=config.y_vars,
        x_features=config.x_features,
        patch_size={
            "x": config.patch_size["x"],
            "y": config.patch_size["y"],
            "stride_x": 4,
            "stride_y": 4,
        },
        transform=config.transform,
        target_transform=config.target_transform,
        timeindices=validtimeindex[::10],
        random_sample=None,
        shuffle=False,
        batch_in_time=10,
    )
    train_dataloaders = DataLoader(train_dataset, shuffle=False, num_workers=0)
    valid_dataloaders = DataLoader(valid_dataset, shuffle=False, num_workers=0)

    model = ConvResNet(
        num_attr=len(config.x_features),
        input_channels=len(config.x_vars),
        output_channels=len(config.y_vars),
    )
    wandb_logger = WandbLogger(project="SIS_point_estimation", log_model=True)

    if rank_zero_only.rank == 0:  # only update the wandb.config on the rank 0 process
        wandb_logger.experiment.config.update(vars(config))

    trainer = Trainer(
        # profiler="simple",
        # fast_dev_run=True,
        num_sanity_val_steps=2,
        logger=wandb_logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=4,
        precision=config.PRECISION,
        log_every_n_steps=500,
        val_check_interval=0.25,
    )

    estimator = LitEstimatorPoint(
        model=model,
        learning_rate=0.001,
        config=config,
    )
    trainer.fit(
        estimator, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )

    wandb.finish()

if __name__ == "__main__":
    
    main()

