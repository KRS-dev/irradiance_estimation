import traceback
from dataset.station_dataset import GroundstationDataset
import matplotlib.pyplot as plt
from models.FCN import residual_FCN
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import wandb
import xarray
from dataset.dataset import ImageDataset, SeviriDataset, valid_test_split, pickle_read
from dataset.normalization import MinMax, ZeroMinMax
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from models.ConvResNet_Jiang import ConvResNet, ConvResNet_batchnormMLP, ConvResNet_dropout, ConvResNet_inputCdropout, ConvResNet_BNdropout
from models.LightningModule import LitEstimator, LitEstimatorPoint
from tqdm import tqdm

# from pytorch_lightning.pytorch.callbacks import DeviceStatsMonitor
from types import SimpleNamespace

config = {
    "batch_size": 1024,
    "patch_size": {
        "x": 3,
        "y": 3,
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
    'max_epochs': 50,
    # Compute related
    'num_workers': 24,
    'ACCELERATOR': "gpu",
    'DEVICES': -1,
    'NUM_NODES': 32,
    'STRATEGY': "ddp",
    'PRECISION': "32",
    'EarlyStopping': {'patience':5},
    'ModelCheckpoint':{'every_n_epochs':1, 'save_top_k':3}
}
config = SimpleNamespace(**config)

def get_dataloaders(config):
    
    timeindex = pd.DatetimeIndex(pickle_read('/scratch/snx3000/kschuurm/ZARR/timeindices.pkl'))
    # timeindex = timeindex[(timeindex.hour >10) & (timeindex.hour <17)]
    traintimeindex = timeindex[(timeindex.year <= 2021)]
    # _, validtimeindex = valid_test_split(timeindex[(timeindex.year == 2017)])
    validtimeindex= timeindex[(timeindex.year==2022)]

    rng = np.random.default_rng(seed=420)
    train_dataset = SeviriDataset(
        x_vars=config.x_vars,
        y_vars=config.y_vars,
        x_features=config.x_features,
        patch_size=config.patch_size,
        transform=config.transform,
        target_transform=config.target_transform,
        patches_per_image=config.batch_size,
        timeindices=traintimeindex,
        rng=rng,
    )
    valid_dataset = SeviriDataset(
        x_vars=config.x_vars,
        y_vars=config.y_vars,
        x_features=config.x_features,
        patch_size=config.patch_size,
        transform=config.transform,
        target_transform=config.target_transform,
        patches_per_image=config.batch_size,
        timeindices=validtimeindex,
        seed=0,
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
       
    # model = ConvResNet_BNdropout(
    #     num_attr=len(config.x_features),
    #     input_channels=len(config.x_vars),
    #     output_channels=len(config.y_vars),
    # )
    model = residual_FCN(
        patch_size=(config.patch_size['x'], config.patch_size['y']),
        input_channels=len(config.x_vars),
        input_features=len(config.x_features),
        channel_size=256,
        output_channels=len(config.y_vars)
    )
    config.model = type(model).__name__
    wandb_logger = WandbLogger(project="SIS_point_estimation", log_model=True)

    if rank_zero_only.rank == 0:  # only update the wandb.config on the rank 0 process
        wandb_logger.experiment.config.update(vars(config))

    mc_sarah = ModelCheckpoint(
        monitor='val_loss', 
        every_n_epochs=config.ModelCheckpoint['every_n_epochs'], 
        save_top_k = config.ModelCheckpoint['save_top_k'],
        filename='{epoch}-{val_loss:.5f}'
    ) 
    early_stopping = EarlyStopping(monitor='val_loss', patience=config.EarlyStopping['patience'])

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
        callbacks=[early_stopping, mc_sarah],
        max_time="00:02:00:00"
    )

    testtrainer = Trainer(
        logger=wandb_logger,
        accelerator=config.ACCELERATOR,
        devices=1, num_nodes=1)

    estimator = LitEstimatorPoint(
        model=model,
        learning_rate=0.0001,
        config=config,
    )

    train_dataloader, val_dataloader = get_dataloaders(config)

    trainer.fit(
        estimator, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    test_dataloaders = get_testdataloader(config)
    try:
        for dl in test_dataloaders.values():
            testtrainer.test(
                estimator, dataloaders=dl
            )
    except Exception as e:
        print('failed test')
        traceback.print_exc()
    finally:
        wandb.finish()

if __name__ == "__main__":
    
    main()


