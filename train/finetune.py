from glob import glob
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
        'NUM_NODES': 1,
        'STRATEGY': "ddp",
        'PRECISION': "32",
        'EarlyStopping': {'patience':5},
        'ModelCheckpoint':{'every_n_epochs':1, 'save_top_k':-1},
        'ckpt_fn':'/scratch/snx3000/kschuurm/irradiance_estimation/train/SIS_point_estimation/4nbyae30/checkpoints/epoch=7-val_loss=0.01023.ckpt',
        'train_id':  [15000, 2638, 662, 342, 691, 4104, 1684, 5426, 1766, 3167, 596, 880, 1346, 4271, 1550, 3196, 5792, 2485, 856, 1468, 3287, 4336, 701, 3126, 891, 1078, 4393, 963, 5705, 5546, 7368, 4887, 164, 704, 2261, 656, 2559, 6197, 3513, 3032, 7351, 430, 1443, 2907, 5856, 5404, 6163, 2483, 3268, 2601, 15444, 13674, 7374, 5480, 7367, 4745, 2014, 4625, 5100, 3761, 460, 7369, 3086, 3366, 282, 591, 1639, 232, 4177, 7370, 2667, 4928, 2712, 4466, 5397, 5516, 1975, 1503, 2115, 1605],
        'valid_id': [1757, 5109, 953, 3028, 2290, 5906, 2171, 427, 2932, 2812, 5839, 1691, 3811, 1420, 5142, 4911, 3660, 3730, 1048],

    }
    config = SimpleNamespace(**config)

       
    model = ConvResNet_batchnormMLP(
        num_attr=len(config.x_features),
        input_channels=len(config.x_vars),
        output_channels=len(config.y_vars),
    )

    config.model = type(model).__name__
    wandb_logger = WandbLogger(project="SIS_point_estimation_groundstation", log_model=True)

    if rank_zero_only.rank == 0:  # only update the wandb.config on the rank 0 process
        wandb_logger.experiment.config.update(vars(config))



    mc_sarah = ModelCheckpoint(
        monitor='val_loss', 
        every_n_epochs=config.ModelCheckpoint['every_n_epochs'], 
        save_top_k = config.ModelCheckpoint['save_top_k'],
        filename='{epoch}-{val_loss:.5f}'
    ) 

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
        max_time="00:03:00:00"
    )
    
    model = ConvResNet_batchnormMLP(
        num_attr=len(config.x_features),
        input_channels=len(config.x_vars),
        output_channels=len(config.y_vars),
    )
    estimator = LitEstimatorPoint(
        model=model,
        learning_rate=0.0001,
        config=config,
        metric=MeanSquaredError(),
        parameter_loss=True,
        alpha=3000
    )
    ch = torch.load(config.ckpt_fn, map_location=torch.device('cuda'))
    estimator.load_state_dict(ch['state_dict'])
    estimator.set_reference_parameters([par.clone().detach() for par in estimator.model.parameters()])

    with benchmark('datasets'):
        valid_datasets = tqdm(map(create_gsdataset, zip(config.valid_id, itertools.repeat(config))))
        train_datasets = tqdm(map(create_gsdataset, zip(config.train_id, itertools.repeat(config))))
        
        # zarr_fns = glob('../../ZARR/IEA_PVPS/IEA_PVPS-*.zarr')
        # station_names_bsrn = [os.path.basename(fn).split('IEA_PVPS-')[-1].split('.')[0] for fn in zarr_fns]
        # bsrn_datasets = [GroundstationDataset2(f'../../ZARR/IEA_PVPS/IEA_PVPS-{x}.zarr', 
        #                                 config.y_vars, config.x_vars, config.x_features, config.patch_size['x'], 
        #                                 config.transform, config.target_transform, sarah_idx_only=True)
        #                     for x in tqdm(station_names_bsrn)]
        # train_datasets = bsrn_datasets
    
    valid_dataset = ConcatDataset(valid_datasets)
    train_dataset = ConcatDataset(train_datasets)

    dm = DataModule(train_dataset, valid_dataset, config.batch_size)

    trainer.validate(estimator, dm.val_dataloader())
    trainer.fit(estimator, dm)
    
  

if __name__ == "__main__":
    
    main()


