import torch, wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


from dataset.dataset import MSGDataModule, MSGDataModulePoint
from dataset.normalization import MinMax
from models.FNO import FNO2d
from models.ConvResNet_Jiang import ConvResNet
from models.LightningModule import LitEstimator, LitEstimatorPoint


# from pytorch_lightning.pytorch.callbacks import DeviceStatsMonitor
from train import config
from utils.plotting import best_worst_plot, prediction_error_plot

if __name__ == "__main__":
    # wandb_logger = WandbLogger(project="SIS_estimation")
    
    dm = MSGDataModulePoint(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        patch_size=config.INPUT_SIZE,
        x_vars=['channel_1'],
        transform=MinMax(),
        # target_transform=MinMax(),
    )

    model = ConvResNet(num_attr=5)

    estimator = LitEstimatorPoint(
        model=model,
        learning_rate=config.LEARNING_RATE,
        dm=dm,
    )

    wandb_logger = WandbLogger(project="SIS_point_estimation")

    trainer = Trainer(
        # profiler="simple",
        # fast_dev_run=True,
        # callbacks=[DeviceStatsMonitor(cpu_stats=true)]
        num_sanity_val_steps=2,
        logger=wandb_logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION,
        log_every_n_steps=100,
    )
    trainer.fit(model=estimator, train_dataloaders=dm)
