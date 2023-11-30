import torch
from lightning import Trainer, WandbLogger

from dataset import MSGDataModule
import config


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="SIS_estimation")

    dm = MSGDataModule(batch_size=config.BATCH_SIZE)

    trainer = Trainer(
        logger=wandb_logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochts=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION,
    )
