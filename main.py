import torch
from lightning import Trainer
from dataset.dataset import MSGDataModule
from models.FNO import FNO2d_estimation
from models.auto_encoder_decoder import LitEstimator
from train import config

if __name__ == "__main__":
    # wandb_logger = WandbLogger(project="SIS_estimation")

    dm = MSGDataModule(batch_size=config.BATCH_SIZE)

    model = FNO2d_estimation(
        n_modes_height =10,
        n_modes_width = 10,
        hidden_channels = 20,
    )

    LitEstimator(
        learning_rate = config.LEARNING_RATE,
        model = model,
    )

    trainer = Trainer(
        fast_dev_run=True,
        # logger=wandb_logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION,
    )

    trainer.fit(model=model, train_dataloaders=dm)
