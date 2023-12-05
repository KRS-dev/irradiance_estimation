from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from irradiance_estimation.dataset.dataset_old import MSGDataModule
from models.FNO import FNO2d
from models.auto_encoder_decoder import LitEstimator
from train import config
from preprocess.etc import benchmark
from dask.distributed import Client, LocalCluster


if __name__ == "__main__":

    wandb_logger = WandbLogger(project="SIS_estimation")
    dm = MSGDataModule(batch_size=config.BATCH_SIZE, num_workers=24, patch_size=config.INPUT_SIZE)


    model = FNO2d(modes=(16,16), input_channels=config.INPUT_CHANNELS, output_channels=config.OUTPUT_CHANNELS, channels=20)
    
    print(model.parameters())
    estimator = LitEstimator(
        model = model,
        learning_rate = config.LEARNING_RATE,
    )
    trainer = Trainer(
        logger=wandb_logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION,
        log_every_n_steps=1,
    )
    trainer.fit(model=estimator, train_dataloaders=dm)
