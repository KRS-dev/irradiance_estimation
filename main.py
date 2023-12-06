from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from dataset.dataset import MSGDataModule
from models.FNO import FNO2d
from models.LightningModule import LitEstimator

# from pytorch_lightning.pytorch.callbacks import DeviceStatsMonitor
from train import config

if __name__ == "__main__":
    print('start')
    wandb_logger = WandbLogger(project="SIS_estimation")
    dm = MSGDataModule(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, patch_size=config.INPUT_SIZE)

    print('dm')

    model = FNO2d(modes=(24,24), input_channels=config.INPUT_CHANNELS, output_channels=config.OUTPUT_CHANNELS, channels=10)
    
    print(model.parameters())
    estimator = LitEstimator(
        model = model,
        learning_rate = config.LEARNING_RATE,
    )
    print('model')
    trainer = Trainer(
        # profiler='simple',
        # callbacks=[DeviceStatsMonitor(cpu_stats=true)]
        num_sanity_val_steps=2,
        logger=wandb_logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION,
        log_every_n_steps=1,
    )
    trainer.fit(model=estimator, train_dataloaders=dm)

