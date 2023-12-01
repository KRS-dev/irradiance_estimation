from pytorch_lightning import Trainer
from dataset import MSGDataModule
from models import FNO2d_estimation, LitEstimator
import config


if __name__ == "__main__":
    # wandb_logger = WandbLogger(project="SIS_estimation")

    dm = MSGDataModule(batch_size=config.BATCH_SIZE)

    model = FNO2d_estimation(
        n_modes_height =10,
        n_modes_width =10,
        hidden_channels = 20,
    )

    LitEstimator(
        learning_rate = config.LEARNING_RATE,
        model = model,
    )

    trainer = Trainer(
        # logger=wandb_logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION,
    )

    
