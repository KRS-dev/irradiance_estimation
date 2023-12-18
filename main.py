import torch
import wandb
from dataset.dataset import MSGDataModule, MSGDataModulePoint
from dataset.normalization import MinMax
from models.ConvResNet_Jiang import ConvResNet
from models.FNO import FNO2d
from models.LightningModule import LitEstimator, LitEstimatorPoint
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities import rank_zero_only

# from pytorch_lightning.pytorch.callbacks import DeviceStatsMonitor
from train import config
from utils.plotting import best_worst_plot, prediction_error_plot

# Training Hyperparameters
PATCH_SIZE = (15, 15)
PATCH_OVERLAP = {'lat':10,'lon':10} # dataset sampled with overlapping (15,15) patches
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
NUM_EPOCHS = 20
MIN_EPOCHS = 1
MAX_EPOCHS = 30

CHPT = './SIS_point_estimation/ddmsa5e9/checkpoints/epoch=29-step=19680.ckpt'

# Dataset
# DATA_DIR
NUM_WORKERS = 12
X_VARS = ["channel_1", "channel_7"]
Y_VARS = ["SIS"]

# Compute related
ACCELERATOR = "gpu"
DEVICES = -1
NUM_NODES = 16
STRATEGY = "ddp"
PRECISION = 32



def main():
    dm = MSGDataModulePoint(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        patch_size=PATCH_SIZE,
        input_overlap= PATCH_OVERLAP,
        x_vars=X_VARS,
        y_vars=Y_VARS,
        transform=MinMax(),
        target_transform=MinMax(),
    )

    model = ConvResNet(num_attr=5, input_channels=len(X_VARS))

    if CHPT is not None:
        estimator = LitEstimatorPoint.load_from_checkpoint(
            CHPT,
            model=model,
            learning_rate=LEARNING_RATE,
            dm=dm,
        )
    else:
        estimator = LitEstimatorPoint(
            model=model,
            learning_rate=LEARNING_RATE,
            dm=dm,
        )


    wandb_logger = WandbLogger(project="SIS_point_estimation", log_model=True)
    
    if rank_zero_only.rank == 0: # only update the wandb.config on the rank 0 process
        wandb_logger.experiment.config.update({
            "PATCH_SIZE":PATCH_SIZE,
            "PATCH_OVERLAP":PATCH_OVERLAP,
            "X_VARS":X_VARS,
            "Y_VARS":Y_VARS,
            "TRANSFORM": [str(dm.transform)],
            "CHPT_START": CHPT,
        })

    trainer = Trainer(
        # profiler="simple",
        # fast_dev_run=True,
        num_sanity_val_steps=2,
        logger=wandb_logger,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        num_nodes=NUM_NODES,
        strategy=STRATEGY,
        min_epochs=MIN_EPOCHS,
        max_epochs=MAX_EPOCHS,
        precision=PRECISION,
        log_every_n_steps=200,
        # plugins=[SLURMEnvironment(auto_requeue=False)],
    )

    trainer.fit(model=estimator, train_dataloaders=dm)


if __name__ == "__main__":
    
    main()

