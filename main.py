from pytorch_lightning import Trainer
from dataset.dataset import MSGDataModule
from models.FNO import FNO2d
from models.auto_encoder_decoder import LitEstimator
from train import config


if __name__ == "__main__":
    # wandb_logger = WandbLogger(project="SIS_estimation")
    dm = MSGDataModule(batch_size=config.BATCH_SIZE, patch_size=config.INPUT_SIZE)

    # model = FNO2d_estimation(
    #     n_modes_height = 5,
    #     n_modes_width = 5,
    #     hidden_channels = 2,
    #     num_workers=24,
    # )
    model = FNO2d(modes=(8,8), input_channels=config.INPUT_CHANNELS, output_channels=config.OUTPUT_CHANNELS, channels=20),

    dm.setup('fit')
    dl = dm.train_dataloader()
    print(dl)
    for x, y in dl:
        print(x.shape)
        yhat = model(x)
        print(model(x))
        print(yhat.shape)
        break

    # estimator = LitEstimator(
    #     model = model,
    #     learning_rate = config.LEARNING_RATE,
    # )
    # trainer = Trainer(
    #     fast_dev_run=True,
    #     num_sanity_val_steps=2,
    #     # logger=wandb_logger,
    #     accelerator=config.ACCELERATOR,
    #     devices=config.DEVICES,
    #     min_epochs=config.MIN_EPOCHS,
    #     max_epochs=config.MAX_EPOCHS,
    #     precision=config.PRECISION,
    #     log_every_n_steps=1,
    # )
    # trainer.fit(model=estimator, train_dataloaders=dm)
