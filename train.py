import os
import argparse
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from caxton.src.data.data_module import ParametersDataModule
from caxton.src.model.network_module import ParametersClassifier
from train_config import *
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-s", "--seed", default=123, type=int, help="Set seed for training"
)
    parser.add_argument(
    "-e",
    "--epochs",
    default=MAX_EPOCHS,
    type=int,
    help="Number of epochs to train the model for",
)

    args = parser.parse_args()
    seed = args.seed

    set_seed(seed)
    logs_dir = "logs/logs-{}/{}/".format(DATE, seed)
    logs_dir_default = os.path.join(logs_dir, "default")

    make_dirs(logs_dir)
    make_dirs(logs_dir_default)
    csv_logger = CSVLogger(
        save_dir=logs_dir,
        name="csv_logs",  # 日志会保存在logs_dir/csv_logs/version_x目录下
        version=f"seed_{seed}")
    tb_logger = pl_loggers.TensorBoardLogger(logs_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best_model-{epoch}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    model = ParametersClassifier(
    num_classes=3,
    lr=INITIAL_LR,
    gpus=NUM_GPUS,
    transfer=False,
    retrieve_layers=False,
    retrieve_masks=False,
)
    data = ParametersDataModule(
    batch_size=BATCH_SIZE,
    data_dir=DATA_DIR,
    csv_file=DATA_CSV,
    dataset_name=DATASET_NAME,
    mean=DATASET_MEAN,
    std=DATASET_STD,
)

    trainer = pl.Trainer(
    num_nodes=NUM_NODES,
    gpus=NUM_GPUS,
    progress_bar_refresh_rate=1,
    distributed_backend=ACCELERATOR,
    max_epochs=args.epochs,
    logger=[tb_logger, csv_logger],
    weights_summary=None,
    precision=16,
    callbacks=[checkpoint_callback],
)


    trainer.fit(model, data)
    # 测试流程（自动加载最佳模型）
    model.save_training_history("custom_history.csv")