import argparse
import pytorch_lightning as pl
from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier
from train_config import *
import os


def get_latest_ckpt(folder_path):
    # 获取所有.ckpt文件并排除目录
    ckpt_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.ckpt') and os.path.isfile(os.path.join(folder_path, f))
    ]

    if not ckpt_files:
        return None  # 没有找到.ckpt文件

    # 返回修改时间最新的文件
    return max(ckpt_files, key=os.path.getmtime)


# 示例用法
folder = 'checkpoints'
latest_ckpt = get_latest_ckpt(folder)
parser = argparse.ArgumentParser()

parser.add_argument(
    "-s", "--seed", default=1234, type=int, help="Set seed for training"
)

args = parser.parse_args()
seed = args.seed

set_seed(seed)

model = ParametersClassifier.load_from_checkpoint(
    checkpoint_path=latest_ckpt,
    num_classes=3,
    lr=INITIAL_LR,
    gpus=1,
    transfer=False,
    retrieve_masks=True
)
model.eval()

data = ParametersDataModule(
    batch_size=BATCH_SIZE,
    data_dir=DATA_DIR,
    csv_file=DATA_CSV,
    image_dim=(320, 320),
    dataset_name=DATASET_NAME,
    mean=DATASET_MEAN,
    std=DATASET_STD,
    transform=False,
)
data.setup('test')

trainer = pl.Trainer(
    num_nodes=1,
    gpus=2,
    weights_summary=None,
    precision=16,
)

trainer.test(model, datamodule=data)