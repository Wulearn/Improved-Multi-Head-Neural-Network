import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from caxton.src.model.residual_attention_network import (
    ResidualAttentionModel_56 as ResidualAttentionModel,
)
import pytorch_lightning as pl
from datetime import datetime
import pandas as pd
import os
from torchmetrics import Precision, Recall, F1Score
import numpy as np
from sklearn.metrics import classification_report
class MLP(nn.Module):
    def __init__(self, dim, regress=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, 8)
        self.fc2 = nn.Linear(8, 5)
        self.regress = regress
        if self.regress:
            self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.regress:
            x = self.fc3(x)
        return x

class ParametersClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr=1e-3,
        transfer=False,
        trainable_layers=1,
        gpus=1,
        retrieve_layers=False,
        retrieve_masks=False,
        test_overwrite_filename=False,
    ):
        super().__init__()
        self.lr = lr
        self.__dict__.update(locals())
        self.mlp = MLP(dim=5, regress=False)
        self.attention_model = ResidualAttentionModel(
            retrieve_layers=retrieve_layers, retrieve_masks=retrieve_masks
        )
        num_ftrs = self.attention_model.fc.in_features
        self.attention_model.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs+5, num_classes)
        self.fc2 = nn.Linear(num_ftrs+5, num_classes)
        self.fc3 = nn.Linear(num_ftrs+5, num_classes)
        self.fc4 = nn.Linear(num_ftrs+5, num_classes)

        if transfer:
            for child in list(self.attention_model.children())[:-trainable_layers]:
                for param in child.parameters():
                    param.requires_grad = False
        self.save_hyperparameters()

        self.train_acc = pl.metrics.Accuracy()
        self.train_acc0 = pl.metrics.Accuracy()
        self.train_acc1 = pl.metrics.Accuracy()
        self.train_acc2 = pl.metrics.Accuracy()
        self.train_acc3 = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.val_acc0 = pl.metrics.Accuracy()
        self.val_acc1 = pl.metrics.Accuracy()
        self.val_acc2 = pl.metrics.Accuracy()
        self.val_acc3 = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.name = "ResidualAttentionClassifier"
        self.retrieve_layers = retrieve_layers
        self.retrieve_masks = retrieve_masks
        self.gpus = gpus
        self.sync_dist = True if self.gpus > 1 else False
        self.test_overwrite_filename = test_overwrite_filename
        self.training_history = {'acc': [], 'acc0': [], 'acc1': [], 'acc2': [], 'acc3': [],
                                  'loss': [], 'loss0': [], 'loss1': [], 'loss2': [], 'loss3': []}
        self.validation_history = {'acc': [], 'acc0': [], 'acc1': [], 'acc2': [], 'acc3': [],
                                    'loss': [], 'loss0': [], 'loss1': [], 'loss2': [], 'loss3': []}
        self.metrics = {}
        for phase in ["train", "val", "test"]:
            for head in range(4):
                # 精确率
                self.metrics[f"{phase}_precision_head{head}"] = Precision(
                    num_classes=num_classes,
                    average='macro'  # 使用macro平均，也可以根据需求改为'micro'或'weighted'
                )
                # 召回率
                self.metrics[f"{phase}_recall_head{head}"] = Recall(
                    num_classes=num_classes,
                    average='macro'
                )
                # F1分数
                self.metrics[f"{phase}_f1_head{head}"] = F1Score(
                    num_classes=num_classes,
                    average='macro'
                )
        self.metrics = nn.ModuleDict(self.metrics)

    @staticmethod
    def _denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """反归一化图像张量"""
        tensor = tensor.clone().cpu()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor.clamp_(0, 1).permute(1, 2, 0)  # CHW -> HWC
    # def forward(self, X):
    #     # mlp_model = MLP(dim=5, regress=False)
    #     # parmeters = mlp_model(parmeters)
    #     # combination = torch.cat((X, parmeters), 1)
    #     if self.retrieve_layers or self.retrieve_masks:
    #         features = self.attention_model(X)
    #         main_output = features[0]  # 主输出
    #         additional_output = features[1:]  # 包含层或掩码
    #     else:
    #         main_output = self.attention_model(X)
    #         additional_output = None
    #
    #     out1 = self.fc1(main_output)
    #     out2 = self.fc2(main_output)
    #     out3 = self.fc3(main_output)
    #     out4 = self.fc4(main_output)
    #
    #     if self.retrieve_layers or self.retrieve_masks:
    #         return (out1, out2, out3, out4), additional_output
    #     return (out1, out2, out3, out4)
    def forward(self, image, parmeters):
        X = self.attention_model(image)
        if self.retrieve_layers or self.retrieve_masks:
            out1 = self.fc1(X[0])
            out2 = self.fc2(X[0])
            out3 = self.fc3(X[0])
            out4 = self.fc4(X[0])
            return (out1, out2, out3, out4), X
        parmeters = self.mlp(parmeters)
        combination = torch.cat((X, parmeters), 1)
        out1 = self.fc1(combination)
        out2 = self.fc2(combination)
        out3 = self.fc3(combination)
        out4 = self.fc4(combination)
        return (out1, out2, out3, out4)

    def _update_metrics(self, phase, preds, targets):
        """更新指定阶段的指标"""
        targets = targets.t()  # 转置为(4, batch_size)

        for head in range(4):
            # 获取当前头的预测和目标
            head_preds = preds[head]
            head_targets = targets[head]

            # 更新指标
            self.metrics[f"{phase}_precision_head{head}"].update(head_preds, head_targets)
            self.metrics[f"{phase}_recall_head{head}"].update(head_preds, head_targets)
            self.metrics[f"{phase}_f1_head{head}"].update(head_preds, head_targets)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, threshold=0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, train_batch, batch_idx):
        x, parameters, y = train_batch
        if self.retrieve_masks:
            y_hats, masks = self.forward(x, parameters)  # 修改前向传播返回值的解包方式
        else:
            y_hats = self.forward(x,parameters)
        y_hat0, y_hat1, y_hat2, y_hat3 = y_hats
        y = y.t()

        _, preds0 = torch.max(y_hat0, 1)
        loss0 = F.cross_entropy(y_hat0, y[0])

        _, preds1 = torch.max(y_hat1, 1)
        loss1 = F.cross_entropy(y_hat1, y[1])

        _, preds2 = torch.max(y_hat2, 1)
        loss2 = F.cross_entropy(y_hat2, y[2])

        _, preds3 = torch.max(y_hat3, 1)
        loss3 = F.cross_entropy(y_hat3, y[3])

        loss = loss0 + loss1 + loss2 + loss3
        preds = torch.stack((preds0, preds1, preds2, preds3))

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_loss0",
            loss0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_loss1",
            loss1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_loss2",
            loss2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_loss3",
            loss3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )

        self.train_acc(preds, y)
        self.train_acc0(preds0, y[0])
        self.train_acc1(preds1, y[1])
        self.train_acc2(preds2, y[2])
        self.train_acc3(preds3, y[3])

        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_acc0",
            self.train_acc0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_acc1",
            self.train_acc1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_acc2",
            self.train_acc2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_acc3",
            self.train_acc3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        # 所有头的预测必须全部正确才算样本正确
        correct_samples = ((preds0 == y[0]) &
                           (preds1 == y[1]) &
                           (preds2 == y[2]) &
                           (preds3 == y[3])).sum().item()
        batch_acc = correct_samples / x.size(0)  # x.size(0)是当前batch大小
        self.log("train_lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        self.log("train_batch_acc", batch_acc, prog_bar=True)
        self._update_metrics("train", preds, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, parameters, y = val_batch
        if self.retrieve_masks:
            y_hats, masks = self.forward(x,parameters)  # 修改前向传播返回值的解包方式
        else:
            y_hats = self.forward(x,parameters)
        y_hat0, y_hat1, y_hat2, y_hat3 = y_hats
        y = y.t()

        _, preds0 = torch.max(y_hat0, 1)
        loss0 = F.cross_entropy(y_hat0, y[0])

        _, preds1 = torch.max(y_hat1, 1)
        loss1 = F.cross_entropy(y_hat1, y[1])

        _, preds2 = torch.max(y_hat2, 1)
        loss2 = F.cross_entropy(y_hat2, y[2])

        _, preds3 = torch.max(y_hat3, 1)
        loss3 = F.cross_entropy(y_hat3, y[3])

        loss = loss0 + loss1 + loss2 + loss3
        preds = torch.stack((preds0, preds1, preds2, preds3))

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_loss0",
            loss0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_loss1",
            loss1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_loss2",
            loss2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_loss3",
            loss3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )

        self.val_acc(preds, y)
        self.val_acc0(preds0, y[0])
        self.val_acc1(preds1, y[1])
        self.val_acc2(preds2, y[2])
        self.val_acc3(preds3, y[3])

        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_acc0",
            self.val_acc0,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_acc1",
            self.val_acc1,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_acc2",
            self.val_acc2,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "val_acc3",
            self.val_acc3,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self._update_metrics("val", preds, y)
        return loss

    def _log_metrics(self, phase):
        """记录指定阶段的指标"""
        for head in range(4):
            # 记录每个头的指标
            precision = self.metrics[f"{phase}_precision_head{head}"].compute()
            recall = self.metrics[f"{phase}_recall_head{head}"].compute()
            f1 = self.metrics[f"{phase}_f1_head{head}"].compute()

            self.log(f"{phase}_precision_head{head}", precision, sync_dist=self.sync_dist)
            self.log(f"{phase}_recall_head{head}", recall, sync_dist=self.sync_dist)
            self.log(f"{phase}_f1_head{head}", f1, sync_dist=self.sync_dist)

            # 重置指标
            self.metrics[f"{phase}_precision_head{head}"].reset()
            self.metrics[f"{phase}_recall_head{head}"].reset()
            self.metrics[f"{phase}_f1_head{head}"].reset()
    def test_step(self, test_batch, batch_idx):
        x, parameters, y = test_batch
        if self.retrieve_masks:
            y_hats, masks = self.forward(x,parameters)  # 修改前向传播返回值的解包方式
        else:
            y_hats = self.forward(x,parameters)
        y_hat0, y_hat1, y_hat2, y_hat3 = y_hats
        y = y.t()

        _, preds0 = torch.max(y_hat0, 1)
        loss0 = F.cross_entropy(y_hat0, y[0])

        _, preds1 = torch.max(y_hat1, 1)
        loss1 = F.cross_entropy(y_hat1, y[1])

        _, preds2 = torch.max(y_hat2, 1)
        loss2 = F.cross_entropy(y_hat2, y[2])

        _, preds3 = torch.max(y_hat3, 1)
        loss3 = F.cross_entropy(y_hat3, y[3])

        loss = loss0 + loss1 + loss2 + loss3

        self.log("test_loss0", loss0)
        self.log("test_loss1", loss1)
        self.log("test_loss2", loss2)
        self.log("test_loss3", loss3)

        preds = torch.stack((preds0, preds1, preds2, preds3))
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.test_acc(preds, y)
        self.log(
            "test_acc",
            self.test_acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )

        self._update_metrics("test", preds, y)
        return {"loss": loss, "preds": preds, "targets": y}

    def test_epoch_end(self, outputs):
        preds = [output["preds"] for output in outputs]
        targets = [output["targets"] for output in outputs]

        preds = torch.cat(preds, dim=1)
        targets = torch.cat(targets, dim=1)

        os.makedirs("test/", exist_ok=True)
        if self.test_overwrite_filename:
            torch.save(preds, "test/preds_test.pt")
            torch.save(targets, "test/targets_test.pt")
        else:
            date_string = datetime.now().strftime("%H-%M_%d-%m-%y")
            torch.save(preds, "test/preds_{}.pt".format(date_string))
            torch.save(targets, "test/targets_{}.pt".format(date_string))
        self._log_metrics("test")
        # 生成详细分类报告
        all_preds = torch.cat([x["preds"] for x in self.test_outputs], dim=1)
        all_targets = torch.cat([x["targets"] for x in self.test_outputs], dim=1)

        # 保存分类报告到文件
        os.makedirs("reports", exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_path = f"reports/classification_report_{date_str}.txt"

        with open(report_path, "w") as f:
            for head in range(4):
                f.write(f"\n{'=' * 40}\nHead {head} Report\n{'=' * 40}\n")
                head_preds = all_preds[head].cpu().numpy()
                head_targets = all_targets[head].cpu().numpy()
                report = classification_report(
                    head_targets,
                    head_preds,
                    target_names=[f"Class_{i}" for i in range(self.num_classes)],
                    digits=4
                )
                f.write(report)

    def on_train_epoch_end(self,*args, **kwarg):
        # 收集训练指标
        metrics = [
            'train_acc', 'train_acc0', 'train_acc1', 'train_acc2', 'train_acc3',
            'train_loss', 'train_loss0', 'train_loss1', 'train_loss2', 'train_loss3'  # 新增损失指标
        ]
        for metric in metrics:
            key = metric.split('_')[-1]
            value = self.trainer.callback_metrics.get(metric)
            if value is not None:
                self.training_history[key].append(value.item())
            else:
                self.training_history[key].append(float('nan'))
        self._log_metrics("train")
    def on_validation_epoch_end(self,*args, **kwargs):
        # 收集验证指标
        metrics = [
            'val_acc', 'val_acc0', 'val_acc1', 'val_acc2', 'val_acc3',
            'val_loss', 'val_loss0', 'val_loss1', 'val_loss2', 'val_loss3'  # 新增损失指标
        ]
        for metric in metrics:
            key = metric.split('_')[-1]
            value = self.trainer.callback_metrics.get(metric)
            if value is not None:
                self.validation_history[key].append(value.item())
            else:
                self.validation_history[key].append(float('nan'))


    def save_training_history(self, filename="training_history.csv"):
        import matplotlib.pyplot as plt
        from pathlib import Path
        # 创建保存目录
        save_dir = Path("training_plots")
        save_dir.mkdir(exist_ok=True)
        # 确保所有数组等长
        min_length = min(
            len(self.training_history['acc']),
            len(self.validation_history['acc']))
        # 创建DataFrame并保存
        data = {
            'epoch': list(range(1, min_length + 1)),
            'train_acc': self.training_history['acc'][:min_length],
            'val_acc': self.validation_history['acc'][:min_length],
            'train_acc0': self.training_history['acc0'][:min_length],
            'train_acc1': self.training_history['acc1'][:min_length],
            'train_acc2': self.training_history['acc2'][:min_length],
            'train_acc3': self.training_history['acc3'][:min_length],
            'val_acc0': self.validation_history['acc0'][:min_length],
            'val_acc1': self.validation_history['acc1'][:min_length],
            'val_acc2': self.validation_history['acc2'][:min_length],
            'val_acc3': self.validation_history['acc3'][:min_length],
            # 损失指标（新增部分）
            'train_loss': self.training_history['loss'][:min_length],
            'val_loss': self.validation_history['loss'][:min_length],
            'train_loss0': self.training_history['loss0'][:min_length],
            'train_loss1': self.training_history['loss1'][:min_length],
            'train_loss2': self.training_history['loss2'][:min_length],
            'train_loss3': self.training_history['loss3'][:min_length],
            'val_loss0': self.validation_history['loss0'][:min_length],
            'val_loss1': self.validation_history['loss1'][:min_length],
            'val_loss2': self.validation_history['loss2'][:min_length],
            'val_loss3': self.validation_history['loss3'][:min_length],
        }
        # 创建DataFrame
        df = pd.DataFrame(data)
        csv_path = save_dir / f"{filename}1.csv"
        df.to_csv(csv_path, index=False)
        print(f"Training history saved to {csv_path}")
        # 绘制总准确率曲线
        plt.figure(figsize=(12, 6))
        plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
        plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy')
        plt.title('Overall Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plot_path = save_dir / f"{filename}_overall.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Overall accuracy plot saved to {plot_path}")

        # 绘制各分类头准确率曲线
        plt.figure(figsize=(15, 10))

        # 训练准确率
        plt.subplot(2, 1, 1)
        for i in range(4):
            if i == 0:
                plt.plot(df['epoch'], df[f'train_acc{i}'], label=f'Retraction distance')
            elif i == 1:
                plt.plot(df['epoch'], df[f'train_acc{i}'], label=f'Retraction speed')
            elif i == 2:
                plt.plot(df['epoch'], df[f'train_acc{i}'], label=f'Hotend temperature')
            elif i == 3:
                plt.plot(df['epoch'], df[f'train_acc{i}'], label=f'Print speed')
        plt.title('Training Accuracy per Head')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # 验证准确率
        plt.subplot(2, 1, 2)
        for i in range(4):
            if i == 0:
                plt.plot(df['epoch'], df[f'train_acc{i}'], label=f'Retraction distance')
            elif i == 1:
                plt.plot(df['epoch'], df[f'train_acc{i}'], label=f'Retraction speed')
            elif i == 2:
                plt.plot(df['epoch'], df[f'train_acc{i}'], label=f'Hotend temperature')
            elif i == 3:
                plt.plot(df['epoch'], df[f'train_acc{i}'], label=f'Print speed')
        plt.title('Validation Accuracy per Head')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = save_dir / f"{filename}_per_head.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Per-head accuracy plot saved to {plot_path}")
        self._plot_loss_curves(df, save_dir, filename)


def _plot_loss_curves(self, df, save_dir, filename):
    import matplotlib.pyplot as plt

    # 绘制总损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.title('Overall Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = save_dir / f"{filename}_loss_overall.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Overall loss plot saved to {plot_path}")

    # 绘制各分类头损失曲线
    plt.figure(figsize=(15, 10))

    # 训练损失
    plt.subplot(2, 1, 1)
    for i in range(4):
        if i == 0:
            plt.plot(df['epoch'], df[f'train_loss{i}'], label='Retraction distance')
        elif i == 1:
            plt.plot(df['epoch'], df[f'train_loss{i}'], label='Retraction speed')
        elif i == 2:
            plt.plot(df['epoch'], df[f'train_loss{i}'], label='Hotend temperature')
        elif i == 3:
            plt.plot(df['epoch'], df[f'train_loss{i}'], label='Print speed')
    plt.title('Training Loss per Head')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 验证损失
    plt.subplot(2, 1, 2)
    for i in range(4):
        if i == 0:
            plt.plot(df['epoch'], df[f'val_loss{i}'], label='Retraction distance')
        elif i == 1:
            plt.plot(df['epoch'], df[f'val_loss{i}'], label='Retraction speed')
        elif i == 2:
            plt.plot(df['epoch'], df[f'val_loss{i}'], label='Hotend temperature')
        elif i == 3:
            plt.plot(df['epoch'], df[f'val_loss{i}'], label='Print speed')
    plt.title('Validation Loss per Head')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = save_dir / f"{filename}_loss_per_head.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Per-head loss plot saved to {plot_path}")
