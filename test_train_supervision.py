import re
import comet_ml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tools.cfg import arg2cfg, py2cfg
import os
import torch
from torch import nn, multiprocessing
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger, CometLogger, MLFlowLogger
import random
from tabulate import tabulate
from ptflops import get_model_complexity_info

multiprocessing.set_sharing_strategy("file_system")


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-m", "--model_name", type=str, help="Name of the model/log.", required=False)
    arg("-lr", "--lr", type=float, help="Learning rate.", required=False)
    arg("-blr", "--backbone_lr", type=float, help="Base learning rate.", required=False)
    arg("-wd", "--weight_decay", type=float, help="Weight decay.", required=False)
    arg(
        "-bwd",
        "--backbone_weight_decay",
        type=str,
        help="Backbone Weight Decay.",
        required=False,
    )
    arg("-d", "--dataset_name", type=str, help="Name of the dataset.", required=False)
    arg("-aux", "--use_aux_loss", type=bool, help="Use aux loss.", required=False)
    arg(
        "-a",
        "--alpha",
        type=float,
        help="Aux loss weight.",
        required=False,
    )
    arg("-g", "--gpus", type=str, help="Number of gpus.", required=False)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net.cuda()

        self.loss = config.loss
        self.training_step_loss = []
        self.val_step_loss = []

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch["img"], batch["gt_semantic_seg"]
        self.net.train()
        torch.set_grad_enabled(True)
        prediction = self.net(img)
        loss = self.loss(prediction, mask)
        self.training_step_loss.append(loss)
        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(
                mask[i].cpu().numpy(), pre_mask[i].cpu().numpy()
            )

        return {"loss": loss}

    def on_train_epoch_end(self):
        iou_oa_f1_pre_acc = self.metrics_train.get_all_metrics(needAcc=True)
        pattern = r"vaihingen|potsdam|whubuilding|massbuilding|cropland"
        if re.search(pattern, self.config.dataset_name):
            mIoU = np.nanmean(iou_oa_f1_pre_acc["IoU"][:-1])
            F1 = np.nanmean(iou_oa_f1_pre_acc["F1"][:-1])
            Acc = np.nanmean(iou_oa_f1_pre_acc["Pixel_Accuracy_Class"][:-1])
        else:
            mIoU = np.nanmean(iou_oa_f1_pre_acc["IoU"])
            F1 = np.nanmean(iou_oa_f1_pre_acc["F1"])
            Acc = np.nanmean(iou_oa_f1_pre_acc["Pixel_Accuracy_Class"])
        OA = np.nanmean(iou_oa_f1_pre_acc["OA"])
        iou_per_class = iou_oa_f1_pre_acc["IoU"]
        print("train:")

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        # print(
        #     tabulate(
        #         np.array(list(iou_value.items())),
        #         headers=["TRAIN_CLASS", "IoU"],
        #         tablefmt="fancy_grid",
        #     )
        # )
        self.metrics_train.reset()
        epoch_loss_mean = torch.stack(self.training_step_loss).mean()
        self.training_step_loss.clear()
        log_dict = {
            "T_mIoU": mIoU,
            "T_F1": F1,
            "T_OA": OA,
            "T_Acc": Acc,
            "T_loss": epoch_loss_mean,
        }
        print(
            tabulate(
                [list(log_dict.values())],
                headers=list(log_dict.keys()),
                tablefmt="fancy_grid",
            )
        )
        self.log_dict(log_dict, prog_bar=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch["img"], batch["gt_semantic_seg"]
        self.net.eval()
        torch.set_grad_enabled(False)
        prediction = self.net(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        self.val_step_loss.append(loss_val)
        self.log_dict({"val_loss": loss_val}, prog_bar=False, sync_dist=True)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        iou_oa_f1_pre_acc = self.metrics_val.get_all_metrics(needAcc=True)
        pattern = r"vaihingen|potsdam|whubuilding|massbuilding|cropland"
        if re.search(pattern, self.config.dataset_name):
            mIoU = np.nanmean(iou_oa_f1_pre_acc["IoU"][:-1])
            F1 = np.nanmean(iou_oa_f1_pre_acc["F1"][:-1])
            Acc = np.nanmean(iou_oa_f1_pre_acc["Pixel_Accuracy_Class"][:-1])
        else:
            mIoU = np.nanmean(iou_oa_f1_pre_acc["IoU"])
            F1 = np.nanmean(iou_oa_f1_pre_acc["F1"])
            Acc = np.nanmean(iou_oa_f1_pre_acc["Pixel_Accuracy_Class"])
        OA = np.nanmean(iou_oa_f1_pre_acc["OA"])
        iou_per_class = iou_oa_f1_pre_acc["IoU"]
        print("val:")
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        # print(list(iou_value.values()), type(iou_value.values()))
        print(
            tabulate(
                np.array(list(iou_value.items())),
                headers=["VAL_CLASS", "IoU"],
                tablefmt="fancy_grid",
            )
        )
        self.log_dict(iou_value, prog_bar=False, sync_dist=True)

        self.metrics_val.reset()
        epoch_loss_mean = torch.stack(self.val_step_loss).mean()
        self.val_step_loss.clear()
        log_dict = {
            "val_mIoU": mIoU,
            "val_F1": F1,
            "val_OA": OA,
            "val_Acc": Acc,
            "val_loss": epoch_loss_mean,
        }
        self.log_dict(log_dict, prog_bar=False, sync_dist=True)
        log_dict = {
            "V_mIoU": mIoU,
            "V_F1": F1,
            "V_OA": OA,
            "V_Acc": Acc,
            "V_loss": epoch_loss_mean,
        }
        print(
            tabulate(
                [list(log_dict.values())],
                headers=list(log_dict.keys()),
                tablefmt="fancy_grid",
            )
        )

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    # config = py2cfg(args.config_path)
    config = arg2cfg(args)
    print(config.log_name)
    # quit()
    seed_everything(42)
    torch.set_float32_matmul_precision("high")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        save_last=config.save_last,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name + "-{epoch:02d}-{val_mIoU:.4f}",
        auto_insert_metric_name=True,
        save_weights_only=True,
    )
    early_stop_callback = EarlyStopping(
        monitor=config.monitor,
        mode=config.monitor_mode,
        patience=10,
    )
    loggers = [
        # CSVLogger("lightning_logs", name=f"{config.log_name}/csv"),
        # CometLogger(
        #     api_key="qGr3I8rJ46dWzjRvVIrUtiFlT",
        #     # save_dir=f"lightning_logs/{config.log_name}/comet",
        #     project_name=config.dataset_name,
        #     experiment_name=f"{config.weights_name}-{config.lr}-{config.current_date}",
        # ),
    ]
    config.gmac, config.params = get_model_complexity_info(
        config.net.cuda(), (3, 1024, 1024), as_strings=True, print_per_layer_stat=False
    )
    print(f"Model FLOPs: {config.gmac}, params: {config.params}")
    # loggers[1].log_hyperparams(
    #     {
    #         key: value
    #         for key, value in zip(config.keys(), config.values())
    #         if isinstance(value, (str, int, float, bool))
    #     }
    # )
    model = Supervision_Train(config)
    # model = torch.compile(model)
    # if config.pretrained_ckpt_path:
    #     model = Supervision_Train.load_from_checkpoint(
    #         config.pretrained_ckpt_path, config=config
    #     )

    trainer = pl.Trainer(
        devices=config.gpus,
        max_epochs=config.max_epoch,
        accelerator="auto",
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback, early_stop_callback],
        strategy="auto",
        logger=loggers,
        # log_every_n_steps=50,
    )
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
    main()
