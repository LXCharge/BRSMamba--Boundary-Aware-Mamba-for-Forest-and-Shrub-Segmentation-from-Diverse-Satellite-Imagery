from argparse import Namespace
import datetime
import importlib
import os
import pathlib
import pydoc
import sys
from importlib import import_module
from pathlib import Path
from typing import Union

from addict import Dict
import torch
from torch.utils.data import DataLoader

from tools.utils import Lookahead, process_model_params


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        else:
            return value
        raise ex


def py2dict(file_path: Union[str, Path]) -> dict:
    """Convert python file to dictionary.
    The main use - config parser.
    file:
    ```
    a = 1
    b = 3
    c = range(10)
    ```
    will be converted to
    {'a':1,
     'b':3,
     'c': range(10)
    }
    Args:
        file_path: path to the original python file.
    Returns: {key: value}, where key - all variables defined in the file and value is their value.
    """
    file_path = Path(file_path).absolute()

    if file_path.suffix != ".py":
        raise TypeError(
            f"Only Py file can be parsed, but got {file_path.name} instead."
        )

    if not file_path.exists():
        raise FileExistsError(f"There is no file at the path {file_path}")

    module_name = file_path.stem

    if "." in module_name:
        raise ValueError("Dots are not allowed in config file path.")

    config_dir = str(file_path.parent)

    sys.path.insert(0, config_dir)

    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value for name, value in mod.__dict__.items() if not name.startswith("__")
    }

    return cfg_dict


def py2cfg(file_path: Union[str, Path]) -> ConfigDict:
    cfg_dict = py2dict(file_path)

    return ConfigDict(cfg_dict)


def arg2cfg(args: Namespace) -> ConfigDict:
    cfg_dict = py2dict(args.config_path)

    if args.model_name is not None and cfg_dict["model_name"] != args.model_name:
        cfg_dict["model_name"] = args.model_name
        cfg_dict["weights_name"] = (
            f"{cfg_dict['model_name']}-{cfg_dict['backbone_name']}-e{cfg_dict['max_epoch']}"
        )
        cfg_dict["weights_path"] = (
            f"model_weights/{cfg_dict['model_name']}/{cfg_dict['dataset_name']}/{cfg_dict['weights_name']}"
        )
        cfg_dict["test_weights_path"] = f"{cfg_dict['weights_path']}-last"
        current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        cfg_dict["log_name"] = (
            f"{cfg_dict['model_name']}/{cfg_dict['dataset_name']}/{cfg_dict['weights_name']}/{current_date}"
        )
    if hasattr(args, "in_dt_blk") and args.in_dt_blk is not None:
        cfg_dict["in_dt_blk"] = args.in_dt_blk
    if args.alpha is not None:
        cfg_dict["alpha"] = args.alpha
        cfg_dict["loss"].set_alpha(cfg_dict["alpha"])
    """ if args.dataset_name is not None and cfg_dict["dataset_name"] != args.dataset_name:
        cfg_dict["dataset_name"] = args.dataset_name
        cfg_dict["weights_path"] = (
            f"model_weights/{cfg_dict['model_name']}/{args.dataset_name}/{cfg_dict['weights_name']}"
        )
        cfg_dict["test_weights_path"] = f"{cfg_dict['weights_path']}-last"
        cfg_dict["log_name"] = (
            f"{cfg_dict['model_name']}/{args.dataset_name}/{cfg_dict['weights_name']}/{cfg_dict['current_date']}"
        )
        if args.dataset_name == "loveda":
            dataset = importlib.import_module("geoseg.datasets.loveda_dataset")
            train_dataset = dataset.LoveDATrainDataset(
                transform=dataset.train_aug, data_root="data/LoveDA/Train"
            )
            val_dataset = dataset.loveda_val_dataset
        elif args.dataset_name == "vaihingen":
            dataset = importlib.import_module("geoseg.datasets.vaihingen_dataset")
            train_dataset = dataset.VaihingenDataset(
                transform=dataset.train_aug,
                data_root="data/vaihingen/train",
                mode="train",
                mosaic_ratio=0.25,
            )
            val_dataset = dataset.VaihingenDataset(transform=dataset.val_aug)
        elif args.dataset_name == "potsdam_t":
            # dataset = importlib.import_module("geoseg.datasets.potsdam_dataset")
            train_dataset = dataset.PotsdamDataset(
                transform=dataset.train_aug,
                data_root="data/potsdam_t/train",
                mode="train",
                mosaic_ratio=0.25,
            )
            val_dataset = dataset.PotsdamDataset(
                transform=dataset.val_aug, data_root="data/potsdam_t/test"
            )
        cfg_dict["classes"] = dataset.CLASSES
        cfg_dict["num_classes"] = len(dataset.CLASSES)
        cfg_dict["ignore_index"] = len(dataset.CLASSES)
        cfg_dict["train_loader"] = DataLoader(
            dataset=train_dataset,
            batch_size=cfg_dict["train_batch_size"],
            num_workers=cfg_dict["train_batch_size"] // 2,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        cfg_dict["train_dataset"] = train_dataset
        cfg_dict["val_loader"] = DataLoader(
            dataset=val_dataset,
            batch_size=cfg_dict["val_batch_size"],
            num_workers=cfg_dict["val_batch_size"] // 2,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        cfg_dict["val_dataset"] = val_dataset
    if args.backbone_weight_decay is not None:
        cfg_dict["backbone_weight_decay"] = args.backbone_weight_decay
    if args.weight_decay is not None:
        cfg_dict["weight_decay"] = args.weight_decay
    if args.backbone_lr is not None:
        cfg_dict["backbone_lr"] = args.backbone_lr
        cfg_dict["layerwise_params"] = {
            "backbone.*": dict(
                lr=args.backbone_lr, weight_decay=cfg_dict["backbone_weight_decay"]
            )
        }
    if args.lr is not None:
        assert args.lr > 0, "lr should be positive"
        cfg_dict["lr"] = args.lr """
    if hasattr(args, "head") and args.head is not None:
        cfg_dict["head"] = args.head
    if hasattr(args, "window_size") and args.window_size is not None:
        cfg_dict["window_size"] = args.window_size

    if cfg_dict["net"] is not None:
        cfg_dict["net_params"] = process_model_params(
            cfg_dict["net"], layerwise_params=cfg_dict["layerwise_params"]
        )
        cfg_dict["base_optimizer"] = torch.optim.AdamW(
            cfg_dict["net_params"],
            lr=cfg_dict["lr"],
            weight_decay=cfg_dict["weight_decay"],
        )
        cfg_dict["optimizer"] = Lookahead(cfg_dict["base_optimizer"])
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=15, T_mult=2
        # )
        cfg_dict["lr_scheduler"] = torch.optim.lr_scheduler.ExponentialLR(
            cfg_dict["optimizer"], gamma=0.98, verbose=True
        )
    for key, value in args._get_kwargs():
        if key in cfg_dict and value is not None:
            cfg_dict[key] = value
    return ConfigDict(cfg_dict)


# def arg2cfg(args: dict) -> ConfigDict:
#     cfg_dict = {
#         name: value for name, value in args.items() if not name.startswith("__")
#     }
#     dataset = importlib.import_module("geoseg.datasets." + args.dataset_name)
#     cfg_dict.classes = dataset.CLASSES
#     cfg_dict.num_classes = len(dataset.CLASSES)
#     cfg_dict.ignore_index = len(dataset.CLASSES)
#     cfg_dict.weights_name = (
#         f"{cfg_dict.model_name}-{cfg_dict.backbone_name}-e{cfg_dict.max_epoch}"
#     )
#     cfg_dict.weights_path = f"model_weights/{cfg_dict.model_name}/{cfg_dict.dataset_name}/{cfg_dict.weights_name}"
#     cfg_dict.test_weights_name = f"{cfg_dict.weights_name}-last"
#     current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
#     cfg_dict.log_name = f"{cfg_dict.model_name}/{cfg_dict.dataset_name}/{cfg_dict.weights_name}/{current_date}"
#     print(f"log_name: {cfg_dict.log_name}")
#     monitor = "val_F1"
#     cfg_dict.monitor_mode = "max"
#     cfg_dict.save_top_k = 1
#     cfg_dict.save_last = "link"
#     cfg_dict.check_val_every_n_epoch = 1
#     cfg_dict.pretrained_ckpt_path = None  # the path for the pretrained model weight
#     # cfg_dict.gpus = "auto"  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
#     cfg_dict.resume_ckpt_path = (
#         None  # whether continue training with the checkpoint, default None
#     )


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    return pydoc.locate(object_type)(**kwargs)
