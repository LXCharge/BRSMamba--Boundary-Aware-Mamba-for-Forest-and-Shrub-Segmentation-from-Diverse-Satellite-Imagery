import datetime
from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader
from geoseg.datasets.vaihingen_dataset import *
from geoseg.losses import *
from geoseg.models.BRSMamba import BRSMamba
from tools.utils import Lookahead
from tools.utils import process_model_params

model_name = "atmbnetv8"
# training hparam
max_epoch = 200
ignore_index = len(CLASSES)
train_batch_size = 12
val_batch_size = train_batch_size
lr = 1e-2
backbone_lr = 1e-3
weight_decay = 0.01
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

dataset_name = "vaihingen"
# backbone_name = "resnet50"
# backbone_name = "mamba_vision_t_1k"
backbone_name = "efficientnet_b3"
# backbone_name = "resnext50_32x4d"
# backbone_name = "resnest26d"
# backbone_name = "efficientvit_b3"
weights_name = f"{model_name}-{backbone_name}-e{max_epoch}"
# weights_path = "model_weights/potsdam/{}".format(weights_name)
weights_path = f"model_weights/{model_name}/{dataset_name}/{weights_name}"
test_weights_name = f"{weights_name}-last"
current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
log_name = f"{model_name}/{dataset_name}/{weights_name}/{current_date}"
print(f"log_name: {log_name}")
monitor = "val_F1"
monitor_mode = "max"
save_top_k = 1
save_last = "link"
check_val_every_n_epoch = 1
pretrained_ckpt_path = None  # the path for the pretrained model weight
gpus = "auto"  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
num_heads = 4
windown_size = 8
net = BRSMamba(
    num_classes=num_classes,
    backbone_name=backbone_name,
    in_dt=True,
    window_size=[windown_size] * 4,
    num_heads=[num_heads] * 4,
    las_decode_channels=32,
)


# define the loss
alpha = 0.4
loss = AtMbLoss(ignore_index=ignore_index, alpha=alpha)
use_aux_loss = True

# define the dataloader

train_dataset = VaihingenDataset(
    data_root="data/vaihingen/train",
    mode="train",
    mosaic_ratio=0.25,
    transform=train_aug,
)

val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root="data/vaihingen/test", transform=val_aug)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=train_batch_size // 2,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=val_batch_size // 2,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

# define the optimizer
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2
)
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
#     optimizer, gamma=0.98, verbose=True
# )
