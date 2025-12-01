import os
from typing import get_args
from pytorch_lightning import seed_everything

from geoseg.datasets.vaihingen_dataset import val_aug
from lr_search_train_supervision import Supervision_Train
from tools.cfg import py2cfg

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)
    test_weights_name = os.listdir(config.weights_path)[-1]
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, test_weights_name),
        config=config,
    )
    print(model)


def gen_cam(model_name, dataset_name, start_end=[0, 50], target_layers_map={}):
    start_idx = start_end[0]
    stop_idx = start_end[1]

    image_path_dir = f"data/{dataset_name}/test/images_1024"
    output_path = f"output/CAM/{dataset_name}/{model_name}"
    mask_path_dir = f"data/{dataset_name}/test/masks_1024"
    config_path = f"./config/{dataset_name}/{model_name}.py"
    config = py2cfg(config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    ).cuda()
    model.eval()

    for img_idx, image_name in enumerate(os.listdir(image_path_dir)):
        if img_idx < start_idx:
            continue
        image_path = os.path.join(image_path_dir, image_name)
        mask_path = os.path.join(mask_path_dir, image_name.replace(".tif", ".png"))
        image = Image.open(image_path)
        hstack_image = image
        mask = Image.open(mask_path)
        image, mask = val_aug(img=image, mask=mask)
        img = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img = img.unsqueeze(0).cuda()
        pred = model(img)
        for idx, cls in enumerate(config.classes):
            pred_cls = pred.squeeze(0).argmax(dim=0).detach().cpu().numpy()
            if not os.path.exists(os.path.join(output_path, cls)):
                os.makedirs(os.path.join(output_path, cls))
            mask_uint8 = 255 * np.uint8(pred_cls == idx)
            mask_float = np.float32(pred_cls == idx)
            both_images = np.hstack(
                (hstack_image, np.repeat(mask_uint8[:, :, None], 3, axis=-1))
            )
            both_image_save_path = os.path.join(output_path, cls, "both")
            if not os.path.exists(both_image_save_path):
                os.makedirs(both_image_save_path)
            Image.fromarray(both_images).save(
                os.path.join(
                    both_image_save_path, f"{image_name.replace('.tif', '.png')}"
                )
            )
            target_layers_map = {
                "decoder.blacks[0]": model.net.decoder.blacks[0],
                "decoder.blacks[1]": model.net.decoder.blacks[1],
                "decoder.blacks[2]": model.net.decoder.blacks[2],
                "decoder.segmentation_head": model.net.decoder.segmentation_head,
                # "decoder.b4": model.net.decoder.b4,
                # "decoder.b3": model.net.decoder.b3,
                # "decoder.b2": model.net.decoder.b2,
                # "decoder.segmentation_head": model.net.decoder.segmentation_head,
                "backbone.blocks[1]": model.net.backbone.blocks[1],
                "backbone.blocks[2]": model.net.backbone.blocks[3],
                "backbone.blocks[3]": model.net.backbone.blocks[3],
                "backbone.blocks[4]": model.net.backbone.blocks[4],
            }
            targets = [SemanticSegmentationTarget(idx, mask_float)]
            for target_layers_name, target_layer in target_layers_map.items():
                with GradCAM(
                    model=model,
                    target_layers=[target_layer],
                ) as cam:
                    grayscale_cam = cam(input_tensor=img, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]
                    visualization = show_cam_on_image(
                        np.float32(hstack_image) / 255, grayscale_cam, use_rgb=True
                    )
                    targets_save_path = os.path.join(
                        output_path, cls, target_layers_name
                    )
                    if not os.path.exists(targets_save_path):
                        os.makedirs(targets_save_path)
                    Image.fromarray(visualization).save(
                        os.path.join(
                            targets_save_path, f"{image_name.replace('.tif', '.png')}"
                        )
                    )
                print(f"{idx}: {image_name}-{cls}-{target_layers_name} done")
        if img_idx >= stop_idx:
            break


if __name__ == "__main__":
    # model_name = "atmbnetv8"
    # model_name = "mifnet"
    # model_name = "atmbnetv8_no_ssdindt"
    dataset_name = "vaihingen"
    model_names = [
        # "atmbnetv8_no_ssdindt",
        "atmbnetv8_no_edge",
        "atmbnetv8_no_erhindt",
    ]
    for model_name in model_names:
        gen_cam(model_name, dataset_name, start_end=[0, 50])
