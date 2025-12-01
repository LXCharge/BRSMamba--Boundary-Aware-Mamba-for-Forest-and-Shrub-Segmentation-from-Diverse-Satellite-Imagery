import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from geoseg.datasets.vaihingen_dataset import VaihingenDataset, val_aug
from geoseg.models.BRSMamba import LAS, GaussianFilter, Laplacian, Sobel
from torchvision.utils import save_image

val_dataset = VaihingenDataset(data_root="data/vaihingen/test/", transform=val_aug)
save_dir = "./output/broundry"
stop_idx = 30


class Sobel(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_x = self.create_sobel_kernel_x().repeat(1, self.in_channels, 1, 1)
        self.kernel_y = self.create_sobel_kernel_y().repeat(1, self.in_channels, 1, 1)

    def create_sobel_kernel_x(self):
        return (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .float()
            .reshape(1, 1, 3, 3)
        )

    def create_sobel_kernel_y(self):
        return (
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            .float()
            .reshape(1, 1, 3, 3)
        )

    def forward(self, x):
        x_x = F.conv2d(x, self.kernel_x.to(x.device), padding=1, groups=1)
        x_y = F.conv2d(x, self.kernel_y.to(x.device), padding=1, groups=1)
        grad_mag = torch.sqrt(torch.pow(x_x, 2) + torch.pow(x_y, 2))
        grad_dir = torch.atan2(x_y, x_x)
        return grad_mag, grad_dir


class LoG(nn.Module):
    def __init__(self, gs_kernel_size=7, **kwargs):
        super().__init__(**kwargs)

        self.gray = transforms.Grayscale()
        self.gaussian = GaussianFilter(1, kernel_size=gs_kernel_size)
        self.laplacian = Laplacian(1, 1)
        # self.gaussianLaplacian = nn.Sequential(
        #     transforms.Grayscale()
        #     GaussianFilter(1, kernel_size=kernel_size[0]),
        #     Laplacian(1, 1, kernel_size=kernel_size[1]),
        # )

    def forward(self, x):
        x = self.gray(x)
        x = self.gaussian(x)
        x = self.laplacian(x)
        return x


def test_cv_log():
    if not os.path.exists(os.path.join(save_dir, "cv_lap")):
        os.makedirs(os.path.join(save_dir, "cv_lap"))
    for i in range(len(val_dataset)):
        item = val_dataset.__getitem__(i)
        img = item.get("img").numpy()
        print(img)
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"cv_lap/{item.get('img_id')}_ori.png"), img)
        print(img.shape)
        cv2.imwrite(os.path.join(save_dir, f"cv_lap/{item.get('img_id')}.png"), img)
        edges = cv2.Laplacian(img, cv2.CV_64F)
        edges = cv2.convertScaleAbs(edges)
        cv2.imwrite(os.path.join(save_dir, f"cv_lap/{item.get('img_id')}.png"), edges)
        if i == stop_idx:
            break


def test_log():
    with torch.no_grad():
        gs_kernel_size = 12
        log = LoG(gs_kernel_size).cuda()
        log.eval()
        log_save_dir = os.path.join(save_dir, f"log_l_k{gs_kernel_size}")
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        for i in range(len(val_dataset)):
            item = val_dataset.__getitem__(i)
            img = item.get("img").cuda()
            img = img.unsqueeze(0)
            pred = log(img)
            sum_pred = torch.sum(pred, dim=1, keepdim=False).squeeze(0)
            sum_pred = sum_pred.cpu().numpy() * 255
            # sum_pred = cv2.normalize(sum_pred, None, 0, 255, cv2.NORM_MINMAX)
            # sum_pred = pred.squeeze(0)
            cv2.imwrite(
                os.path.join(log_save_dir, f"{item.get('img_id')}.png"),
                cv2.convertScaleAbs(sum_pred),
            )
            if i == stop_idx:
                break
            if i == 0:
                time_start = time.time()
        time_end = time.time()
        print(f"log time: {time_end - time_start}")
        print(f"log avg time: {(time_end - time_start) / i}")


def test_las():
    with torch.no_grad():
        las = LAS(in_channels=3, out_channels=32).cuda()
        las.eval()
        if not os.path.exists(os.path.join(save_dir, "log")):
            os.makedirs(os.path.join(save_dir, "log"))
        for i in range(len(val_dataset)):
            item = val_dataset.__getitem__(i)
            img = item.get("img").cuda()
            img = img.unsqueeze(0)
            pred = las(img)
            sum_pred = torch.sum(pred, dim=1, keepdim=False).squeeze(0)
            sum_pred = sum_pred.cpu().numpy()
            sum_pred = cv2.normalize(sum_pred, None, 0, 255, cv2.NORM_MINMAX)
            # sum_pred = pred.squeeze(0)
            cv2.imwrite(
                os.path.join(save_dir, f"log/{item.get('img_id')}.png"),
                cv2.convertScaleAbs(sum_pred),
            )
            if i == stop_idx:
                break


def test_sobal():
    with torch.no_grad():
        _sobel = Sobel(in_channels=1, out_channels=1).cuda()
        _sobel.eval()
        if not os.path.exists(os.path.join(save_dir, "sobel")):
            os.makedirs(os.path.join(save_dir, "sobel"))
        for i in range(len(val_dataset)):
            item = val_dataset.__getitem__(i)
            img = item.get("img").cuda()
            img = torchvision.transforms.Grayscale()(img)
            img = img.unsqueeze(0)
            grad_mag, grad_dir = _sobel(img)
            grad_mag = grad_mag.squeeze(0).cpu().numpy()
            grad_dir = grad_dir.squeeze(0).cpu().numpy()
            print(grad_mag.shape, grad_dir.shape)
            grad_mag = cv2.normalize(grad_mag[0, :, :], None, 0, 255, cv2.NORM_MINMAX)
            grad_dir = cv2.normalize(grad_dir[0, :, :], None, 0, 255, cv2.NORM_MINMAX)
            # sum_pred = pred.squeeze(0)
            cv2.imwrite(
                os.path.join(save_dir, f"sobel/{item.get('img_id')}.png"),
                cv2.convertScaleAbs(grad_mag),
            )
            # cv2.imwrite(
            #     os.path.join(save_dir, f"sobel/grad_dir/{item.get('img_id')}.png"),
            #     cv2.convertScaleAbs(grad_dir),
            # )
            if i == stop_idx:
                break


class Canny(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gray = transforms.Grayscale()
        self.sobel = Sobel(in_channels=1, out_channels=1)

    def non_maximum_suppression(self, gradient, angle):
        h, w = gradient.shape[-2:]
        suppressed = torch.zeros_like(gradient)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                a = angle[0, 0, i, j].item() / np.pi * 180
                if (
                    (a < -22.5 or a >= 157.5)
                    and gradient[0, 0, i, j] >= gradient[0, 0, i, j - 1]
                    and gradient[0, 0, i, j] >= gradient[0, 0, i, j + 1]
                ):
                    suppressed[0, 0, i, j] = gradient[0, 0, i, j]
                elif (
                    (a >= -22.5 and a < 22.5)
                    and gradient[0, 0, i, j] >= gradient[0, 0, i - 1, j]
                    and gradient[0, 0, i, j] >= gradient[0, 0, i + 1, j]
                ):
                    suppressed[0, 0, i, j] = gradient[0, 0, i, j]
                elif (
                    (a >= 22.5 and a < 67.5)
                    and gradient[0, 0, i, j] >= gradient[0, 0, i - 1, j - 1]
                    and gradient[0, 0, i, j] >= gradient[0, 0, i + 1, j + 1]
                ):
                    suppressed[0, 0, i, j] = gradient[0, 0, i, j]
                elif (
                    (a >= 67.5 and a < 112.5)
                    and gradient[0, 0, i, j] >= gradient[0, 0, i - 1, j]
                    and gradient[0, 0, i, j] >= gradient[0, 0, i + 1, j]
                ):
                    suppressed[0, 0, i, j] = gradient[0, 0, i, j]
                elif (
                    (a >= 112.5 and a < 157.5)
                    and gradient[0, 0, i, j] >= gradient[0, 0, i - 1, j + 1]
                    and gradient[0, 0, i, j] >= gradient[0, 0, i + 1, j - 1]
                ):
                    suppressed[0, 0, i, j] = gradient[0, 0, i, j]
        return suppressed

    def double_threshold(self, suppressed, low_threshold=20, high_threshold=50):
        strong = (suppressed >= high_threshold).float()
        weak = (suppressed >= low_threshold).float() - strong
        return strong, weak

    def edge_tracking(self, strong, weak):
        h, w = strong.shape[-2:]
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if (
                    weak[0, 0, i, j]
                    and strong[0, 0, i - 1 : i + 2, j - 1 : j + 2].max() > 0
                ):
                    strong[0, 0, i, j] = 1
                    weak[0, 0, i, j] = 0
        return strong

    def forward(self, x):
        x = self.gray(x)
        grad_mag, grad_dir = self.sobel(x)
        suppressed = self.non_maximum_suppression(grad_mag, grad_dir)
        strong, weak = self.double_threshold(suppressed)
        edges = self.edge_tracking(strong, weak)
        return edges


def test_canny():
    with torch.no_grad():
        canny = Canny().cuda()
        canny.eval()
        if not os.path.exists(os.path.join(save_dir, "canny")):
            os.makedirs(os.path.join(save_dir, "canny"))
        for i in range(len(val_dataset)):
            item = val_dataset.__getitem__(i)
            img = item.get("img").cuda()
            img = img.unsqueeze(0)
            pred = canny(img)
            pred = torch.sum(pred, dim=1, keepdim=False).squeeze(0).cpu().numpy() * 255
            cv2.imwrite(
                os.path.join(save_dir, f"canny/{item.get('img_id')}.png"),
                cv2.convertScaleAbs(pred),
            )
            if i == stop_idx:
                break
            if i == 0:
                time_start = time.time()
    time_end = time.time()
    print(f"canny time: {time_end - time_start}")
    print(f"canny avg time: {(time_end - time_start) / i}")


def test_cv_canny():
    cv_canny_save_dir = os.path.join(save_dir, "cv_canny")
    if not os.path.exists(cv_canny_save_dir):
        os.makedirs(cv_canny_save_dir)
    for i in range(len(val_dataset)):
        item = val_dataset.__getitem__(i)
        img = item.get("img").numpy() * 255
        img = img.astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        gauss_img = cv2.GaussianBlur(img, (3, 3), 0)
        pred = cv2.Canny(gauss_img, 800, 400)
        cv2.imwrite(
            os.path.join(cv_canny_save_dir, f"{item.get('img_id')}.png"),
            pred,
        )
        if i == stop_idx:
            break
        if i == 0:
            time_start = time.time()
    time_end = time.time()
    print(f"cv_canny time: {time_end - time_start}")
    print(f"cv_canny avg time: {(time_end - time_start) / i}")


def test():
    # test_las()
    # test_sobal()
    test_log()
    # test_cv_log()
    # test_canny()
    # test_cv_canny()


if __name__ == "__main__":
    test()
