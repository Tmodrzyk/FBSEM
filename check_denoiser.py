# %%
import torch
from deepinv.loss import PSNR
from deepinv.loss.metric import MSE
import deepinv as dinv
from pathlib import Path
from torchvision import transforms, datasets
import sys
from deepinv.loss import SupLoss
import wandb
from trainer import Trainer
import numpy as np
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from models.deeplib import PETdatasetHD, dotstruct, toNumpy, crop
from models.modellib import FBSEMnet_v3, fbsemInference
import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


save_testing_dir = r"./MoDL/testingDatasets/brainweb/2D"

is3d = False
temPath = r"./tmp/"
radialBinCropFactor = 0.5
psf_cm = 0.15
niters = 8
nsubs = 6
training_flname = [save_testing_dir + os.sep, "data-"]
save_dir = r"./MoDL/output/brainweb/2D" + os.sep
num_workers = 0
batch_size = 16
test_size = 0.0
valid_size = 0.1
num_train = 100
pretrained_path = Path("./weights/25-10-13-14:12:28/ckp_best.pth.tar")

# build PET object
PET = BuildGeometry_v4("mmr", radialBinCropFactor)
PET.loadSystemMatrix(temPath, is3d=False)

target_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.CenterCrop((128, 128)),
        transforms.Lambda(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        ),  # → [0,1]
    ]
)

# load dataloaders
train_loader, val_loader, test_loader = PETdatasetHD(
    training_flname,
    num_train=num_train,
    is3d=is3d,
    batch_size=batch_size,
    test_size=test_size,
    valid_size=valid_size,
    num_workers=num_workers,
    target_transform=target_transform,
)
model = dinv.models.GSDRUNet(
    in_channels=1,
    out_channels=1,
    pretrained=pretrained_path,
    device=device,
)


# %%
sigma_noise = 50 / 255.0
physics = dinv.physics.Denoising(
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise)
)

x = next(iter(val_loader)).to(device)
y = physics(x)
recons = model(y, sigma_noise)

dinv.utils.plot([x, y, recons], figsize=(15, 10), titles=["x", "y", "recons"])

# %%
