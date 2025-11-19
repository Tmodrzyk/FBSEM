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
import numpy as np

import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append("/home/modrzyk/code/FBSEM")
from trainer import Trainer
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from models.deeplib import PETdatasetHD, dotstruct, toNumpy, crop
from models.modellib import FBSEMnet_v3, fbsemInference

save_training_dir = r"../MoDL/train/brainweb/2D"

is3d = False
temPath = r"./tmp/"
radialBinCropFactor = 0.5
psf_cm = 0.15
niters = 8
nsubs = 6
training_flname = [save_training_dir + os.sep, "data-"]
save_dir = r"./MoDL/output/brainweb/2D" + os.sep
num_workers = 0
batch_size = 4
test_size = 0.0
valid_size = 0.1
num_train = 1049


# build PET object
PET = BuildGeometry_v4("mmr", radialBinCropFactor)
PET.loadSystemMatrix(temPath, is3d=False)

target_transform = transforms.Compose(
    [
        transforms.ToTensor(),
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
    shuffle=True,
)
model = dinv.models.GSDRUNet(
    in_channels=1,
    out_channels=1,
    pretrained=Path(
        "/home/modrzyk/code/FBSEM/weights/GSDRUNet-brainweb/25-11-18-09:22:17/ckp_best.pth.tar"
    ),
    device=device,
)
# %%
sigma_noise = 20
physics = dinv.physics.Denoising(
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    device=device,
)

model.eval()
with torch.no_grad():
    x = next(iter(train_loader)).to(device)
    y = physics(x)
    x_net = model(y, sigma_noise)
    dinv.utils.plot([x, y, x_net], ["GGT", "Noisy", "Denoised"], figsize=(12, 12))

for i, arr in enumerate([x, y, x_net]):
    arr_np = arr.detach().cpu().numpy()
    # arr_np shape: (batch, channel, H, W)
    for b in range(arr_np.shape[0]):
        mn = arr_np[b].min()
        mx = arr_np[b].max()
        mean = arr_np[b].mean()
        std = arr_np[b].std()
        print(
            f"Tensor {['GGT', 'Noisy', 'Denoised'][i]}, Batch {b}: min={mn:.4f}, max={mx:.4f}, mean={mean:.4f}, std={std:.4f}"
        )

mse = MSE()

# %%
