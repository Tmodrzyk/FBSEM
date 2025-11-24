# %%
import torch
from deepinv.loss import PSNR, NMSE
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
import datetime
from tqdm import tqdm

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


save_training_dir = r"./MoDL/testFBSEM/brainweb/2D"

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
test_size = 1.0
valid_size = 0.0
num_train = 500


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
pretrained_path = Path(
    "/home/modrzyk/code/FBSEM/weights/GSDRUNet-brainweb/25-11-18-09:22:17/ckp_499.pth.tar"
)
model = dinv.models.GSDRUNet(
    in_channels=1,
    out_channels=1,
    pretrained=pretrained_path,
    device=device,
)
model.detach = False

sigma_min = 0
sigma_max = 40

# Set up physics (e.g., denoising)
physics = dinv.physics.Denoising(
    noise_model=dinv.physics.GaussianNoise(
        sigma=0.1
    ),  # placeholder, will be overwritten by physics generator
    device=device,
)

physics_generator = dinv.physics.generator.SigmaGenerator(
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    device=device,
)

# Set up loss function and optimizer
loss = SupLoss(metric=MSE())

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
epochs = 500

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

sigma_noise = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
physics = dinv.physics.Denoising(
    noise_model=dinv.physics.GaussianNoise(sigma=0.1), device=device
)

model.eval()
nmse = NMSE()
log_file = Path(f"./tests/denoising/{timestamp}/results.log")
os.makedirs(log_file.parent, exist_ok=True)

with torch.no_grad():
    for sigma in sigma_noise:
        print(f"Testing for noise sigma: {sigma}")
        with open(log_file, "a") as f:
            f.write(f"Noise Sigma: {sigma}\n")
        physics.update_parameters(sigma=sigma)

        nmse_values = []

        for batch in tqdm(test_loader):
            x_gt = batch[0].to(device).unsqueeze(1)
            y = physics(x_gt)
            x_net = model(y, sigma)
            nmse_val = nmse(x_net, x_gt).sqrt()
            nmse_values.append(nmse_val.cpu().numpy())

        nmse_values = np.concatenate(nmse_values)
        mean_nrmse = np.mean(nmse_values)
        std_nrmse = np.std(nmse_values)
        print(f"Mean NRMSE: {mean_nrmse:.6f}, Std NRMSE: {std_nrmse:.6f}")
        with open(log_file, "a") as f:
            f.write(f"Mean NRMSE: {mean_nrmse:.6f}\n")
            f.write(f"Std NRMSE: {std_nrmse:.6f}\n")
print("Test complete.")

# %%
