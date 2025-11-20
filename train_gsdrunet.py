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

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


save_training_dir = r"./MoDL/trainFBSEM/brainweb/2D"

is3d = False
temPath = r"./tmp/"
radialBinCropFactor = 0.5
psf_cm = 0.15
niters = 8
nsubs = 6
training_flname = [save_training_dir + os.sep, "data-"]
save_dir = r"./MoDL/output/brainweb/2D" + os.sep
num_workers = 0
batch_size = 32
test_size = 0.0
valid_size = 0.1
# num_train = 1049
num_train = 500


# build PET object
PET = BuildGeometry_v4("mmr", radialBinCropFactor)
PET.loadSystemMatrix(temPath, is3d=False)

target_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
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
    pretrained=None,
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

save_path = Path("weights/GSDRUNet-brainweb/")
os.makedirs(save_path, exist_ok=True)

# Initialize wandb
with open(Path("wandb_api_key.txt"), "r") as f:
    api_key = f.read().strip()

wandb.login(key=api_key)
wandb.init(project="TIP-2024", name="GSDRUNet-brainweb")

trainer = Trainer(
    model=model,
    physics=physics,
    physics_generator=physics_generator,
    online_measurements=True,
    losses=loss,
    optimizer=optimizer,
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
    metrics=[NMSE()],
    eval_interval=1,
    device=device,
    epochs=epochs,
    ckp_interval=2,
    grad_clip=1.0,
    wandb_vis=True,
)

model = trainer.train()

print("Training complete.")
