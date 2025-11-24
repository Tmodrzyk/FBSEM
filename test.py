# %%
from geometry.BuildGeometry_v4 import BuildGeometry_v4
import numpy as np
import torch
import pathlib
import deepinv as dinv
from models.deeplib import DatasetPetMr_v2, dotstruct, toNumpy, crop
from phantoms.brainweb import PETbrainWebPhantom
import matplotlib.pyplot as plt
import seaborn as sns
from models.modellib import FBSEMnet_v3, Trainer, fbsemInference
from tqdm import tqdm
import random
import argparse
import datetime

sns.set_theme("notebook")

parser = argparse.ArgumentParser(description="PET reconstruction algorithms")
parser.add_argument(
    "--algo",
    type=str,
    choices=["OSEM", "MAPEM", "FBSEM-pet", "FBSEM-petmr", "PNPMM-nat", "PNPMM-pet"],
    default="OSEM",
    help="Choose reconstruction algorithm",
)
parser.add_argument(
    "--sigma_denoiser",
    type=float,
    default=5,
    help="Sigma value for the denoiser",
)
parser.add_argument(
    "--lambda_reg",
    type=float,
    default=0.3,
    help="Regularization parameter lambda",
)

device = "cuda"
temPath = r"./tmp"
phanPath = r"../phantoms/Brainweb/"
dataPath = r"./MoDL/testFBSEM/brainweb/2D/"
suffix = r"data-"
radialBinCropFactor = 0.5

psf_hd = 0.25
psf_ld = 0.4
niter_hd = 15
niter_ld = 10
nsubs_hd = 14
nsubs_ld = 14
counts_hd = 1e10
counts_ld = 1e6

args = parser.parse_args()

algorithm = args.algo
sigma_denoiser = args.sigma_denoiser
lambda_reg = args.lambda_reg

# Reproducilibity
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

PET = BuildGeometry_v4("mmr", radialBinCropFactor)
PET.loadSystemMatrix(temPath, is3d=False, tof=False)


dataset = DatasetPetMr_v2(
    filename=[dataPath, suffix],
    num_train=500,
    transform=None,
    target_transform=None,
    is3d=False,
    crop_factor=0,
)
test_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=0
)

mse = dinv.loss.metric.MSE()
nmse = dinv.loss.metric.NMSE()

# Testing loop
nrmse_values = []

if algorithm == "PNPMM-nat":
    ckpt_path = pathlib.Path("./weights/GSDRUNet_grayscale_torch.ckpt")
    denoiser = dinv.models.GSDRUNet(
        in_channels=1, out_channels=1, pretrained=ckpt_path
    ).to(device)
elif algorithm == "PNPMM-pet":
    ckpt_path = pathlib.Path(
        "./weights/GSDRUNet-brainweb/25-11-18-09:22:17/ckp_499.pth.tar"
    )
    denoiser = dinv.models.GSDRUNet(
        in_channels=1, out_channels=1, pretrained=ckpt_path
    ).to(device)

# Create timestamp and directory structure
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
recons_dir = pathlib.Path(f"./tests/{algorithm}/{timestamp}/recons/")
gt_dir = pathlib.Path(f"./tests/{algorithm}/{timestamp}/gt/")
recons_dir.mkdir(parents=True, exist_ok=True)
gt_dir.mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    for batch_idx, data in enumerate(tqdm(test_dataloader, desc="Testing")):
        sinoLD, imgHD, AN, RS, imgLD, imgLD_psf, mrImg, counts, imgGT, index = data
        # Perform reconstruction based on selected algorithm
        if algorithm == "OSEM":
            reconstructed = PET.OSEM2D(
                prompts=sinoLD.numpy(),
                img=None,
                AN=AN.numpy(),
                RS=None,
                niter=niter_ld,
                nsubs=nsubs_ld,
                psf=psf_ld,
            )
        elif algorithm == "MAPEM":
            reconstructed = PET.mrMAPEM2DBatch(
                prompts=sinoLD.numpy(),
                AN=AN.numpy(),
                mrImg=mrImg.numpy(),
                RS=None,
                beta=0.06,
                niters=10,
                nsubs=6,
                psf=psf_ld,
            )
        elif algorithm == "FBSEM-pet":
            fbsem_weights = r"./weights/FBSEM-brainweb/run3/fbsem-pm-03-epo-50.pth"
            reconstructed = fbsemInference(
                dl_model_flname=fbsem_weights,
                PET=PET,
                sinoLD=sinoLD,
                AN=AN,
                mrImg=None,
                niters=10,
                nsubs=6,
            )
        elif algorithm == "FBSEM-petmr":
            fbsem_weights = (
                r"/home/modrzyk/code/FBSEM/model_zoo/brainweb/2d/fbsem-pm-03-epo-45.pth"
            )
            reconstructed = fbsemInference(
                dl_model_flname=fbsem_weights,
                PET=PET,
                sinoLD=sinoLD,
                AN=AN,
                mrImg=mrImg,
                niters=10,
                nsubs=6,
            )
        elif algorithm == "PNPMM-nat" or algorithm == "PNPMM-pet":
            # Add PNPMM reconstruction logic here
            reconstructed, _ = PET.PnP_MM2D(
                prompts=sinoLD.numpy(),
                img=None,
                RS=None,
                AN=AN.numpy(),
                iSensImg=None,
                niter=50,
                nsubs=14,
                psf=psf_ld,
                denoiser=denoiser,
                sigma=sigma_denoiser,
                lambda_reg=lambda_reg,
                tau=100,
            )

        if isinstance(reconstructed, np.ndarray):
            reconstructed = torch.from_numpy(reconstructed).unsqueeze(0)

        # Save reconstruction and ground truth as numpy arrays
        np.save(recons_dir / f"recon_{batch_idx:03d}.npy", reconstructed.cpu().numpy())
        np.save(gt_dir / f"gt_{batch_idx:03d}.npy", imgHD.cpu().numpy())
        nmrse_val = nmse(reconstructed, imgHD).sqrt()
        nrmse_values.append(nmrse_val.item())

# Convert to numpy array and compute statistics
# Convert to numpy array and compute statistics
nrmse_array = np.array(nrmse_values)
mean_nrmse = np.mean(nrmse_array)
std_nrmse = np.std(nrmse_array)

# Save results to log file
log_file = pathlib.Path(f"./tests/{algorithm}/{timestamp}/results.log")
with open(log_file, "w") as f:
    f.write(f"Algorithm: {algorithm}\n")
    f.write(f"Parameters: sigma_denoiser={sigma_denoiser}, lambda_reg={lambda_reg}\n")
    f.write(f"Timestamp: {timestamp}\n\n")

    f.write("NRMSE values per image:\n")
    for i, nrmse_val in enumerate(nrmse_values):
        f.write(f"Image {i:03d}: {nrmse_val:.6f}\n")

    f.write(f"\nStatistics:\n")
    f.write(f"Mean NRMSE: {mean_nrmse:.6f}\n")
    f.write(f"Std NRMSE: {std_nrmse:.6f}\n")

print(f"Results saved to: {log_file}")
print(f"Mean NRMSE: {mean_nrmse:.6f}")
print(f"Std NRMSE: {std_nrmse:.6f}")
