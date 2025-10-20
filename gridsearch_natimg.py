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
import optuna
from tqdm import tqdm
import random

sns.set_theme("notebook")

device = "cuda"

temPath = r"./tmp"
phanPath = r"../phantoms/Brainweb/"
dataPath = r"./MoDL/valDatasets/brainweb/2D/"
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


# Reproducilibity
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

PET = BuildGeometry_v4("mmr", radialBinCropFactor)
PET.loadSystemMatrix(temPath, is3d=False, tof=False)


def objective(trial, dataset, denoiser):
    """
    Objective function for Optuna to optimize.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        dataset (torch.utils.data.Dataset): Dataset to evaluate.
        denoiser (SimpleDenoiser): Denoising model.
        device (str): Device to run computations on.

    Returns:
        tuple: (average PSNR, average SSIM) across the dataset to maximize.
    """
    # Suggest hyperparameters
    sigma_denoiser = trial.suggest_int("sigma", 1, 60) / 255
    lambda_reg = trial.suggest_float("lambda_reg", 0.1, 0.99)

    # Initialize metrics
    num_samples = len(dataset)

    # Define loss functions
    mse = dinv.metric.MSE()

    # Create a DataLoader for batch processing if needed
    val_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    denoiser.alpha = lambda_reg

    avg_nrmse = 0.0

    # Iterate over the dataset and the blur kernels
    for data in tqdm(val_dataloader, desc="Evaluating Trial", disable=True):

        sinoLD, imgHD, AN, RS, imgLD, imgLD_psf, mrImg, counts, imgGT, index = data
        pnp_mm_ld, xs = PET.PnP_MM2D(
            prompts=sinoLD.numpy(),
            img=None,
            RS=None,
            AN=AN.numpy(),
            iSensImg=None,
            niter=60,
            nsubs=1,
            psf=psf_ld,
            denoiser=denoiser,
            sigma=sigma_denoiser,
            lambda_reg=lambda_reg,
            tau=1e2,
        )

        pnp_mm_ld = torch.from_numpy(pnp_mm_ld).unsqueeze(0).to(device)
        nmrse_val = (
            mse(
                pnp_mm_ld,
                imgHD.to(device),
            ).sqrt()
            / torch.norm(imgHD.to(device)).sqrt()
        ) * 100
        avg_nrmse += nmrse_val.item()

    avg_nrmse /= num_samples

    return avg_nrmse


dataset = DatasetPetMr_v2(
    filename=[dataPath, suffix],
    num_train=20,
    transform=None,
    target_transform=None,
    is3d=False,
    crop_factor=0,
)

# Define Optuna study for multi-objective optimization (maximize PSNR and SSIM)
study = optuna.create_study(
    directions=["minimize"],  # NRMSE
    study_name=f"gridsearch_pnpmm_natural_images",
    storage=f"sqlite:///gridsearch_pnpmm_natural_images.db",  # Persist study results
    load_if_exists=True,  # Continue from existing study if available
)
ckpt_path = pathlib.Path("./weights/GSDRUNet_grayscale_torch.ckpt")
denoiser = dinv.models.GSDRUNet(in_channels=1, out_channels=1, pretrained=ckpt_path).to(
    device
)

# Optimizeg
study.optimize(
    lambda trial: objective(trial, dataset, denoiser),
    n_trials=500,
)

# %%
