# %%
import sys

sys.path.append("..")
from geometry.BuildGeometry_v4 import BuildGeometry_v4
import numpy as np
import torch
import pathlib
import deepinv as dinv
from phantoms.brainweb import PETbrainWebPhantom
import matplotlib.pyplot as plt
import seaborn as sns
from models.modellib import FBSEMnet_v3, Trainer, fbsemInference
from models.deeplib import DatasetPetMr_v2, dotstruct, toNumpy, crop
import random
import deepinv as dinv

sns.set_theme("notebook")

temPath = r"./tmp"
phanPath = r"../phantoms/Brainweb/"
radialBinCropFactor = 0.5

PET = BuildGeometry_v4("mmr", radialBinCropFactor)
PET.loadSystemMatrix(temPath, is3d=False, tof=False)

device = "cuda"
temPath = r"./tmp"
phanPath = r"../phantoms/Brainweb/"
dataPath = r"../MoDL/testFBSEM/brainweb/2D/"
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


dataset = DatasetPetMr_v2(
    filename=[dataPath, suffix],
    num_train=100,
    transform=None,
    target_transform=None,
    is3d=False,
    crop_factor=0,
)
test_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=0
)
# %%
sinoLD, imgHD, AN, RS, imgLD, imgLD_psf, mrImg, counts, imgGT, index = next(
    iter(
        torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    )
)


dinv.utils.plot(
    [
        imgGT,
        imgLD_psf,
        mrImg,
    ],
    titles=["High-count OSEM", "Low-count OSEM", "T1-weighted MRI"],
    cmap="gist_gray_r",
    figsize=(15, 5),
)
# %%
pretrained_path = pathlib.Path(
    "/home/modrzyk/code/FBSEM/weights/GSDRUNet-brainweb/25-11-18-09:22:17/ckp_499.pth.tar"
)
denoiser = dinv.models.GSDRUNet(
    in_channels=1, out_channels=1, pretrained=pretrained_path
).to("cuda")
# denoiser = dinv.models.GSDRUNet(
#     in_channels=1, out_channels=1, pretrained="download"
# ).to("cuda")

# Vary sigma_denoiser and compare results
sigma_values = [0, 5, 10, 15, 25, 40]
pnp_mm_results = {}
xs_results = {}
rnmse_curves = {}


def center_crop(img, crop_size):
    h, w = img.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h : start_h + crop_size, start_w : start_w + crop_size]


for sigma in sigma_values:
    iter_pnpmm = 50
    lambda_reg_ld = 0.95
    stepsize_ld = 100

    nmse = dinv.metric.NMSE()

    pnp_mm_ld, xs = PET.PnP_MM2D(
        prompts=sinoLD.numpy(),
        img=None,
        RS=None,
        AN=AN.numpy(),
        iSensImg=None,
        niter=iter_pnpmm,
        nsubs=14,
        psf=psf_ld,
        denoiser=denoiser,
        sigma=sigma,
        lambda_reg=lambda_reg_ld,
        tau=stepsize_ld,
    )
    pnp_mm_results[sigma] = pnp_mm_ld
    np.save(f"./figure4/reconstruction_sigma_{sigma}.npy", pnp_mm_ld)
    xs_results[sigma] = xs

    # Calculate RNMSE curve
    rnmse = [
        nmse(
            torch.from_numpy(x).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(imgGT.squeeze().numpy()).unsqueeze(0).unsqueeze(0),
        )
        .sqrt()
        .item()
        * 100
        for x in xs[1:]
    ]
    rnmse_curves[sigma] = rnmse
# %%
from matplotlib import colors

vmin_img = min([img.min() for img in pnp_mm_results.values()])
vmax_img = max([img.max() for img in pnp_mm_results.values()])
crop_size = 110

# Plot reconstructions side by side
fig, axes = plt.subplots(1, 6, figsize=(20, 4))
for ax, sigma in zip(axes, sigma_values):
    img = center_crop(pnp_mm_results[sigma], crop_size)
    im = ax.imshow(img, cmap="gist_gray_r", vmin=vmin_img, vmax=vmax_img)
    rnmse_final = rnmse_curves[sigma][-1]
    ax.set_title(
        f"$\lambda$ = {lambda_reg_ld}\n$\sigma$ = {sigma}\nRNMSE = {rnmse_final:.2f}%",
        fontsize=12,
    )
    ax.axis("off")
fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()

# %%
# Vary lambda_reg_ld and compare results
lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
pnp_mm_results_lambda = {}
xs_results_lambda = {}
rnmse_curves_lambda = {}

for lambda_reg in lambda_values:
    iter_pnpmm = 50
    sigma_ld = 20
    stepsize_ld = 100

    nmse = dinv.metric.NMSE()

    pnp_mm_ld, xs = PET.PnP_MM2D(
        prompts=sinoLD.numpy(),
        img=None,
        RS=None,
        AN=AN.numpy(),
        iSensImg=None,
        niter=iter_pnpmm,
        nsubs=14,
        psf=psf_ld,
        denoiser=denoiser,
        sigma=sigma_ld,
        lambda_reg=lambda_reg,
        tau=stepsize_ld,
    )
    pnp_mm_results_lambda[lambda_reg] = pnp_mm_ld
    xs_results_lambda[lambda_reg] = xs
    np.save(f"./figure4/reconstruction_lambda_{lambda_reg}.npy", pnp_mm_ld)

    # Calculate RNMSE curve
    rnmse = [
        nmse(
            torch.from_numpy(x).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(imgGT.squeeze().numpy()).unsqueeze(0).unsqueeze(0),
        )
        .sqrt()
        .item()
        * 100
        for x in xs[1:]
    ]
    rnmse_curves_lambda[lambda_reg] = rnmse
# %%
# Plot reconstructions side by side
fig, axes = plt.subplots(1, 6, figsize=(20, 4))
for ax, lambda_reg in zip(axes, lambda_values):
    img = center_crop(pnp_mm_results_lambda[lambda_reg], crop_size)
    im = ax.imshow(img, cmap="gist_gray_r", vmin=vmin_img, vmax=vmax_img)
    rnmse_final = rnmse_curves_lambda[lambda_reg][-1]
    ax.set_title(f"$\lambda$ = {lambda_reg}\nRNMSE = {rnmse_final:.2f}%", fontsize=12)
    ax.axis("off")
fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()

# %%
# Load all images to determine global vmin and vmax for consistent coloring
all_images = []
for lambda_reg in lambda_values:
    img = np.load(f"./figure4/reconstruction_lambda_{lambda_reg}.npy")
    all_images.append(img)
for sigma in sigma_values:
    img = np.load(f"./figure4/reconstruction_sigma_{sigma}.npy")
    all_images.append(img)

vmin_global = min(img.min() for img in all_images)
vmax_global = max(img.max() for img in all_images)
crop_size = 100

# Create the figure
fig, axes = plt.subplots(2, 6, figsize=(20, 7))
fontsize = 16
# Plot varying lambda results
for i, lambda_reg in enumerate(lambda_values):
    ax = axes[0, i]
    img = np.load(f"./figure4/reconstruction_lambda_{lambda_reg}.npy")
    img_cropped = center_crop(img, crop_size)
    im = ax.imshow(img_cropped, cmap="gist_gray_r", vmin=vmin_global, vmax=vmax_global)
    rnmse_final = rnmse_curves_lambda[lambda_reg][-1]
    ax.set_title(
        f"$\sigma$ = {sigma_ld}\n$\lambda$ = {lambda_reg}\nRNMSE = {rnmse_final:.2f}%",
        fontsize=fontsize,
    )
    ax.axis("off")

# Plot varying sigma results
for i, sigma in enumerate(sigma_values):
    ax = axes[1, i]
    img = np.load(f"./figure4/reconstruction_sigma_{sigma}.npy")
    img_cropped = center_crop(img, crop_size)
    im = ax.imshow(img_cropped, cmap="gist_gray_r", vmin=vmin_global, vmax=vmax_global)
    rnmse_final = rnmse_curves[sigma][-1]
    ax.set_title(
        f"$\sigma$ = {sigma}\n$\lambda$ = {lambda_reg_ld}\nRNMSE = {rnmse_final:.2f}%",
        fontsize=fontsize,
    )
    ax.axis("off")

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02, anchor=(0.0, 1.2))
plt.savefig("figure4.pdf", bbox_inches="tight", dpi=300)
plt.show()
