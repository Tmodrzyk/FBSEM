# %%
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

sns.set_theme("notebook")

temPath = r"./tmp"
phanPath = r"../phantoms/Brainweb/"
radialBinCropFactor = 0.5

PET = BuildGeometry_v4("mmr", radialBinCropFactor)
PET.loadSystemMatrix(temPath, is3d=False, tof=False)

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
## 2D forward project
# y = PET.forwardProjectBatch2D(img_2d, psf = psf_cm)
# y_batch = PET.forwardProjectBatch2D(img_2d_batch, psf = psf_cm)
print(sinoLD.shape, AN.shape, mrImg.shape)
# %%
# 2D OSEM
map_em_ld = PET.mrMAPEM2DBatch(
    prompts=sinoLD.numpy(),
    AN=AN.numpy(),
    mrImg=mrImg.numpy(),
    beta=0.06,
    niters=10,
    nsubs=6,
    psf=psf_ld,
).squeeze()

# %%
# 2D FBSEM
dl_model_flname = (
    r"/home/modrzyk/code/FBSEM/model_zoo/brainweb/2d/fbsem-pm-03-epo-45.pth"
)
mrfbsem_ld = fbsemInference(
    dl_model_flname,
    PET,
    sinoLD,
    AN,
    mrImg=mrImg,
    niters=10,
    nsubs=6,
)

dl_model_flname = (
    r"/home/modrzyk/code/FBSEM/weights/FBSEM-brainweb/fbsem-pm-03-epo-49.pth"
)
fbsem_ld = fbsemInference(
    dl_model_flname,
    PET,
    sinoLD=sinoLD,
    AN=AN,
    mrImg=None,
    niters=10,
    nsubs=6,
)


# %%
pretrained_path = pathlib.Path(
    "/home/modrzyk/code/FBSEM/weights/GSDRUNet-brainweb/25-11-18-09:22:17/ckp_best.pth.tar"
)
denoiser = dinv.models.GSDRUNet(
    in_channels=1, out_channels=1, pretrained=pretrained_path
).to("cuda")
# denoiser = dinv.models.GSDRUNet(
#     in_channels=1, out_channels=1, pretrained="download"
# ).to("cuda")


psf_ld = 0.4
nsubs_ld = 14
iter_pnpmm = 30
sigma_denoiser_ld = 8
lambda_reg_ld = 0.2
stepsize_ld = 100

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
    sigma=sigma_denoiser_ld,
    lambda_reg=lambda_reg_ld,
    tau=stepsize_ld,
)
# %%
# Center crop the images first
from matplotlib import colors


def center_crop(img, crop_size):
    h, w = img.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h : start_h + crop_size, start_w : start_w + crop_size]


crop_size = 128  # adjust as needed
reference_cropped = center_crop(imgGT.squeeze().numpy(), crop_size)
osem_cropped = center_crop(imgLD_psf.squeeze().numpy(), crop_size)
pnpmm_nat_cropped = center_crop(pnp_mm_ld, crop_size)
mapem_cropped = center_crop(map_em_ld, crop_size)
mrfbsem_cropped = center_crop(mrfbsem_ld, crop_size)
fbsem_ld_cropped = center_crop(fbsem_ld, crop_size)

images_lc = [
    reference_cropped,
    osem_cropped,
    mapem_cropped,
    mrfbsem_cropped,
    fbsem_ld_cropped,
    pnpmm_nat_cropped,
]
# Calculate relative error maps
eps = 1e-8
mask = reference_cropped > (0.02 * reference_cropped.max())

# --- define percentage error (pixelwise relative to reference where mask=True)
error_maps_pct = []
for img in images_lc:
    rel = np.zeros_like(reference_cropped, dtype=float)
    denom = np.maximum(reference_cropped, eps)
    rel[mask] = 100.0 * (img[mask] - reference_cropped[mask]) / denom[mask]
    error_maps_pct.append(rel)

# Create subplot with 2 rows


# --- originals: share vmin/vmax so their gray scales are comparable
vmin_img = min(img.min() for img in images_lc)
vmax_img = max(img.max() for img in images_lc)

fig, axes = plt.subplots(1, 6, figsize=(20, 4))
for ax, img, title in zip(
    axes,
    images_lc,
    [
        "Reference OSEM \n (high-dose)",
        "OSEM",
        "mr-MAP-EM",
        "mr-FBSEM",
        "FBSEM",
        "PnP-MM",
    ],
):
    im0 = ax.imshow(img, cmap="gist_gray_r", vmin=vmin_img, vmax=vmax_img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
cbar0 = fig.colorbar(im0, ax=axes.ravel().tolist())
cbar0.set_label("Intensity (a.u.)")
plt.show()
plt.close()

# --- shared norm for ALL error maps, centered at 0
# use a robust limit to avoid a single outlier blowing the scale
abs_vals = np.concatenate([np.abs(e[mask]).ravel() for e in error_maps_pct])
v = np.percentile(abs_vals, 90)  # robust symmetric range
norm = colors.TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

# --- error maps: all share the same norm/colorbar
fig, axes = plt.subplots(1, 6, figsize=(20, 4))
for ax, emap, title in zip(
    axes,
    error_maps_pct,
    [
        "Reference (no error)",
        "OSEM",
        "mr-MAP-EM",
        "mr-FBSEM",
        "FBSEM",
        "PnP-MM",
    ],
):
    im = ax.imshow(emap, cmap="bwr", norm=norm)
    ax.set_title(f"{title}", fontsize=12)
    ax.axis("off")

fig.colorbar(im, ax=axes.ravel().tolist(), label=r"Error (\%)")
plt.show()

mse = dinv.metric.MSE()
rnmse_pnpmm = [
    (
        mse(
            torch.from_numpy(center_crop(x, crop_size)).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0),
        ).sqrt()
        / torch.norm(
            torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0)
        ).sqrt()
    )
    * 100
    for x in xs
]

plt.plot(rnmse_pnpmm)
plt.show()
# Refactored NMSE calculation and printing for all methods
rnmse_results = {}
methods = [
    ("OSEM LD", osem_cropped),
    ("MAP EM LD", mapem_cropped),
    ("MR-FBSEM LD", mrfbsem_cropped),
    ("FBSEM LD", fbsem_ld_cropped),
    ("PnP MM LD", pnpmm_nat_cropped),
]
for name, img in methods:
    rnmse = (
        mse(
            torch.from_numpy(img).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0),
        ).sqrt()
        / torch.norm(
            torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0)
        ).sqrt()
    ).item() * 100
    rnmse_results[name] = rnmse
    print(f"RNMSE {name}: {rnmse:.4f}")
print("\n")
print("Max values of reconstructions:")
print(f"OSEM LD max: {imgLD_psf.max():.6f}")
print(f"MAP-EM LD max: {map_em_ld.max():.6f}")
print(f"MR-FBSEM LD max: {mrfbsem_ld.max():.6f}")
print(f"FBSEM LD max: {fbsem_ld.max():.6f}")
print(f"PnP-MM LD max: {pnp_mm_ld.max():.6f}")
print(f"OSEM HD max: {imgHD.max():.6f}")

# %%
for error_map in error_maps_pct:
    print(error_map.min(), error_map.max())
