# %%
import numpy as np
import torch
import pathlib
import deepinv as dinv
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("../../")
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from phantoms.brainweb import PETbrainWebPhantom
from models.modellib import FBSEMnet_v3, Trainer, fbsemInference

sns.set_theme("notebook")

temPath = r"./tmp"
phanPath = r"../phantoms/Brainweb/"
radialBinCropFactor = 0.5

PET = BuildGeometry_v4("mmr", radialBinCropFactor)
PET.loadSystemMatrix(temPath, is3d=False, tof=False)

img_3d, mumap_3d, t1_3d, t2_3d = PETbrainWebPhantom(
    phanPath,
    phantom_number=18,
    voxel_size=np.array(PET.image.voxelSizeCm) * 10,
    image_size=PET.image.matrixSize,
    pet_lesion=True,
    t1_lesion=True,
    t2_lesion=True,
    num_lesions=15,
    hot_cold_ratio=0.8,
)
# %%
slice_index = 65
img_2d = img_3d[:, :, slice_index]
mumap_2d = mumap_3d[:, :, slice_index]
t1_2d = t1_3d[:, :, slice_index]
t2_2d = t2_3d[:, :, slice_index]
psf_cm = 0.25

dinv.utils.plot(
    [
        torch.from_numpy(img_2d).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(t1_2d).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(t2_2d).unsqueeze(0).unsqueeze(0),
    ],
    titles=["PET phantom", "T1-weighted MRI", "T2-weighted MRI"],
    cmap="gist_gray_r",
    figsize=(15, 5),
    fontsize=25,
)

# %%
## 2D forward project
# y = PET.forwardProjectBatch2D(img_2d, psf = psf_cm)
# y_batch = PET.forwardProjectBatch2D(img_2d_batch, psf = psf_cm)


psf_hd = 0.25
psf_ld = 0.4
niter_hd = 15
niter_ld = 10
nsubs_hd = 14
nsubs_ld = 14
counts_hd = 1e10
counts_ld = 1e6

# simulate 2D noisy sinograms
y_hd, AF_hd, NF_hd, Randoms_hd = PET.simulateSinogramData(
    img_2d, mumap=mumap_2d, counts=counts_hd, psf=psf_hd
)
y_ld, AF_ld, NF_ld, Randoms_ld = PET.simulateSinogramData(
    img_2d, mumap=mumap_2d, counts=counts_ld, psf=psf_ld
)
# %%
# 2D OSEM
AN_hd = AF_hd * NF_hd
AN_ld = AF_ld * NF_ld
osem_hd = PET.OSEM2D(y_hd, AN=AN_hd, niter=niter_hd, nsubs=nsubs_hd, psf=psf_hd)
osem_ld = PET.OSEM2D(y_ld, AN=AN_ld, niter=niter_ld, nsubs=nsubs_ld, psf=psf_ld)
map_em_ld = PET.mrMAPEM2DBatch(
    np.expand_dims(y_ld, axis=0),
    AN=np.expand_dims(AN_ld, axis=0),
    mrImg=np.expand_dims(t1_2d, axis=0),
    beta=0.06,
    niters=10,
    nsubs=6,
    psf=psf_ld,
).squeeze()
dl_model_flname = (
    r"/home/modrzyk/code/FBSEM/model_zoo/brainweb/2d/fbsem-pm-03-epo-45.pth"
)
fbsem_ld = fbsemInference(
    dl_model_flname,
    PET,
    torch.from_numpy(y_ld).unsqueeze(0),
    torch.from_numpy(AN_ld).unsqueeze(0),
    torch.from_numpy(t1_2d).unsqueeze(0),
    niters=10,
    nsubs=6,
)


# %%
def center_crop(img, crop_size):
    h, w = img.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h : start_h + crop_size, start_w : start_w + crop_size]


mse = dinv.metric.MSE()


crop_size = 128  # adjust as needed
reference_cropped = center_crop(osem_hd, crop_size)
osem_cropped = center_crop(osem_ld, crop_size)
mapem_cropped = center_crop(map_em_ld, crop_size)
fbsem_cropped = center_crop(fbsem_ld, crop_size)

images_lc = [
    reference_cropped,
    osem_cropped,
    mapem_cropped,
    fbsem_cropped,
]

# --- originals: share vmin/vmax so their gray scales are comparable
vmin_img = min(img.min() for img in images_lc)
vmax_img = max(img.max() for img in images_lc)

nmse_results = {}
methods = [
    ("OSEM LD", osem_cropped),
    ("MAP EM LD", mapem_cropped),
    ("FBSEM LD", fbsem_cropped),
]
for name, img in methods:
    nmse = (
        torch.sqrt(
            mse(
                torch.from_numpy(img).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
        / torch.norm(
            torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0)
        ).sqrt()
    ).item() * 100
    nmse_results[name] = nmse
    print(f"NMSE {name}: {nmse:.4f}")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    titles = [
        "Reference OSEM \n (high-dose)",
        "OSEM",
        "mr-MAP-EM",
        "mr-FBSEM",
    ]
    nmse_methods = [
        None,  # Reference has no NMSE
        nmse_results.get("OSEM LD", None),
        nmse_results.get("MAP EM LD", None),
        nmse_results.get("FBSEM LD", None),
    ]
    for ax, img, title, nmse in zip(axes, images_lc, titles, nmse_methods):
        im0 = ax.imshow(img, cmap="gist_gray_r", vmin=vmin_img, vmax=vmax_img)
        ax.set_title(title, fontsize=24)
        ax.axis("off")
        # Add subtitle with NMSE below the image
        if nmse is not None:
            ax.text(
                0.5,
                0.1,
                f"NMSE: {nmse:.2f}%",
                fontsize=18,
                ha="center",
                va="top",
                transform=ax.transAxes,
            )
    cbar0 = fig.colorbar(im0, ax=axes.ravel().tolist())
    cbar0.set_label("Intensity (a.u.)")
    plt.show()
    plt.close()
# %%
pretrained_path = pathlib.Path(
    "/home/modrzyk/code/FBSEM/weights/GSDRUNet-brainweb/25-11-05-15:09:25/ckp_best.pth.tar"
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
sigma_denoiser_ld = 1
lambda_reg_ld = 0.1
stepsize_ld = 10

pnp_mm_ld, xs = PET.PnP_MM2D(
    prompts=y_ld,
    img=None,
    RS=None,
    AN=AN_ld,
    iSensImg=None,
    niter=iter_pnpmm,
    nsubs=14,
    psf=psf_ld,
    denoiser=denoiser,
    sigma=sigma_denoiser_ld,
    lambda_reg=lambda_reg_ld,
    tau=stepsize_ld,
)
# Center crop the images first
from matplotlib import colors


crop_size = 128  # adjust as needed
reference_cropped = center_crop(osem_hd, crop_size)
osem_cropped = center_crop(osem_ld, crop_size)
pnpmm_nat_cropped = center_crop(pnp_mm_ld, crop_size)
mapem_cropped = center_crop(map_em_ld, crop_size)
fbsem_cropped = center_crop(fbsem_ld, crop_size)

images_lc = [
    reference_cropped,
    osem_cropped,
    mapem_cropped,
    fbsem_cropped,
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

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, img, title in zip(
    axes,
    images_lc,
    [
        "Reference OSEM \n (high-dose)",
        "OSEM",
        "mr-MAP-EM",
        "mr-FBSEM",
        "PnP-MM \n (natural image prior)",
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
v = np.percentile(abs_vals, 99.5)  # robust symmetric range
norm = colors.TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

# --- error maps: all share the same norm/colorbar
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, emap, title in zip(
    axes,
    error_maps_pct,
    [
        "Reference (no error)",
        "OSEM",
        "mr-MAP-EM",
        "mr-FBSEM",
        "PnP-MM \n (natural image prior)",
    ],
):
    im = ax.imshow(emap, cmap="bwr", norm=norm)
    ax.set_title(f"{title}", fontsize=12)
    ax.axis("off")

fig.colorbar(im, ax=axes.ravel().tolist(), label=r"Error (\%)")
plt.show()

rnmse_pnpmm = [
    (
        torch.sqrt(
            mse(
                torch.from_numpy(center_crop(x, crop_size)).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
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
nmse_results = {}
methods = [
    ("OSEM LD", osem_cropped),
    ("MAP EM LD", mapem_cropped),
    ("FBSEM LD", fbsem_cropped),
    ("PnP MM LD", pnpmm_nat_cropped),
]
for name, img in methods:
    nmse = (
        torch.sqrt(
            mse(
                torch.from_numpy(img).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
        / torch.norm(
            torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0)
        ).sqrt()
    ).item() * 100
    nmse_results[name] = nmse
    print(f"NMSE {name}: {nmse:.4f}")

print("Max values of reconstructions:")
print(f"OSEM LD max: {osem_ld.max():.6f}")
print(f"MAP-EM LD max: {map_em_ld.max():.6f}")
print(f"FBSEM LD max: {fbsem_ld.max():.6f}")
print(f"PnP-MM LD max: {pnp_mm_ld.max():.6f}")
print(f"OSEM HD max: {osem_hd.max():.6f}")

# %%
for error_map in error_maps_pct:
    print(error_map.min(), error_map.max())
