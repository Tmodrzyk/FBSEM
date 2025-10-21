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

sns.set_theme("notebook")
data_path = pathlib.Path(r"./MoDL/trainDatasets/brainweb/2D/data-0.npy")
data = np.load(data_path, allow_pickle=True).item()

dinv.utils.plot(
    [
        torch.from_numpy(data["mrImg"]).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(data["imgHD"]).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(data["imgLD"]).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(data["imgLD_psf"]).unsqueeze(0).unsqueeze(0),
    ],
    figsize=(20, 10),
)

print(data.keys())
print(data["phanType"], data["phanPath"])
# %%

temPath = r"./tmp"
phanPath = r"../phantoms/Brainweb/"
radialBinCropFactor = 0.5

PET = BuildGeometry_v4("mmr", radialBinCropFactor)
PET.loadSystemMatrix(temPath, is3d=False, tof=False)

img_3d, mumap_3d, t1_3d, _ = PETbrainWebPhantom(
    phanPath,
    phantom_number=1,
    voxel_size=np.array(PET.image.voxelSizeCm) * 10,
    image_size=PET.image.matrixSize,
    pet_lesion=False,
    t1_lesion=False,
    num_lesions=0,
    hot_cold_ratio=0.9,
)

img_2d = img_3d[:, :, 50]
mumap_2d = mumap_3d[:, :, 50]
t1_2d = t1_3d[:, :, 50]
psf_cm = 0.25

dinv.utils.plot(
    [
        torch.from_numpy(img_2d).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(mumap_2d).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(t1_2d).unsqueeze(0).unsqueeze(0),
    ],
    figsize=(15, 5),
)
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
iter_pnpmm = 80
# pretrained_path = pathlib.Path("./weights/25-10-13-14:12:28/ckp_best.pth.tar")
denoiser = dinv.models.GSDRUNet(
    in_channels=1, out_channels=1, pretrained="download"
).to("cuda")

sigma_denoiser_ld = 5 / 255.0
lambda_reg_ld = 0.15
stepsize_ld = 1e3

pnp_mm_ld, xs = PET.PnP_MM2D(
    prompts=y_ld,
    img=None,
    RS=None,
    AN=AN_ld,
    iSensImg=None,
    niter=iter_pnpmm,
    nsubs=1,
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

mse = dinv.metric.MSE()
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
print(
    "NMSE OSEM LD: ",
    (
        torch.sqrt(
            mse(
                torch.from_numpy(osem_cropped).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
        / torch.norm(
            torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0)
        ).sqrt()
    ).item()
    * 100,
)
print(
    "NMSE FBSEM LD: ",
    (
        torch.sqrt(
            mse(
                torch.from_numpy(fbsem_cropped).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
        / torch.norm(
            torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0)
        ).sqrt()
    ).item()
    * 100,
)
print(
    "NMSE MAP EM LD: ",
    (
        torch.sqrt(
            mse(
                torch.from_numpy(mapem_cropped).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
        / torch.norm(
            torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0)
        ).sqrt()
    ).item()
    * 100,
)
print(
    "NMSE PnP MM LD: ",
    (
        torch.sqrt(
            mse(
                torch.from_numpy(pnpmm_nat_cropped).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
        / torch.norm(
            torch.from_numpy(reference_cropped).unsqueeze(0).unsqueeze(0)
        ).sqrt()
    ).item()
    * 100,
)

print("Max values of reconstructions:")
print(f"OSEM LD max: {osem_ld.max():.6f}")
print(f"MAP-EM LD max: {map_em_ld.max():.6f}")
print(f"FBSEM LD max: {fbsem_ld.max():.6f}")
print(f"PnP-MM LD max: {pnp_mm_ld.max():.6f}")
print(f"OSEM HD max: {osem_hd.max():.6f}")

# %%
for error_map in error_maps_pct:
    print(error_map.min(), error_map.max())
