# %%
import numpy as np
import torch
import deepinv as dinv
import matplotlib.pyplot as plt
from matplotlib import colors

img_idx = 15
reference = np.load(
    f"../tests/OSEM/20251118_142934/gt/gt_{str(img_idx).zfill(3)}.npy"
).squeeze()
osem_recon = np.load(
    f"../tests/OSEM/20251118_142934/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
mapem_recon = (
    np.load(f"../tests/MAPEM/20251118_142932/recons/recon_{str(img_idx).zfill(3)}.npy")
    .squeeze()
    .squeeze()
)
mrfbsem_recon = np.load(
    f"../tests/FBSEM-petmr/20251118_142933/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
fbsem_recon = np.load(
    f"../tests/FBSEM-pet/20251118_142918/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
pnpmm_pet_recon = np.load(
    f"../tests/PNPMM-pet/20251118_145825/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
pnpmm_nat_recon = np.load(
    f"../tests/PNPMM-nat/20251022_090703/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()


def center_crop(img, crop_size):
    h, w = img.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h : start_h + crop_size, start_w : start_w + crop_size]


crop_size = 128  # adjust as needed
reference_cropped = center_crop(reference, crop_size)
osem_cropped = center_crop(osem_recon, crop_size)
mapem_cropped = center_crop(mapem_recon, crop_size)
mrfbsem_cropped = center_crop(mrfbsem_recon, crop_size)
fbsem_cropped = center_crop(fbsem_recon, crop_size)
pnpmm_pet_cropped = center_crop(pnpmm_pet_recon, crop_size)
pnpmm_nat_cropped = center_crop(pnpmm_nat_recon, crop_size)


images_lc = [
    reference_cropped,
    osem_cropped,
    mapem_cropped,
    mrfbsem_cropped,
    fbsem_cropped,
    pnpmm_pet_cropped,
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


# --- shared norm for ALL error maps, centered at 0
# use a robust limit to avoid a single outlier blowing the scale
abs_vals = np.concatenate([np.abs(e[mask]).ravel() for e in error_maps_pct])
v = np.percentile(abs_vals, 99.5)  # robust symmetric range
norm = colors.TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

# --- originals: share vmin/vmax so their gray scales are comparable
vmin_img = min(img.min() for img in images_lc)
vmax_img = max(img.max() for img in images_lc)

fig, axes = plt.subplots(1, 7, figsize=(20, 4))
for ax, img, title in zip(
    axes,
    images_lc,
    [
        "Reference OSEM \n (high-dose)",
        "OSEM",
        "mr-MAP-EM",
        "mr-FBSEM",
        "FBSEM",
        "PnP-MM \n (pet prior)",
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

# --- error maps: all share the same norm/colorbar
fig, axes = plt.subplots(1, 7, figsize=(20, 4))
for ax, emap, title in zip(
    axes,
    error_maps_pct,
    [
        "Reference (no error)",
        "OSEM",
        "mr-MAP-EM",
        "mr-FBSEM",
        "FBSEM",
        "PnP-MM (pet prior)",
        "PnP-MM (natural prior)",
    ],
):
    im = ax.imshow(emap, cmap="bwr", norm=norm)
    ax.set_title(f"{title}", fontsize=12)
    ax.axis("off")

fig.colorbar(im, ax=axes.ravel().tolist(), label="Error (%)")
plt.show()

mse = dinv.metric.MSE()

nrmse_values = []
for img in images_lc[1:]:
    nrmse = (
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
    nrmse_values.append(nrmse)
print("NRMSE values of reconstructions:")
print(f"OSEM: {nrmse_values[0]:.2f}%")
print(f"mr-MAP-EM: {nrmse_values[1]:.2f}%")
print(f"mr-FBSEM: {nrmse_values[2]:.2f}%")
print(f"FBSEM: {nrmse_values[3]:.2f}%")
print(f"pet-PnP-MM: {nrmse_values[4]:.2f}%")
print(f"nat-PnP-MM: {nrmse_values[5]:.2f}%")
print("---------------------------")
print("Max values of reconstructions:")
print(f"OSEM: {osem_cropped.max():.2f}")
print(f"mr-MAP-EM: {mapem_cropped.max():.2f}")
print(f"mr-FBSEM: {mrfbsem_cropped.max():.2f}")
print(f"FBSEM: {fbsem_cropped.max():.2f}")
print(f"pet-PnP-MM: {pnpmm_pet_cropped.max():.2f}")
print(f"nat-PnP-MM: {pnpmm_nat_cropped.max():.2f}")
print(f"Reference: {reference_cropped.max():.2f}")
print("---------------------------")


# %%
