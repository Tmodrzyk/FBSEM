# %%
import numpy as np
import torch
import deepinv as dinv
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

img_idx = 311  # mr-FBSEM artefact
reference = np.load(
    f"../tests/OSEM/20251124_112527/gt/gt_{str(img_idx).zfill(3)}.npy"
).squeeze()
osem_recon = np.load(
    f"../tests/OSEM/20251124_112527/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
mapem_recon = (
    np.load(f"../tests/MAPEM/20251124_112530/recons/recon_{str(img_idx).zfill(3)}.npy")
    .squeeze()
    .squeeze()
)
mrfbsem_recon = np.load(
    f"../tests/FBSEM-petmr/20251124_112537/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
fbsem_recon = np.load(
    f"../tests/FBSEM-pet/20251124_112659/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
# pnpmm_pet_recon = np.load(
#     f"../tests/PNPMM-pet/20251124_112720/recons/recon_{str(img_idx).zfill(3)}.npy"
# ).squeeze()
pnpmm_pet_recon = np.load(
    f"../tests/PNPMM-pet/20251124_142235/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
data = np.load(
    f"/home/modrzyk/code/FBSEM/MoDL/testFBSEM/brainweb/2D/data-{str(img_idx)}.npy",
    allow_pickle=True,
).item()
t1 = data["mrImg"]
imgs = [
    reference,
    osem_recon,
    fbsem_recon,
    pnpmm_pet_recon,
    mapem_recon,
    mrfbsem_recon,
]
nmse = dinv.metric.NMSE()

rnmse_values = [
    nmse(torch.from_numpy(img).unsqueeze(0), torch.from_numpy(reference).unsqueeze(0))
    .sqrt()
    .item()
    for img in imgs
]


def center_crop(img, crop_size):
    h, w = img.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h : start_h + crop_size, start_w : start_w + crop_size]


crop_size = 100  # adjust as needed
reference_cropped = center_crop(reference, crop_size)
osem_cropped = center_crop(osem_recon, crop_size)
mapem_cropped = center_crop(mapem_recon, crop_size)
mrfbsem_cropped = center_crop(mrfbsem_recon, crop_size)
fbsem_cropped = center_crop(fbsem_recon, crop_size)
pnpmm_pet_cropped = center_crop(pnpmm_pet_recon, crop_size)
t1_cropped = center_crop(t1, crop_size)
images_lc = [
    reference_cropped,
    osem_cropped,
    fbsem_cropped,
    pnpmm_pet_cropped,
    mapem_cropped,
    mrfbsem_cropped,
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
fig, axes = plt.subplots(1, 6, figsize=(30, 20))

# Define the zoom-in rectangle (x, y, width, height) in cropped image coordinates
zoom_x, zoom_y, zoom_w, zoom_h = 45, 28, 27, 27

for ax, img, title, rnmse in zip(
    axes,
    images_lc,
    [
        "Reference OSEM \n(high-dose)",
        "OSEM \n(low-dose)",
        "FBSEM",
        "PnP-MM",
        "mr-MAP-EM",
        "mr-FBSEM",
    ],
    rnmse_values,
):
    im0 = ax.imshow(img, cmap="gist_gray_r", vmin=vmin_img, vmax=vmax_img)
    ax.set_title(f"{title}\nNRMSE: {100*rnmse:.2f}%", fontsize=12)
    ax.axis("off")

    # Create a rectangle patch to indicate the zoom area on the main image
    ax.add_patch(
        Rectangle(
            (zoom_x, zoom_y),
            zoom_w,
            zoom_h,
            edgecolor="red",
            facecolor="none",
            lw=1,
        )
    )

    # Create inset axes for the zoom
    # The parameters are [x, y, width, height] in relative coordinates of the parent axis
    axins = ax.inset_axes([0.75, -0.2, 0.4, 0.4])

    # Plot the zoomed portion in the inset
    axins.imshow(
        img[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w],
        cmap="gist_gray_r",
        vmin=vmin_img,
        vmax=vmax_img,
        interpolation="none",  # Use 'none' to see individual pixels
    )

    # Customize the inset appearance
    axins.set_xticks([])
    axins.set_yticks([])
    # Add a border to the inset to make it stand out
    for spine in axins.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(1)

# Adjust the shrink parameter to control the colorbar height.
# A value like 0.6 should make it roughly the height of the images.
cbar0 = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.1)
cbar0.set_label("Intensity (a.u.)")
plt.show()
plt.close()

# --- error maps: all share the same norm/colorbar
fig, axes = plt.subplots(1, 6, figsize=(30, 20))
# Plot T1-weighted MR image in the first subplot
ax = axes[0]
im_t1 = ax.imshow(t1_cropped, cmap="gray", vmin=t1_cropped.min(), vmax=t1_cropped.max())
ax.set_title("T1-weighted MR", fontsize=12)
ax.axis("off")
# Add zoom rectangle
ax.add_patch(
    Rectangle((zoom_x, zoom_y), zoom_w, zoom_h, edgecolor="red", facecolor="none", lw=1)
)
# Create and plot inset
axins = ax.inset_axes([0.75, -0.2, 0.4, 0.4])
axins.imshow(
    t1_cropped[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w],
    cmap="gray",
    vmin=t1_cropped.min(),
    vmax=t1_cropped.max(),
    interpolation="none",
)
axins.set_xticks([])
axins.set_yticks([])
for spine in axins.spines.values():
    spine.set_edgecolor("red")
    spine.set_linewidth(1)

# Add a colorbar for the T1 image, making it the same height as the error map colorbar
cbar_t1 = fig.colorbar(im_t1, ax=ax, shrink=0.1)
cbar_t1.ax.set_visible(False)


# Plot the error maps in the remaining subplots
titles = [
    "OSEM",
    "FBSEM",
    "PnP-MM",
    "mr-MAP-EM",
    "mr-FBSEM",
]
# We skip the first error map which is for the reference (all zeros)
# We also skip the first rnmse value which is for the reference
for ax, emap, title, rnmse in zip(
    axes[1:], error_maps_pct[1:], titles, rnmse_values[1:]
):
    im = ax.imshow(emap, cmap="bwr", norm=norm)
    ax.axis("off")

    # Add zoom rectangle
    ax.add_patch(
        Rectangle(
            (zoom_x, zoom_y), zoom_w, zoom_h, edgecolor="red", facecolor="none", lw=1
        )
    )

    # Create and plot inset
    axins = ax.inset_axes([0.75, -0.2, 0.4, 0.4])
    axins.imshow(
        emap[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w],
        cmap="bwr",
        norm=norm,
        interpolation="none",
    )
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(1)

# Add a single colorbar for the error maps
cbar = fig.colorbar(im, ax=axes[1:].ravel().tolist(), shrink=0.1)
cbar.set_label("Error (%)")
plt.show()

# %%

img_idx = 30  # poor mr-FBSEM reconstruction
reference = np.load(
    f"../tests/OSEM/20251124_112527/gt/gt_{str(img_idx).zfill(3)}.npy"
).squeeze()
osem_recon = np.load(
    f"../tests/OSEM/20251124_112527/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
mapem_recon = (
    np.load(f"../tests/MAPEM/20251124_112530/recons/recon_{str(img_idx).zfill(3)}.npy")
    .squeeze()
    .squeeze()
)
mrfbsem_recon = np.load(
    f"../tests/FBSEM-petmr/20251124_112537/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
fbsem_recon = np.load(
    f"../tests/FBSEM-pet/20251124_112659/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
# pnpmm_pet_recon = np.load(
#     f"../tests/PNPMM-pet/20251124_112720/recons/recon_{str(img_idx).zfill(3)}.npy"
# ).squeeze()
pnpmm_pet_recon = np.load(
    f"../tests/PNPMM-pet/20251124_142235/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
data = np.load(
    f"/home/modrzyk/code/FBSEM/MoDL/testFBSEM/brainweb/2D/data-{str(img_idx)}.npy",
    allow_pickle=True,
).item()
t1 = data["mrImg"]
imgs = [
    reference,
    osem_recon,
    fbsem_recon,
    pnpmm_pet_recon,
    mapem_recon,
    mrfbsem_recon,
]
nmse = dinv.metric.NMSE()

rnmse_values = [
    nmse(torch.from_numpy(img).unsqueeze(0), torch.from_numpy(reference).unsqueeze(0))
    .sqrt()
    .item()
    for img in imgs
]


def center_crop(img, crop_size):
    h, w = img.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h : start_h + crop_size, start_w : start_w + crop_size]


crop_size = 100  # adjust as needed
reference_cropped = center_crop(reference, crop_size)
osem_cropped = center_crop(osem_recon, crop_size)
mapem_cropped = center_crop(mapem_recon, crop_size)
mrfbsem_cropped = center_crop(mrfbsem_recon, crop_size)
fbsem_cropped = center_crop(fbsem_recon, crop_size)
pnpmm_pet_cropped = center_crop(pnpmm_pet_recon, crop_size)
t1_cropped = center_crop(t1, crop_size)
images_lc = [
    reference_cropped,
    osem_cropped,
    fbsem_cropped,
    pnpmm_pet_cropped,
    mapem_cropped,
    mrfbsem_cropped,
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
fig, axes = plt.subplots(1, 6, figsize=(30, 20))

# Define the zoom-in rectangle (x, y, width, height) in cropped image coordinates
zoom_x, zoom_y, zoom_w, zoom_h = 58, 37, 27, 27

for ax, img, title, rnmse in zip(
    axes,
    images_lc,
    [
        "Reference OSEM \n(high-dose)",
        "OSEM \n(low-dose)",
        "FBSEM",
        "PnP-MM",
        "mr-MAP-EM",
        "mr-FBSEM",
    ],
    rnmse_values,
):
    im0 = ax.imshow(img, cmap="gist_gray_r", vmin=vmin_img, vmax=vmax_img)
    ax.set_title(f"{title}\nNRMSE: {100*rnmse:.2f}%", fontsize=12)
    ax.axis("off")

    # Create a rectangle patch to indicate the zoom area on the main image
    ax.add_patch(
        Rectangle(
            (zoom_x, zoom_y),
            zoom_w,
            zoom_h,
            edgecolor="red",
            facecolor="none",
            lw=1,
        )
    )

    # Create inset axes for the zoom
    # The parameters are [x, y, width, height] in relative coordinates of the parent axis
    axins = ax.inset_axes([0.75, -0.2, 0.4, 0.4])

    # Plot the zoomed portion in the inset
    axins.imshow(
        img[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w],
        cmap="gist_gray_r",
        vmin=vmin_img,
        vmax=vmax_img,
        interpolation="none",  # Use 'none' to see individual pixels
    )

    # Customize the inset appearance
    axins.set_xticks([])
    axins.set_yticks([])
    # Add a border to the inset to make it stand out
    for spine in axins.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(1)

# Adjust the shrink parameter to control the colorbar height.
# A value like 0.6 should make it roughly the height of the images.
cbar0 = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.2)
cbar0.set_label("Intensity (a.u.)")
plt.show()
plt.close()

# --- error maps: all share the same norm/colorbar
fig, axes = plt.subplots(1, 6, figsize=(30, 20))
# Plot T1-weighted MR image in the first subplot
ax = axes[0]
im_t1 = ax.imshow(t1_cropped, cmap="gray", vmin=t1_cropped.min(), vmax=t1_cropped.max())
ax.set_title("T1-weighted MR", fontsize=12)
ax.axis("off")
# Add zoom rectangle
ax.add_patch(
    Rectangle((zoom_x, zoom_y), zoom_w, zoom_h, edgecolor="red", facecolor="none", lw=1)
)
# Create and plot inset
axins = ax.inset_axes([0.75, -0.2, 0.4, 0.4])
axins.imshow(
    t1_cropped[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w],
    cmap="gray",
    vmin=t1_cropped.min(),
    vmax=t1_cropped.max(),
    interpolation="none",
)
axins.set_xticks([])
axins.set_yticks([])
for spine in axins.spines.values():
    spine.set_edgecolor("red")
    spine.set_linewidth(1)

# Add a colorbar for the T1 image, making it the same height as the error map colorbar
cbar_t1 = fig.colorbar(im_t1, ax=ax, shrink=0.0, anchor=(0, 0.5))
cbar_t1.ax.set_visible(False)


# Plot the error maps in the remaining subplots
titles = [
    "OSEM",
    "FBSEM",
    "PnP-MM",
    "mr-MAP-EM",
    "mr-FBSEM",
]
# We skip the first error map which is for the reference (all zeros)
# We also skip the first rnmse value which is for the reference
for ax, emap, title, rnmse in zip(
    axes[1:], error_maps_pct[1:], titles, rnmse_values[1:]
):
    im = ax.imshow(emap, cmap="bwr", norm=norm)
    ax.axis("off")

    # Add zoom rectangle
    ax.add_patch(
        Rectangle(
            (zoom_x, zoom_y), zoom_w, zoom_h, edgecolor="red", facecolor="none", lw=1
        )
    )

    # Create and plot inset
    axins = ax.inset_axes([0.75, -0.2, 0.4, 0.4])
    axins.imshow(
        emap[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w],
        cmap="bwr",
        norm=norm,
        interpolation="none",
    )
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(1)

# Add a single colorbar for the error maps
cbar = fig.colorbar(im, ax=axes[1:].ravel().tolist(), shrink=0.2)
cbar.set_label("Error (%)")
plt.show()

# %%

img_idx = 350  # good lesion reconstruction
reference = np.load(
    f"../tests/OSEM/20251124_112527/gt/gt_{str(img_idx).zfill(3)}.npy"
).squeeze()
osem_recon = np.load(
    f"../tests/OSEM/20251124_112527/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
mapem_recon = (
    np.load(f"../tests/MAPEM/20251124_112530/recons/recon_{str(img_idx).zfill(3)}.npy")
    .squeeze()
    .squeeze()
)
mrfbsem_recon = np.load(
    f"../tests/FBSEM-petmr/20251124_112537/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
fbsem_recon = np.load(
    f"../tests/FBSEM-pet/20251124_112659/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
# pnpmm_pet_recon = np.load(
#     f"../tests/PNPMM-pet/20251124_112720/recons/recon_{str(img_idx).zfill(3)}.npy"
# ).squeeze()
pnpmm_pet_recon = np.load(
    f"../tests/PNPMM-pet/20251124_142235/recons/recon_{str(img_idx).zfill(3)}.npy"
).squeeze()
data = np.load(
    f"/home/modrzyk/code/FBSEM/MoDL/testFBSEM/brainweb/2D/data-{str(img_idx)}.npy",
    allow_pickle=True,
).item()
t1 = data["mrImg"]
imgs = [
    reference,
    osem_recon,
    fbsem_recon,
    pnpmm_pet_recon,
    mapem_recon,
    mrfbsem_recon,
]
nmse = dinv.metric.NMSE()

rnmse_values = [
    nmse(torch.from_numpy(img).unsqueeze(0), torch.from_numpy(reference).unsqueeze(0))
    .sqrt()
    .item()
    for img in imgs
]


def center_crop(img, crop_size):
    h, w = img.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h : start_h + crop_size, start_w : start_w + crop_size]


crop_size = 100  # adjust as needed
reference_cropped = center_crop(reference, crop_size)
osem_cropped = center_crop(osem_recon, crop_size)
mapem_cropped = center_crop(mapem_recon, crop_size)
mrfbsem_cropped = center_crop(mrfbsem_recon, crop_size)
fbsem_cropped = center_crop(fbsem_recon, crop_size)
pnpmm_pet_cropped = center_crop(pnpmm_pet_recon, crop_size)
t1_cropped = center_crop(t1, crop_size)
images_lc = [
    reference_cropped,
    osem_cropped,
    fbsem_cropped,
    pnpmm_pet_cropped,
    mapem_cropped,
    mrfbsem_cropped,
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
fig, axes = plt.subplots(1, 6, figsize=(30, 20))

# Define the zoom-in rectangle (x, y, width, height) in cropped image coordinates
zoom_x, zoom_y, zoom_w, zoom_h = 55, 35, 27, 27

for ax, img, title, rnmse in zip(
    axes,
    images_lc,
    [
        "Reference OSEM \n(high-dose)",
        "OSEM \n(low-dose)",
        "FBSEM",
        "PnP-MM",
        "mr-MAP-EM",
        "mr-FBSEM",
    ],
    rnmse_values,
):
    im0 = ax.imshow(img, cmap="gist_gray_r", vmin=vmin_img, vmax=vmax_img)
    ax.set_title(f"{title}\nNRMSE: {100*rnmse:.2f}%", fontsize=12)
    ax.axis("off")

    # Create a rectangle patch to indicate the zoom area on the main image
    ax.add_patch(
        Rectangle(
            (zoom_x, zoom_y),
            zoom_w,
            zoom_h,
            edgecolor="red",
            facecolor="none",
            lw=1,
        )
    )

    # Create inset axes for the zoom
    # The parameters are [x, y, width, height] in relative coordinates of the parent axis
    axins = ax.inset_axes([0.75, -0.2, 0.4, 0.4])

    # Plot the zoomed portion in the inset
    axins.imshow(
        img[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w],
        cmap="gist_gray_r",
        vmin=vmin_img,
        vmax=vmax_img,
        interpolation="none",  # Use 'none' to see individual pixels
    )

    # Customize the inset appearance
    axins.set_xticks([])
    axins.set_yticks([])
    # Add a border to the inset to make it stand out
    for spine in axins.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(1)

# Adjust the shrink parameter to control the colorbar height.
# A value like 0.6 should make it roughly the height of the images.
cbar0 = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.2)
cbar0.set_label("Intensity (a.u.)")
plt.show()
plt.close()

# --- error maps: all share the same norm/colorbar
fig, axes = plt.subplots(1, 6, figsize=(30, 20))
# Plot T1-weighted MR image in the first subplot
ax = axes[0]
im_t1 = ax.imshow(t1_cropped, cmap="gray", vmin=t1_cropped.min(), vmax=t1_cropped.max())
ax.set_title("T1-weighted MR", fontsize=12)
ax.axis("off")
# Add zoom rectangle
ax.add_patch(
    Rectangle((zoom_x, zoom_y), zoom_w, zoom_h, edgecolor="red", facecolor="none", lw=1)
)
# Create and plot inset
axins = ax.inset_axes([0.75, -0.2, 0.4, 0.4])
axins.imshow(
    t1_cropped[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w],
    cmap="gray",
    vmin=t1_cropped.min(),
    vmax=t1_cropped.max(),
    interpolation="none",
)
axins.set_xticks([])
axins.set_yticks([])
for spine in axins.spines.values():
    spine.set_edgecolor("red")
    spine.set_linewidth(1)

# Add a colorbar for the T1 image, making it the same height as the error map colorbar
cbar_t1 = fig.colorbar(im_t1, ax=ax, shrink=0.1)
cbar_t1.ax.set_visible(False)


# Plot the error maps in the remaining subplots
titles = [
    "OSEM",
    "FBSEM",
    "PnP-MM",
    "mr-MAP-EM",
    "mr-FBSEM",
]
# We skip the first error map which is for the reference (all zeros)
# We also skip the first rnmse value which is for the reference
for ax, emap, title, rnmse in zip(
    axes[1:], error_maps_pct[1:], titles, rnmse_values[1:]
):
    im = ax.imshow(emap, cmap="bwr", norm=norm)
    ax.axis("off")

    # Add zoom rectangle
    ax.add_patch(
        Rectangle(
            (zoom_x, zoom_y), zoom_w, zoom_h, edgecolor="red", facecolor="none", lw=1
        )
    )

    # Create and plot inset
    axins = ax.inset_axes([0.75, -0.2, 0.4, 0.4])
    axins.imshow(
        emap[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w],
        cmap="bwr",
        norm=norm,
        interpolation="none",
    )
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(1)

# Add a single colorbar for the error maps
cbar = fig.colorbar(im, ax=axes[1:].ravel().tolist(), shrink=0.2)
cbar.set_label("Error (%)")
plt.show()
