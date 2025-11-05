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
data_path = pathlib.Path(r"./MoDL/train/brainweb/2D/data-245.npy")
data = np.load(data_path, allow_pickle=True).item()

dinv.utils.plot(
    [
        torch.from_numpy(data["imgHD"]).unsqueeze(0).unsqueeze(0),
    ],
    figsize=(5, 5),
)
print(data["imgHD"].min(), data["imgHD"].max())
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
iter_pnpmm = 50
# pretrained_path = pathlib.Path("./weights/25-10-13-14:12:28/ckp_best.pth.tar")
denoiser = dinv.models.GSDRUNet(
    in_channels=1, out_channels=1, pretrained="download"
).to("cuda")

sigma_denoiser_ld = 5 / 255.0
lambda_reg_ld = 0.3
stepsize_ld = 1e3


def normalize_osem_image(x_osem, quantile=0.01):
    """
    PET-specific sampling normalization using OSEM reconstruction.
    Computes c_OSEM = sum(x_osem) / count(x_osem > Q_0.01)
    and scales the image accordingly.

    Args:
        x_osem (torch.Tensor): reconstructed image, shape (B, C, H, W) or (B, 1, D, H, W)
        quantile (float): quantile used for thresholding (default 0.01 → 1%)

    Returns:
        x_norm (torch.Tensor): normalized image
        c_osem (torch.Tensor): normalization factor per image
    """
    B = x_osem.shape[0]
    spatial_dims = list(range(1, x_osem.ndim))

    c_osem = []
    x_norm = torch.empty_like(x_osem)

    for b in range(B):
        x = x_osem[b]
        q = torch.quantile(x, quantile)  # compute Q_0.01
        mask = x > q
        numerator = x.sum()
        denominator = mask.sum()
        c = numerator / (denominator + 1e-8)
        x_norm[b] = x / (c + 1e-8)
        c_osem.append(c)

    c_osem = torch.stack(c_osem).view(B, *([1] * (x_osem.ndim - 1)))
    return x_norm, c_osem


def PnP_MM2D(
    PET,
    prompts,
    img=None,
    RS=None,
    AN=None,
    iSensImg=None,
    niter=20,
    nsubs=1,
    psf=0,
    denoiser=None,
    sigma=0.1,
    lambda_reg=1.0,
    tau=1e2,
):
    """
    Plug-and-Play Majorize-Minimize algorithm for 2D PET image reconstruction.
    This function performs iterative PET image reconstruction using a Plug-and-Play
    denoising approach combined with a Majorize-Minimize optimization scheme. The
    algorithm alternates between denoising steps and data consistency updates.
    Only works for one subset.
    Args:
        PET: PET system object containing geometry and reconstruction methods
        prompts (np.ndarray): Prompt sinogram data. Shape can be (nr, na) or (batch, nr, na)
        img (np.ndarray, optional): Initial image estimate. If None, initializes to ones.
            Shape can be (H, W) or (batch, H, W).
        RS (np.ndarray, optional): Random/scatter sinogram. If None, initializes to zeros.
            Shape (batch, nr, na).
        AN (np.ndarray, optional): Attenuation/normalization factors. If None, initializes to ones.
            Shape can be (nr, na) or (batch, nr, na).
        iSensImg (np.ndarray, optional): Inverse sensitivity images for subsets.
            Shape (batch, nsubs, nvox). If None, computed automatically.
        niter (int, optional): Number of iterations.
        nsubs (int, optional): Number of subsets for ordered subsets acceleration. Defaults to 1.
        psf (float, optional): Point spread function parameter.
        denoiser (callable, optional): Denoising function that takes (image, sigma) as input.
            If None, no denoising is applied.
        sigma (float, optional): Noise level parameter for denoiser.
        lambda_reg (float, optional): Regularization parameter balancing denoising and data fidelity.
            Should be in [0, 1].
        tau (float, optional): Step size parameter for the majorize-minimize update.
    Returns:
        tuple: A tuple containing:
            - out (np.ndarray): Reconstructed image(s). Shape (H, W) for single image or
              (batch, H, W) for batch reconstruction.
            - xs (list): List of intermediate image estimates at each subset iteration.
              Each element has shape (H, W).
    Notes:
        - The denoiser function is expected to work with normalized images and return
          denoised images in the same format.
        - Images are cropped to 128x128 for denoising and then placed back into original size.
        - The algorithm uses a forward-divide-backward projection method for data consistency.
        - Automatic batching support: single images are automatically batched and unbatched.
    """
    import time

    xs = []
    tic = time.time()

    # --- Shapes / batching ---
    if np.ndim(prompts) == 2:  # (nr, na) -> add batch dim
        prompts = prompts[None, :, :]

    batch_size = prompts.shape[0]
    H, W = PET.image.matrixSize[:2]
    nvox = H * W

    # init image
    if img is None:
        x = np.ones((batch_size, nvox), dtype=float)
    else:
        if np.ndim(img) == 2:
            img = img[None, :, :]
        x = img.reshape((batch_size, nvox), order="F").astype(float)

    # defaults for AN/RS
    if RS is None:
        dims = (batch_size, PET.sinogram.nRadialBins, PET.sinogram.nAngularBins)
        RS = np.zeros(dims, dtype=float)

    if AN is None:
        AN = np.ones(
            (batch_size, PET.sinogram.nRadialBins, PET.sinogram.nAngularBins),
            dtype=float,
        )
    elif np.ndim(AN) == 2:
        AN = AN[None, :, :]

    # per-subset inverse sensitivity:  iSensImg[b, sub, nvox] = 1 / s_sub
    if iSensImg is None:
        iSensImg = PET.iSensImageBatch2D(AN=AN, nsubs=nsubs, psf=psf)
    if np.ndim(iSensImg) == 2:
        iSensImg = iSensImg[None, :, :]

    eps = 1e-24

    # denoiser wrapper (identity if None)
    def _denoise_batch(x_vec):
        # x_vec: (batch, nvox) -> returns (batch, nvox)
        x_imgs = x_vec.reshape((batch_size, H, W), order="F")
        x_imgs = torch.from_numpy(x_imgs).unsqueeze(1).to("cuda")
        # Crop to 128x128 for denoising
        h_orig, w_orig = x_imgs.shape[-2:]
        crop_h, crop_w = 128, 128
        start_h = (h_orig - crop_h) // 2
        start_w = (w_orig - crop_w) // 2
        x_imgs_cropped = x_imgs[
            :, :, start_h : start_h + crop_h, start_w : start_w + crop_w
        ]

        # Normalize the cropped image
        x_imgs_cropped_norm, c_osem = normalize_osem_image(x_imgs_cropped)
        dinv.utils.plot(x_imgs_cropped_norm, cmap="gist_gray_r", show=True)

        # Denoise the cropped image
        x_hat_cropped = denoiser(x_imgs_cropped_norm, sigma)
        x_hat_cropped = torch.clamp(x_hat_cropped, min=eps)

        # Denormalize the cropped denoised image
        x_hat_cropped = (
            x_hat_cropped * (x_imgs_cropped.max() - x_imgs_cropped.min())
            + x_imgs_cropped.min()
        )

        # Put the denoised crop back into the original size
        x_hat = x_imgs.clone()
        x_hat[:, :, start_h : start_h + crop_h, start_w : start_w + crop_w] = (
            x_hat_cropped
        )
        x_imgs_norm = (x_imgs - x_imgs.min()) / (x_imgs.max() - x_imgs.min() + 1e-8)

        # support both batch-aware and per-image denoisers
        x_hat = denoiser(x_imgs_norm, sigma)
        x_hat = torch.clamp(x_hat, min=eps)  # clamp to non-negative values

        # Denormalize the denoised image
        # x_hat *= c_osem
        return x_hat.reshape((batch_size, nvox)).detach().cpu().numpy()

    for it in range(niter):

        if it > 0:
            x_deno = lambda_reg * _denoise_batch(x) + (1 - lambda_reg) * x_cur
            x_cur = x_deno.copy()
        else:
            x_cur = x.copy()
        for sub in range(nsubs):
            back = PET.forwardDivideBackwardBatch2D(
                imgb=x_cur.reshape((batch_size, H, W), order="F"),
                prompts=prompts,
                RS=RS,
                AN=AN,
                nsubs=nsubs,
                subset_i=sub,
                tof=False,
                psf=psf,
            )

            inv_s = iSensImg[:, sub, :]  # = 1 / s_sub
            s_sub = 1.0 / np.maximum(inv_s, eps)

            a = x_cur - tau * s_sub
            x_next = 0.5 * (a + np.sqrt((a * a) + 4.0 * tau * x_cur * back))

            x_cur = x_next

            xs.append(x_cur.reshape((H, W), order="F").copy())
        x = x_cur

    # --- reshape back to images ---
    out = x.reshape((batch_size, H, W), order="F")
    if batch_size == 1:
        out = out[0]
    # print(
    #     f"{batch_size} batches reconstructed with PnP_EM2D in {time.time()-tic:.3f} s."
    # )
    return out, xs


pnp_mm_ld, xs = PnP_MM2D(
    PET,
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
