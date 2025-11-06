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
pretrained_path = pathlib.Path(
    "/home/modrzyk/code/FBSEM/weights/GSDRUNet-brainweb/25-11-05-15:09:25/ckp_best.pth.tar"
)
denoiser = dinv.models.GSDRUNet(
    in_channels=1, out_channels=1, pretrained=pretrained_path
).to("cuda")


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
    import numpy as np

    # --- inputs assumed: PET, prompts, AN, RS, img (optional), niter, nsubs, tau, psf, s (optional)
    # SensImg is built below; if you prefer a custom s, pass it and skip SensImg in the update.

    tic = time.time()
    [numAng, subSize] = PET.angular_subsets(nsubs)

    # --- Batch handling (fix: only add ONE batch dimension if 2D)
    if np.ndim(prompts) == 2:  # (R, A) -> (1, R, A)
        batch_size = 1
        prompts = prompts[None, :, :]
    else:
        batch_size = prompts.shape[0]

    # image buffer
    if img is None:
        img = np.ones([batch_size, np.prod(PET.image.matrixSize[:2])], dtype=float)
    else:
        if batch_size > 1 and img.shape[0] != batch_size:
            raise ValueError("1st img dimension doesn't match batch_size")
    nVoxls = np.prod(PET.image.matrixSize[:2])
    img = np.reshape(img, [batch_size, nVoxls], order="F")

    # RS / AN shapes
    if RS is None:
        RS = np.zeros_like(prompts, dtype=float)
    if AN is None:
        AN = np.ones(
            [batch_size, PET.sinogram.nRadialBins, PET.sinogram.nAngularBins],
            dtype=float,
        )
    elif np.ndim(AN) == 2:
        AN = AN[None, :, :]

    # Sensitivity images (per subset). Use this in the sqrt step (as "s").
    SensImg = PET.SensImageBatch2D(AN, nsubs, psf)  # (B, nsubs, Nvox)
    if SensImg.ndim == 2:
        SensImg = SensImg[None, :, :]
    matrixSize = PET.image.matrixSize
    q = PET.sinogram.nAngularBins // 2

    # Per-subset stepsize (optional but common)
    # tau_sub = tau / float(nsubs)
    tau_sub = tau

    eps_div = 1e-5
    eps_sqrt = 0.0  # bump slightly if roundoff causes tiny negatives
    xs = []  # store intermediate images

    for n in range(niter):
        print(f"Iter {n+1}/{niter}", end="\r")

        for sub in range(nsubs):

            # Pre-blur image for the forward model A(H x)
            img_ = img.copy()
            if np.any(psf != 0):
                for b in range(batch_size):
                    img_[b, :] = PET.gaussFilter(img_[b, :], psf)

            # Ratio backprojection accumulator: A_S^T( AN * y / (AN*A(Hx) + RS) )
            backProjImage = np.zeros_like(img_)  # (B, Nvox)
            # Iterate over angles in this subset
            for ii in range(subSize // 2):
                i = numAng[ii, sub]

                # Radial bins
                for j in range(PET.sinogram.nRadialBins):
                    M0 = PET.geoMatrix[0][i, j]
                    if np.isscalar(M0):
                        continue

                    # Ensure proper 2D array: each row = [x, y, ?, weight]
                    A = np.asarray(M0)
                    if A.ndim != 2 or A.shape[1] < 4:
                        raise ValueError(
                            f"Malformed geoMatrix entry at angle={i}, bin={j}: shape={A.shape}"
                        )

                    voxel_coords = A[:, 0:3].astype(np.int32)  # (nseg, 3)
                    geom_weights = A[:, 3].astype(np.float64) / 1e4  # (nseg,)

                    # Flattened voxel indices for (i, j) and its symmetric partner (i+q, j)
                    x, y = voxel_coords[:, 0], voxel_coords[:, 1]
                    idx1 = x + y * matrixSize[0]
                    idx2 = y + matrixSize[0] * (matrixSize[0] - 1 - x)

                    # Strict length consistency — if these fail, geometry is mismatched upstream
                    nseg = geom_weights.shape[0]
                    if idx1.shape[0] != nseg or idx2.shape[0] != nseg:
                        raise ValueError(
                            f"geo mismatch at angle={i}, bin={j}: "
                            f"len(weights)={nseg}, len(idx1)={idx1.shape[0]}, len(idx2)={idx2.shape[0]}"
                        )

                    for b in range(batch_size):
                        # forward predictions
                        fwd1 = geom_weights.dot(img_[b, idx1])
                        fwd2 = geom_weights.dot(img_[b, idx2])

                        den1 = AN[b, j, i] * fwd1 + RS[b, j, i] + eps_div
                        den2 = AN[b, j, i + q] * fwd2 + RS[b, j, i + q] + eps_div

                        ratio1 = prompts[b, j, i] / den1
                        ratio2 = prompts[b, j, i + q] / den2

                        # Accumulate with np.add.at (robust to duplicate indices)
                        np.add.at(
                            backProjImage[b], idx1, geom_weights * AN[b, j, i] * ratio1
                        )
                        np.add.at(
                            backProjImage[b],
                            idx2,
                            geom_weights * AN[b, j, i + q] * ratio2,
                        )

            # If SensImageBatch2D already includes PSF adjoint, leave the next block off.
            # If it doesn't, uncomment to keep projector/backprojector matched (Gaussian ~ self-adjoint).
            if np.any(psf != 0):
                for b in range(batch_size):
                    backProjImage[b, :] = PET.gaussFilter(backProjImage[b, :], psf)

            # Half-step: x^(n+1/2) = x^n * A_S^T( AN * y / (AN*A(Hx^n) + RS) )
            x_half = img * backProjImage

            # Quadratic/sqrt step:
            # x^{n+1} = 0.5 * [ (x^n - tau_sub * s) + sqrt( (x^n - tau_sub * s)^2 + 4 * tau_sub * x_half ) ]
            # Here we use s = SensImg[:, sub, :] (shape (B, Nvox)), as per your request.
            s_vec = SensImg[:, sub, :]  # per-subset sensitivity image
            tmp = img - tau_sub * s_vec
            img = 0.5 * (tmp + np.sqrt(tmp * tmp + 4.0 * tau_sub * x_half + eps_sqrt))

            xs.append(np.reshape(img[0, :], matrixSize[:2], order="F"))
            dinv.utils.plot(torch.from_numpy(xs[-1]).unsqueeze(0).unsqueeze(0))
    # Reshape back to (B, H, W)
    img = np.reshape(img, [batch_size, matrixSize[0], matrixSize[1]], order="F")
    if batch_size == 1:
        img = img[0, :, :]

    print(f"{batch_size} batches reconstructed in: {(time.time()-tic):.3f} sec.")
    return img, xs


psf_ld = 0.4
nsubs_ld = 14
iter_pnpmm = 15
sigma_denoiser_ld = 10
lambda_reg_ld = 0.0
stepsize_ld = 10

pnp_mm_ld, xs = PnP_MM2D(
    PET,
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
