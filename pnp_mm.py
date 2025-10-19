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
data_path = pathlib.Path(r"./MoDL/trainingDatasets/brainweb/2D/data-0.npy")
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
# 2D OSEM
AN_hd = AF_hd * NF_hd
AN_ld = AF_ld * NF_ld
osem_hd = PET.OSEM2D(y_hd, AN=AN_hd, niter=niter_hd, nsubs=nsubs_hd, psf=0)
osem_ld = PET.OSEM2D(y_ld, AN=AN_ld, niter=niter_ld, nsubs=nsubs_ld, psf=0)

map_em_ld = PET.MAPEM2D(y_ld, AN=AN_ld, beta=0.06, niter=10, nsubs=6)
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
    denoiser=None,  # callable: denoiser(img2d, sigma) -> img2d  (or batch-aware)
    sigma=0.1,
    lambda_reg=1.0,  # \lambda in your equations
    tau=1e-2,  # \tau in your equations
    nonneg=True,
):
    """
    Plug-and-Play EM-like 2D algorithm implementing:
        x^{n+1/3} = (1 - lam*tau) * x^n + lam*tau * D_sigma(x^n)
        x^{n+2/3} = (x^{n+1/3} / s) * A^T [ y / (A x^{n+1/3} + R) ]
        x^{n+1}   = 0.5 * [ x^{n+1/3} - tau*s + sqrt( (x^{n+1/3} - tau*s)^2 + 4*tau*s*x^{n+2/3} ) ]

    where s is the (subset) sensitivity image P_s^T 1 (with AN/TOF/PSF handled consistently).
    The EM ratio/backprojection is realized with self.forwardDivideBackwardBatch2D(...).
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
        x_imgs_cropped_norm = (x_imgs_cropped - x_imgs_cropped.min()) / (
            x_imgs_cropped.max() - x_imgs_cropped.min() + 1e-8
        )

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
        x_hat = x_hat * (x_imgs.max() - x_imgs.min()) + x_imgs.min()
        x_hat = x_hat.squeeze(1).cpu().detach().numpy()
        return x_hat.reshape((batch_size, nvox), order="F")

    for it in range(niter):
        # # 1) denoiser-relaxation

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

            if nonneg:
                x_next = np.maximum(x_next, 0.0)

            x_cur = x_next

            xs.append(x_cur.reshape((H, W), order="F").copy())
        x = x_cur

    # --- reshape back to images ---
    out = x.reshape((batch_size, H, W), order="F")
    if batch_size == 1:
        out = out[0]
    print(
        f"{batch_size} batches reconstructed with PnP_EM2D in {time.time()-tic:.3f} s."
    )
    return out, xs


# %%
iter_pnpmm = 40
pretrained_path = pathlib.Path("./weights/25-10-13-14:12:28/ckp_best.pth.tar")
denoiser = dinv.models.GSDRUNet(
    in_channels=1, out_channels=1, pretrained=pretrained_path
).to("cuda")


sigma_denoiser_ld = 20 / 255.0
lambda_reg_ld = 0.1
stepsize_ld = 1e9

pnp_mm_ld, xs = PnP_MM2D(
    PET,
    y_ld,
    AN=AN_ld,
    niter=iter_pnpmm,
    nsubs=1,
    denoiser=denoiser,
    sigma=sigma_denoiser_ld,
    lambda_reg=lambda_reg_ld,
    tau=stepsize_ld,
    nonneg=True,
)


# Center crop the images first
def center_crop(img, crop_size):
    h, w = img.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h : start_h + crop_size, start_w : start_w + crop_size]


crop_size = 128  # adjust as needed
osem_ld_cropped = center_crop(osem_ld, crop_size)
pnp_mm_ld_cropped = center_crop(pnp_mm_ld, crop_size)
osem_hd_cropped = center_crop(osem_hd, crop_size)
map_em_ld_cropped = center_crop(map_em_ld, crop_size)
fbsem_ld_cropped = center_crop(fbsem_ld, crop_size)

fig, axes = plt.subplots(1, 5, figsize=(20, 10))
images_lc = [
    osem_ld_cropped,
    map_em_ld_cropped,
    fbsem_ld_cropped,
    pnp_mm_ld_cropped,
    osem_hd_cropped,
]
titles_lc = [
    "OSEM\n (low-dose)",
    "MAP-EM \n (low-dose)",
    "FBSEM \n (low-dose)",
    "PnP-MM \n (low-dose)",
    "Reference OSEM \n (high-dose)",
]

for ax, img, title in zip(axes.flatten(), images_lc, titles_lc):
    im = ax.imshow(img, cmap="gist_gray_r")
    ax.set_title(title, fontsize=15)
    ax.axis("off")

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.show()

nmse = dinv.metric.NMSE()
nmse_pnpmm = [
    nmse(
        torch.from_numpy(center_crop(x, crop_size)).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(osem_hd_cropped).unsqueeze(0).unsqueeze(0),
    )
    for x in xs
]

plt.plot(nmse_pnpmm)

print(
    "NMSE OSEM LD: ",
    nmse(
        torch.from_numpy(osem_ld).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(osem_hd).unsqueeze(0).unsqueeze(0),
    ).item(),
)
print(
    "NMSE FBSEM LD: ",
    nmse(
        torch.from_numpy(fbsem_ld).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(osem_hd).unsqueeze(0).unsqueeze(0),
    ).item(),
)
print(
    "NMSE MAP EM LD: ",
    nmse(
        torch.from_numpy(map_em_ld).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(osem_hd).unsqueeze(0).unsqueeze(0),
    ).item(),
)
print(
    "NMSE PnP MM LD: ",
    nmse(
        torch.from_numpy(pnp_mm_ld).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(osem_hd).unsqueeze(0).unsqueeze(0),
    ).item(),
)

print("Max values of reconstructions:")
print(f"OSEM LD max: {osem_ld.max():.6f}")
print(f"MAP-EM LD max: {map_em_ld.max():.6f}")
print(f"FBSEM LD max: {fbsem_ld.max():.6f}")
print(f"PnP-MM LD max: {pnp_mm_ld.max():.6f}")
print(f"OSEM HD max: {osem_hd.max():.6f}")

# %%
xs = [torch.from_numpy(x).unsqueeze(0).unsqueeze(0) for x in xs]
# %%
dinv.utils.plot(xs[5:], figsize=(20, 10), cmap="gist_gray_r")

# %%
