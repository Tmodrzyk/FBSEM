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
# 2D OSEM
AN_hd = AF_hd * NF_hd
AN_ld = AF_ld * NF_ld
osem_hd = PET.OSEM2D(y_hd, AN=AN_hd, niter=niter_hd, nsubs=nsubs_hd, psf=psf_hd)
osem_ld = PET.OSEM2D(y_ld, AN=AN_ld, niter=niter_ld, nsubs=nsubs_ld, psf=psf_ld)

map_em_ld = PET.MAPEM2D(y_ld, AN=AN_ld, beta=0.06, niter=10, nsubs=6, psf=psf_ld)
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
iter_pnpmm = 60
# pretrained_path = pathlib.Path("./weights/25-10-13-14:12:28/ckp_best.pth.tar")
denoiser = dinv.models.GSDRUNet(
    in_channels=1, out_channels=1, pretrained="download"
).to("cuda")


sigma_denoiser_ld = 16 / 255.0
lambda_reg_ld = 0.1
stepsize_ld = 1e2

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

mse = dinv.metric.MSE()
rnmse_pnpmm = [
    (
        torch.sqrt(
            mse(
                torch.from_numpy(center_crop(x, crop_size)).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(osem_hd_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
        / torch.norm(torch.from_numpy(osem_hd_cropped).unsqueeze(0).unsqueeze(0)).sqrt()
    )
    * 100
    for x in xs
]

plt.plot(rnmse_pnpmm)
print(
    "NMSE OSEM LD: ",
    (
        torch.sqrt(
            mse(
                torch.from_numpy(osem_ld_cropped).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(osem_hd_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
        / torch.norm(torch.from_numpy(osem_hd_cropped).unsqueeze(0).unsqueeze(0)).sqrt()
    ).item()
    * 100,
)
print(
    "NMSE FBSEM LD: ",
    (
        torch.sqrt(
            mse(
                torch.from_numpy(fbsem_ld_cropped).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(osem_hd_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
        / torch.norm(torch.from_numpy(osem_hd_cropped).unsqueeze(0).unsqueeze(0)).sqrt()
    ).item()
    * 100,
)
print(
    "NMSE MAP EM LD: ",
    (
        torch.sqrt(
            mse(
                torch.from_numpy(map_em_ld_cropped).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(osem_hd_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
        / torch.norm(torch.from_numpy(osem_hd_cropped).unsqueeze(0).unsqueeze(0)).sqrt()
    ).item()
    * 100,
)
print(
    "NMSE PnP MM LD: ",
    (
        torch.sqrt(
            mse(
                torch.from_numpy(pnp_mm_ld_cropped).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(osem_hd_cropped).unsqueeze(0).unsqueeze(0),
            )
        )
        / torch.norm(torch.from_numpy(osem_hd_cropped).unsqueeze(0).unsqueeze(0)).sqrt()
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
