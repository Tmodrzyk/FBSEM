# %%
from geometry.BuildGeometry_v4 import BuildGeometry_v4
import numpy as np
import torch
import pathlib
import deepinv as dinv
from phantoms.brainweb import PETbrainWebPhantom

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
PET.loadSystemMatrix(temPath, is3d=False)

img_3d, mumap_3d, t1_3d, _ = PETbrainWebPhantom(
    phanPath,
    phantom_number=1,
    voxel_size=np.array(PET.image.voxelSizeCm) * 10,
    image_size=PET.image.matrixSize,
    pet_lesion=False,
    t1_lesion=False,
)

img_2d = img_3d[:, :, 50]
mumap_2d = mumap_3d[:, :, 50]
t1_3d = t1_3d[:, :, 50]
psf_cm = 0.25

dinv.utils.plot(
    [
        torch.from_numpy(img_2d).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(mumap_2d).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(t1_3d).unsqueeze(0).unsqueeze(0),
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
# 2D OSEM
AN_hd = AF_hd * NF_hd
AN_ld = AF_ld * NF_ld
osem_hd = PET.OSEM2D(y_hd, AN=AN_hd, niter=niter_hd, nsubs=nsubs_hd, psf=psf_hd)
osem_ld = PET.OSEM2D(y_ld, AN=AN_ld, niter=niter_ld, nsubs=nsubs_ld, psf=psf_ld)

# %%
dinv.utils.plot(
    [
        torch.from_numpy(osem_hd).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(osem_ld).unsqueeze(0).unsqueeze(0),
    ],
    figsize=(10, 5),
)
# %%
iter_pnpmm = 20
denoiser = dinv.models.GSDRUNet(
    in_channels=1, out_channels=1, pretrained="download"
).to("cuda")

sigma_denoiser = 25 / 255.0
lambda_reg = 0.01
stepsize = 1e5
pnp_em_ld = PET.PnP_MM2D(
    y_ld,
    AN=AN_ld,
    niter=iter_pnpmm,
    nsubs=1,
    denoiser=denoiser,
    sigma=sigma_denoiser,
    lam=lambda_reg,
    tau=stepsize,
    nonneg=True,
)

dinv.utils.plot(
    [
        torch.from_numpy(osem_hd).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(osem_ld).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(pnp_em_ld).unsqueeze(0).unsqueeze(0),
    ],
    figsize=(10, 5),
)
# %%
