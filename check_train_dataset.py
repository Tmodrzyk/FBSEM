# %%
"""
Created on July 2020
Demo for traning a 2D FBSEM net


@author: Abi Mehranian
abolfazl.mehranian@kcl.ac.uk
"""

import numpy as np
from matplotlib import pyplot as plt
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from models.deeplib import buildBrainPhantomDataset
import deepinv as dinv
import torch

# build PET recontruction object
temPath = r"./tmp/"
PET = BuildGeometry_v4("mmr", 0.5)  # scanner mmr, with radial crop factor of 50%
PET.loadSystemMatrix(temPath, is3d=False)

# get some info of Pet object
print("is3d:", PET.is3d)
print("\nscanner info:", PET.scanner.as_dict())
print("\nimage info:", PET.image.as_dict())
print("\nsinogram info:", PET.sinogram.as_dict())


# this will take hours (5 phantoms, 5 random rotations each, lesion & sinogram simulation, 3 different recon,...)
# see 'buildBrainPhantomDataset' for default values, e.g. count level, psf, no. lesions, lesion size, no. rotations, rotation range,....
# LD/ld stands for low-definition low-dose, HD/hd stands for high-definition high-dose

save_training_dir = r"./MoDL/trainingDatasets/brainweb/2D"

# check out the strcuture of the produced datasets, e.g. data-0.npy
d = np.load(save_training_dir + "/" + "data-0.npy", allow_pickle=True).item()
d.keys()

dinv.utils.plot(
    [
        torch.from_numpy(d["mrImg"]).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(d["imgHD"]).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(d["imgLD"]).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(d["imgLD_psf"]).unsqueeze(0).unsqueeze(0),
    ],
    figsize=(20, 10),
)

fig, ax = plt.subplots(1, 4, figsize=(20, 10))

ax[0].imshow(d["mrImg"], cmap="gist_gray"), ax[0].set_title("mrImg", fontsize=20)
ax[1].imshow(d["imgHD"], cmap="gist_gray_r"), ax[1].set_title("imgHD", fontsize=20)
ax[2].imshow(d["imgLD"], cmap="gist_gray_r"), ax[2].set_title("imgLD", fontsize=20)
(
    ax[3].imshow(d["imgLD_psf"], cmap="gist_gray_r"),
    ax[3].set_title("imgLD_psf", fontsize=20),
)


fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(d["sinoLD"]), ax[0].set_title("sinoLD", fontsize=20)
(
    ax[1].imshow(d["AN"]),
    ax[1].set_title("Atten. factors * Norm. Factors (AN)", fontsize=20),
)

# %%
