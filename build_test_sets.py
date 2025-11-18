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


# build PET recontruction object
temPath = r"./tmp/"
PET = BuildGeometry_v4("mmr", 0.5)  # scanner mmr, with radial crop factor of 50%
PET.loadSystemMatrix(temPath, is3d=False)

# get some info of Pet object
print("is3d:", PET.is3d)
print("\nscanner info:", PET.scanner.as_dict())
print("\nimage info:", PET.image.as_dict())
print("\nsinogram info:", PET.sinogram.as_dict())


# Should produce 100 test slices (5 phantoms, 4 random rotations each, 5 slices each,...)

phanPath = r"./phantoms/Brainweb"
save_training_dir = r"./MoDL/testFBSEM/brainweb/2D"
phanType = "brainweb"
phanNumber = np.arange(9, 19, 1)  # Use 10 last brainweb phantom

buildBrainPhantomDataset(
    PET,
    save_training_dir,
    phanPath,
    phanType=phanType,
    phanNumber=phanNumber,
    is3d=False,
    num_rand_rotations=5,
)
