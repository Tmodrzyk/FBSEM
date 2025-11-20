import numpy as np
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from models.deeplib import PETMrDataset, dotstruct, toNumpy, crop
from models.modellib import FBSEMnet_v3, Trainer, fbsemInference
import os
import torch

save_training_dir = r"./MoDL/trainFBSEM/brainweb/2D"

g = dotstruct()
g.is3d = False
g.temPath = r"./tmp/"
g.radialBinCropFactor = 0.5
g.psf_cm = 0.15
g.niters = 8
g.nsubs = 6
g.training_flname = [save_training_dir + os.sep, "data-"]
g.save_dir = r"./MoDL/outputFBSEM/brainweb/2D" + os.sep
g.device = "cpu"
g.num_workers = 0
g.batch_size = 5
g.test_size = 0.2
g.valid_size = 0.1
g.num_train = 500
g.num_kernels = 32
g.kernel_size = 3
g.depth = 5
g.in_channels = 1
g.reg_ccn_model = "resUnit"
g.lr = 0.01
g.epochs = 100
g.model_name = "fbsem-pm-03"
g.display = True
g.disp_figsize = (20, 10)
g.save_from_epoch = 0
g.crop_factor = 0.3
g.do_validation = True


# build PET object
PET = BuildGeometry_v4("mmr", g.radialBinCropFactor)
PET.loadSystemMatrix(g.temPath, is3d=False)

# load dataloaders
train_loader, valid_loader, test_loader = PETMrDataset(
    g.training_flname,
    num_train=g.num_train,
    is3d=g.is3d,
    batch_size=g.batch_size,
    test_size=g.test_size,
    valid_size=g.valid_size,
    num_workers=g.num_workers,
)

# build model
model = FBSEMnet_v3(
    g.depth, g.num_kernels, g.kernel_size, g.in_channels, g.is3d, g.reg_ccn_model
).to(g.device, dtype=torch.float32)

# train
Trainer(PET, model, g, train_loader, valid_loader)
