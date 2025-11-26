# %%
import numpy as np
import torch
import deepinv as dinv
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.rcParams["backend"] = "Qt5Agg"

from matplotlib import colors
from pathlib import Path
import seaborn as sns

sns.set_theme("notebook")
gt_dir = Path("../MoDL/testFBSEM/brainweb/2D/")
osem_dir = Path("../tests/OSEM/20251124_112527/recons/")
mapem_dir = Path("../tests/MAPEM/20251124_112530/recons/")
mrfbsem_dir = Path("../tests/FBSEM-petmr/20251124_112537/recons/")
fbsem_dir = Path("../tests/FBSEM-pet/20251124_112659/recons/")
# pnpmm_pet_dir = Path("../tests/PNPMM-pet/20251124_112720/recons/")
pnpmm_pet_dir = Path("../tests/PNPMM-pet/20251124_142235/recons/")


num_imgs = 500
nmse = dinv.metric.NMSE()

recon_dirs = {
    "OSEM": osem_dir,
    "FBSEM": fbsem_dir,
    "PnP-MM": pnpmm_pet_dir,
    "mr-MAP-EM": mapem_dir,
    "mr-FBSEM": mrfbsem_dir,
}
# Store results for plotting
results = {}

for name, recon_dir in recon_dirs.items():
    print(f"Processing {name}...")
    results[name] = {}
    for mask_name, mask_values in {
        "Complete": None,
        "White Matter": [32.0],
        "Grey Matter": [96.0],
        "Hot Lesions": [144.0],
    }.items():
        nrmse_scores = []
        for idx in range(num_imgs):
            gt_filename = f"data-{idx}.npy"
            gt_path = gt_dir / gt_filename
            recon_filename = f"recon_{idx:03d}.npy"
            recon_path = recon_dir / recon_filename

            if not recon_path.exists():
                continue

            data = np.load(gt_path, allow_pickle=True).item()
            gt_img = torch.from_numpy(data["imgHD"]).unsqueeze(0).unsqueeze(0)
            recon_img = torch.from_numpy(np.load(recon_path))
            if recon_img.ndim == 3:
                recon_img = recon_img.unsqueeze(0)

            if mask_values is None:  # Complete image case
                nmse_val = nmse(gt_img, recon_img)
                nrmse_scores.append(torch.sqrt(nmse_val).item())
            else:  # Masked case
                mask = np.isin(data["imgGT"], mask_values)
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

                if torch.any(mask_tensor):
                    gt_masked = gt_img[mask_tensor]
                    recon_masked = recon_img[mask_tensor]

                    if gt_masked.ndim == 0:
                        gt_masked = gt_masked.unsqueeze(0)
                        recon_masked = recon_masked.unsqueeze(0)

                    # Ensure tensors are not empty before calculating NMSE
                    if gt_masked.numel() > 0:
                        nmse_val = nmse(recon_masked, gt_masked)
                        nrmse_scores.append(torch.sqrt(nmse_val).item())

        if nrmse_scores:
            mean_nrmse = np.mean(nrmse_scores)
            std_nrmse = np.std(nrmse_scores)
            results[name][mask_name] = (mean_nrmse, std_nrmse)
            print(
                f"  {mask_name}: Mean NRMSE: {mean_nrmse * 100:.2f} ± {std_nrmse * 100:.2f}"
            )
        else:
            results[name][mask_name] = (0, 0)
    print()

# %%
# Plotting the results
methods = list(results.keys())
mask_names = list(results[methods[0]].keys())
x = np.arange(len(mask_names))  # the label locations for regions
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout="constrained", figsize=(12, 7))
colors = sns.color_palette()

for i, method in enumerate(methods):
    means = [results[method][mask_name][0] * 100 for mask_name in mask_names]
    stds = [results[method][mask_name][1] * 100 for mask_name in mask_names]
    offset = width * multiplier
    rects = ax.bar(
        x + offset,
        means,
        width,
        yerr=stds,
        label=method,
        capsize=5,
        color=colors[i],
    )
    multiplier += 1

# Add some text for labels, title and axes ticks
fontsize = 25
ax.set_ylabel("NRMSE (%)", fontsize=fontsize)
ax.set_xticks(x + width * (len(methods) - 1) / 2, mask_names, fontsize=fontsize)
ax.legend(loc="upper left", ncols=1, fontsize=20)
ax.set_ylim(0)
ax.grid(axis="y", linestyle="--", alpha=0.7)
fig.savefig("figure2.pdf", format="pdf")

plt.show()

# %%
