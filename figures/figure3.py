# %%
# %%
# Calculate Contrast-to-Noise Ratio (CNR)
# CNR is defined as (mean(GM) - mean(WM)) / std(WM)
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
pnpmm_pet_dir = Path("../tests/PNPMM-pet/20251124_112720/recons/")

num_imgs = 500
nmse = dinv.metric.NMSE()
recon_dirs = {
    "OSEM": osem_dir,
    "FBSEM": fbsem_dir,
    "PnP-MM": pnpmm_pet_dir,
    "mr-MAP-EM": mapem_dir,
    "mr-FBSEM": mrfbsem_dir,
}
cnr_results = {}
cnr_lesion_results = {}
wm_values = [32.0]
gm_values = [96.0]
lesion_values = [144.0]  # Assuming 160.0 is the value for hot lesions

for name, recon_dir in recon_dirs.items():
    print(f"Calculating CNR for {name}...")
    cnr_scores = []
    cnr_lesion_scores = []
    for idx in range(num_imgs):
        gt_filename = f"data-{idx}.npy"
        gt_path = gt_dir / gt_filename
        recon_filename = f"recon_{idx:03d}.npy"
        recon_path = recon_dir / recon_filename

        if not recon_path.exists():
            continue

        data = np.load(gt_path, allow_pickle=True).item()
        recon_img = torch.from_numpy(np.load(recon_path))
        if recon_img.ndim == 3:
            recon_img = recon_img.unsqueeze(0)
        # Create masks for White Matter (WM), Grey Matter (GM), and Lesions
        wm_mask = (
            torch.from_numpy(np.isin(data["imgGT"], wm_values))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        gm_mask = (
            torch.from_numpy(np.isin(data["imgGT"], gm_values))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        lesion_mask = (
            torch.from_numpy(np.isin(data["imgGT"], lesion_values))
            .unsqueeze(0)
            .unsqueeze(0)
        )

        if torch.any(wm_mask) and torch.any(gm_mask):
            recon_wm = recon_img[wm_mask]
            recon_gm = recon_img[gm_mask]

            mean_wm = torch.mean(recon_wm.float())
            std_wm = torch.std(recon_wm.float())
            mean_gm = torch.mean(recon_gm.float())
            std_gm = torch.std(recon_gm.float())

            # Calculate CNR for GM/WM
            if std_wm > 1e-9:
                cnr = (mean_gm - mean_wm) / std_wm
                cnr_scores.append(cnr.item())
                # Calculate CNR for Lesion/WM
                if torch.any(lesion_mask):
                    recon_lesion = recon_img[lesion_mask]
                    mean_lesion = torch.mean(recon_lesion.float())

                    # Avoid division by zero
                    if std_wm > 1e-9:
                        cnr_lesion = (mean_lesion - mean_wm) / std_wm
                        cnr_lesion_scores.append(cnr_lesion.item())

    if cnr_scores:
        mean_cnr = np.mean(cnr_scores)
        std_cnr = np.std(cnr_scores)
        cnr_results[name] = (mean_cnr, std_cnr)
        print(f"  Mean CNR (GM/WM): {mean_cnr:.2f} ± {std_cnr:.2f}")
    else:
        cnr_results[name] = (0, 0)

    if cnr_lesion_scores:
        mean_cnr_lesion = np.mean(cnr_lesion_scores)
        std_cnr_lesion = np.std(cnr_lesion_scores)
        cnr_lesion_results[name] = (mean_cnr_lesion, std_cnr_lesion)
        print(f"  Mean CNR (Lesion/WM): {mean_cnr_lesion:.2f} ± {std_cnr_lesion:.2f}")
    else:
        cnr_lesion_results[name] = (0, 0)
    print()
# %%
# Plotting the CNR results
methods = list(cnr_results.keys())
gm_wm_means = [cnr_results[method][0] for method in methods]
gm_wm_stds = [cnr_results[method][1] for method in methods]
lesion_gm_means = [cnr_lesion_results[method][0] for method in methods]
lesion_gm_stds = [cnr_lesion_results[method][1] for method in methods]

x = np.arange(len(methods))  # the label locations
width = 0.35  # the width of the bars
n_methods = len(methods)

fig, ax = plt.subplots(layout="constrained", figsize=(12, 7))

# Define positions for the two groups
group_centers = np.array([0, 1.5]) * (n_methods * width)
colors = sns.color_palette(n_colors=n_methods)

# Plot bars for GM/WM CNR
for i, method in enumerate(methods):
    pos = group_centers[0] - (n_methods - 1) * width / 2 + i * width
    ax.bar(
        pos,
        gm_wm_means[i],
        width * 0.9,
        yerr=gm_wm_stds[i],
        label=method if i == 0 else "",  # Label only once per group
        color=colors[i],
        capsize=5,
    )

# Plot bars for Lesion/GM CNR
for i, method in enumerate(methods):
    pos = group_centers[1] - (n_methods - 1) * width / 2 + i * width
    ax.bar(
        pos,
        lesion_gm_means[i],
        width * 0.9,
        yerr=lesion_gm_stds[i],
        color=colors[i],
        capsize=5,
    )

# Add some text for labels, title and axes ticks
fontsize = 20
ax.set_ylabel("Contrast-to-Noise Ratio (CNR)", fontsize=fontsize)
ax.set_xticks(group_centers)
ax.set_xticklabels(
    ["Grey-Matter / White-Matter", "Hot Lesions / White-Matter"], fontsize=fontsize
)
ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Create a single legend for all methods
ax.legend(methods, fontsize=fontsize)
plt.savefig("figure3.svg", format="svg")
plt.show()

# %%
