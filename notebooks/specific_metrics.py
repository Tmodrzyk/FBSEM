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
ax.set_ylabel("NRMSE (%)", fontsize=17)
ax.set_xticks(x + width * (len(methods) - 1) / 2, mask_names, fontsize=17)
ax.legend(loc="upper left", ncols=1, fontsize=17)
ax.set_ylim(0)
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()


# %%
# %%
# Calculate Contrast-to-Noise Ratio (CNR)
# CNR is defined as (mean(GM) - mean(WM)) / std(WM)

cnr_results = {}
wm_values = [32.0]
gm_values = [96.0]

for name, recon_dir in recon_dirs.items():
    print(f"Calculating CNR for {name}...")
    cnr_scores = []
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
        # Create masks for White Matter (WM) and Grey Matter (GM)
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

        if torch.any(wm_mask) and torch.any(gm_mask):
            recon_wm = recon_img[wm_mask]
            recon_gm = recon_img[gm_mask]

            mean_wm = torch.mean(recon_wm.float())
            std_wm = torch.std(recon_wm.float())
            mean_gm = torch.mean(recon_gm.float())

            # Avoid division by zero
            if std_wm > 1e-9:
                cnr = (mean_gm - mean_wm) / std_wm
                cnr_scores.append(cnr.item())

    if cnr_scores:
        mean_cnr = np.mean(cnr_scores)
        std_cnr = np.std(cnr_scores)
        cnr_results[name] = (mean_cnr, std_cnr)
        print(f"  Mean CNR: {mean_cnr:.2f} ± {std_cnr:.2f}")
    else:
        cnr_results[name] = (0, 0)
    print()
# %%
# Plotting the CNR results
methods = list(cnr_results.keys())
means = [cnr_results[method][0] for method in methods]
stds = [cnr_results[method][1] for method in methods]
x = np.arange(len(methods))

fig, ax = plt.subplots(layout="constrained", figsize=(10, 6))
rects = ax.bar(x, means, yerr=stds, capsize=5, color=sns.color_palette())

ax.set_ylabel("Contrast-to-Noise Ratio (CNR)")
ax.set_title("Contrast-to-Noise Ratio by Reconstruction Method")
ax.set_xticks(x, methods)
ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# ax.bar_label(rects, padding=3, fmt="%.2f")

plt.show()
# %%
