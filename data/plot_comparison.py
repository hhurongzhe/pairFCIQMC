import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def load_data_by_header(file_path, headers, delimiter=None):
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
    if delimiter is None:
        all_cols = first_line.split()
    else:
        all_cols = first_line.split(delimiter)
    if isinstance(headers, str):
        target_cols = [headers]
    else:
        target_cols = list(headers)
    col_indices = []
    for col in target_cols:
        if col not in all_cols:
            raise ValueError(f"'{col}' not in header. Available columns: {all_cols}")
        col_indices.append(all_cols.index(col))
    data = np.loadtxt(file_path, dtype=float, delimiter=delimiter, skiprows=1, usecols=col_indices)
    if data.ndim == 1 or len(col_indices) == 1:
        return data.flatten()
    else:
        return data


data = "./data/data.txt"
g_list = load_data_by_header(data, "g")
e_ref_list = load_data_by_header(data, "ref")
e_fci = load_data_by_header(data, "fci")
e_fciqmc = load_data_by_header(data, "fciqmc")
e_fciqmc_error = load_data_by_header(data, "dfciqmc")
e_mbpt2 = load_data_by_header(data, "mbpt2")
e_mbpt3 = load_data_by_header(data, "mbpt3")
e_mbpt4 = load_data_by_header(data, "mbpt4")
e_ccd = load_data_by_header(data, "ccd")
e_imsrg2 = load_data_by_header(data, "imsrg2")
e_adc2 = load_data_by_header(data, "adc2")
e_adc3 = load_data_by_header(data, "adc3")
e_adc3_d = load_data_by_header(data, "adc3-d")


config = {
    "figure.dpi": 160,
    "mathtext.fontset": "stix",
    "font.family": "Times New Roman",
    "legend.fancybox": True,
    "legend.handletextpad": 0.7,
    "legend.framealpha": 1.0,
    "legend.handlelength": 1.6,
    "patch.linewidth": 1.0,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "xtick.color": "k",
    "ytick.color": "k",
}
rcParams.update(config)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.3))


ax1.minorticks_on()
ax1.tick_params(direction="in", width=0.7, length=4, top=True, bottom=True, left=True, right=True, axis="both")
ax1.tick_params(which="minor", direction="in", width=0.5, length=2, top=True, bottom=True, left=True, right=True, axis="both")
ax1.tick_params(axis="both", labelsize=9)

ax1.set_title("pairing model", fontsize=9)
ax1.set_xlabel(r"$g\; \mathrm{[a.u.]}$", fontsize=9)
ax1.set_ylabel(r"$E_\mathrm{corr}\; \mathrm{[a.u.]}$", fontsize=9)

ax1.plot(g_list, e_fci, c="black", linewidth=1, linestyle="-", zorder=0, label=r"FCI")
ax1.plot(g_list, e_mbpt2, c="C0", marker="v", markersize=3, linewidth=0.7, linestyle="--", zorder=0, label=r"MBPT(2)", markeredgewidth=0)
ax1.plot(g_list, e_mbpt3, c="C2", marker="^", markersize=3, linewidth=0.7, linestyle="-.", zorder=0, label=r"MBPT(3)", markeredgewidth=0)
ax1.plot(g_list, e_mbpt4, c="C6", marker="s", markersize=2.5, linewidth=0.7, linestyle="-.", zorder=0, label=r"MBPT(4)", markeredgewidth=0)
ax1.plot(g_list, e_ccd, c="C4", marker="D", markersize=2.5, linewidth=0.7, linestyle="--", zorder=0, label=r"CCD", markeredgewidth=0)
ax1.plot(g_list, e_adc2, c="C1", marker="o", markersize=2.5, linewidth=0.7, linestyle="--", zorder=0, label=r"ADC(2)", markeredgewidth=0)
ax1.plot(g_list, e_adc3, c="C5", marker="o", markersize=2.5, linewidth=0.7, linestyle="--", zorder=0, label=r"ADC(3)", markeredgewidth=0)
ax1.plot(g_list, e_adc3_d, c="C8", marker="o", markersize=2.5, linewidth=0.7, linestyle="--", zorder=0, label=r"ADC(3)-D", markeredgewidth=0)
ax1.plot(g_list, e_imsrg2, c="C9", marker="h", markersize=3, linewidth=0.7, linestyle=":", markeredgewidth=0, zorder=0, label=r"IMSRG(2)")
ax1.plot(g_list, e_fciqmc, c="C3", linewidth=0.7, linestyle="--", zorder=1)
ax1.errorbar(g_list, e_fciqmc, yerr=e_fciqmc_error, fmt="o", c="C3", markersize=2.5, linewidth=0.7, elinewidth=0.7, capsize=0, label=r"FCIQMC", zorder=1, markeredgewidth=0)

bwith = 0.7
for spine in ["bottom", "top", "left", "right"]:
    ax1.spines[spine].set_linewidth(bwith)

ax1.legend(loc="lower center", fontsize=6, frameon=True, ncols=2)


ax2.minorticks_on()
ax2.tick_params(direction="in", width=0.7, length=4, top=True, bottom=True, left=True, right=True, axis="both")
ax2.tick_params(which="minor", direction="in", width=0.5, length=2, top=True, bottom=True, left=True, right=True, axis="both")
ax2.tick_params(axis="both", labelsize=9)

ax2.set_title("pairing model", fontsize=9)
ax2.set_xlabel(r"$g\; \mathrm{[a.u.]}$", fontsize=9)
ax2.set_ylabel("Error [a.u.]", fontsize=9)


err_mbpt2 = e_mbpt2 - e_fci
err_mbpt3 = e_mbpt3 - e_fci
err_mbpt4 = e_mbpt4 - e_fci
err_ccd = e_ccd - e_fci
err_adc2 = e_adc2 - e_fci
err_adc3 = e_adc3 - e_fci
err_adc3_d = e_adc3_d - e_fci
err_imsrg2 = e_imsrg2 - e_fci
err_fciqmc = e_fciqmc - e_fci

ax2.axhline(y=0, color="black", linewidth=1.0, linestyle="-", label=r"FCI")
ax2.plot(g_list, err_mbpt2, c="C0", marker="v", markersize=3, linewidth=0.7, linestyle="--", label=r"MBPT(2)", markeredgewidth=0)
ax2.plot(g_list, err_mbpt3, c="C2", marker="^", markersize=3, linewidth=0.7, linestyle="-.", label=r"MBPT(3)", markeredgewidth=0)
ax2.plot(g_list, err_mbpt4, c="C6", marker="s", markersize=2.5, linewidth=0.7, linestyle="-.", label=r"MBPT(4)", markeredgewidth=0)
ax2.plot(g_list, err_ccd, c="C4", marker="D", markersize=2.5, linewidth=0.7, linestyle=":", label=r"CCD", markeredgewidth=0)
ax2.plot(g_list, err_adc2, c="C1", marker="o", markersize=2.5, linewidth=0.7, linestyle=":", label=r"ADC(2)", markeredgewidth=0)
ax2.plot(g_list, err_adc3, c="C5", marker="o", markersize=2.5, linewidth=0.7, linestyle=":", label=r"ADC(3)", markeredgewidth=0)
ax2.plot(g_list, err_adc3_d, c="C8", marker="o", markersize=2.5, linewidth=0.7, linestyle=":", label=r"ADC(3)-D", markeredgewidth=0)
ax2.plot(g_list, err_imsrg2, c="C9", marker="h", markersize=3, linewidth=0.7, linestyle=":", markeredgewidth=0, label=r"IMSRG(2)")
ax2.plot(g_list, err_fciqmc, c="C3", linewidth=0.7, linestyle="--")
ax2.errorbar(g_list, err_fciqmc, yerr=e_fciqmc_error, fmt="o", c="C3", markersize=2.5, linewidth=0.7, elinewidth=0.7, capsize=0, label=r"FCIQMC", markeredgewidth=0)

for spine in ["bottom", "top", "left", "right"]:
    ax2.spines[spine].set_linewidth(bwith)

axins = inset_axes(ax2, width=0.8, height=0.5, bbox_to_anchor=(0.75, 0.5), bbox_transform=ax2.transAxes)
axins.set_xlim(0.58, 1.02)
axins.set_ylim(-0.02, 0.01)
axins.axhline(y=0, color="black", linewidth=1.0, linestyle="-", label=r"FCI")
axins.plot(g_list, err_mbpt2, c="C0", marker="v", markersize=3, linewidth=0.7, linestyle="--", label=r"MBPT(2)", markeredgewidth=0)
axins.plot(g_list, err_mbpt3, c="C2", marker="^", markersize=3, linewidth=0.7, linestyle="-.", label=r"MBPT(3)", markeredgewidth=0)
axins.plot(g_list, err_mbpt4, c="C6", marker="s", markersize=2.5, linewidth=0.7, linestyle="-.", label=r"MBPT(4)", markeredgewidth=0)
axins.plot(g_list, err_ccd, c="C4", marker="D", markersize=2.5, linewidth=0.7, linestyle=":", label=r"CCD", markeredgewidth=0)
axins.plot(g_list, err_adc2, c="C1", marker="o", markersize=2.5, linewidth=0.7, linestyle=":", label=r"ADC(2)", markeredgewidth=0)
axins.plot(g_list, err_adc3, c="C5", marker="o", markersize=2.5, linewidth=0.7, linestyle=":", label=r"ADC(3)", markeredgewidth=0)
axins.plot(g_list, err_adc3_d, c="C8", marker="o", markersize=2.5, linewidth=0.7, linestyle=":", label=r"ADC(3)-D", markeredgewidth=0)
axins.plot(g_list, err_imsrg2, c="C9", marker="h", markersize=3, linewidth=0.7, linestyle=":", markeredgewidth=0, label=r"IMSRG(2)")
axins.plot(g_list, err_fciqmc, c="C3", linewidth=0.7, linestyle="--")
axins.errorbar(g_list, err_fciqmc, yerr=e_fciqmc_error, fmt="o", c="C3", markersize=2.5, linewidth=0.7, elinewidth=0.7, capsize=0, label=r"FCIQMC", markeredgewidth=0)
axins.minorticks_on()
axins.tick_params(direction="in", width=0.7, length=4, top=True, bottom=True, left=True, right=True, axis="both")
axins.tick_params(which="minor", direction="in", width=0.5, length=2, top=True, bottom=True, left=True, right=True, axis="both")
axins.tick_params(axis="both", labelsize=6)
for spine in ["bottom", "top", "left", "right"]:
    axins.spines[spine].set_linewidth(bwith)

plt.tight_layout()
plt.savefig("./result/fig_comparison.png", bbox_inches="tight", transparent=False, dpi=1200)
plt.show()
