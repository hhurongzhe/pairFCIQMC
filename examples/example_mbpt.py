import matplotlib.pyplot as plt
from matplotlib import rcParams

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lib.profiler import *
from lib.utility import *
from lib.orbit import *
from lib.basis import *
from lib.hamiltonian import *
from lib.fci import *
from lib.mbpt import *


def main():
    header_message()

    n = 4
    p_max = 4
    delta = 1.0
    g_list = np.linspace(-1, 1, 21)

    basis = Basis(p_max, delta, n)

    e_ref_list = 2.0 - g_list
    e_fci = np.zeros_like(g_list)
    # independent correlation energy from MBPT
    e_part_mbpt2 = np.zeros_like(g_list)
    e_part_mbpt3 = np.zeros_like(g_list)
    # e_part_mbpt4 = np.zeros_like(g_list)
    for i, g in enumerate(g_list):
        print("calculating for g =", g)
        hamiltonian = Hamiltonian(basis, g)

        fci = FCI(basis, hamiltonian, n)
        fci.build_configurations()
        fci.build_hamiltonian_matrix()
        fci.solve()
        e_fci[i] = fci.emin

        mbpt = MBPT(basis, hamiltonian)
        e_part_mbpt2[i] = mbpt.cal_coor2(g)
        e_part_mbpt3[i] = mbpt.cal_coor3(g)
        # e_part_mbpt4[i] = mbpt.cal_coor4(g)
    e_mbpt2 = e_part_mbpt2
    e_mbpt3 = e_part_mbpt2 + e_part_mbpt3
    # e_mbpt4 = e_part_mbpt2 + e_part_mbpt3 + e_part_mbpt4
    print(e_mbpt2.tolist())
    print(e_mbpt3.tolist())
    # print(e_mbpt4.tolist())
    show = True
    if show:
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
        plt.figure(figsize=(3, 2))
        plt.minorticks_on()
        plt.tick_params(direction="in", width=0.7, length=4, top=True, bottom=True, left=True, right=True, axis="both")
        plt.tick_params(which="minor", direction="in", width=0.5, length=2, top=True, bottom=True, left=True, right=True, axis="both")
        plt.tick_params(axis="both", labelsize=9)
        plt.title("pairing model", fontsize=9)
        plt.xlabel(r"$g\; \mathrm{[a.u.]}$", fontsize=9)
        plt.ylabel(r"$E_\mathrm{corr}\; \mathrm{[a.u.]}$", fontsize=9)
        plt.plot(g_list, e_fci - e_ref_list, c="black", linewidth=0.7, zorder=0, label=r"FCI")
        plt.plot(g_list, e_mbpt2, c="C0", marker="v", markersize=1.4, linewidth=0.7, linestyle="--", zorder=0, label=r"MBPT(2)")
        plt.plot(g_list, e_mbpt3, c="C2", marker="s", markersize=1.4, linewidth=0.7, linestyle="-.", zorder=0, label=r"MBPT(3)")
        # plt.plot(g_list, e_mbpt4, c="C3", marker="D", markersize=1.4, linewidth=0.7, linestyle="-.", zorder=0, label=r"MBPT(4)")
        bwith = 0.7
        tk = plt.gca()
        tk.spines["bottom"].set_linewidth(bwith)
        tk.spines["top"].set_linewidth(bwith)
        tk.spines["left"].set_linewidth(bwith)
        tk.spines["right"].set_linewidth(bwith)
        plt.legend(loc="lower center", fontsize=8, frameon=True)
        plt.tight_layout()
        plt.savefig(f"./result/fig_mbpt.png", bbox_inches="tight", transparent=False, dpi=1200)
        plt.show()

    footer_message()


if __name__ == "__main__":
    main()
