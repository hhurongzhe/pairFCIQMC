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
from lib.fciqmc import *


params = {}
params["initial_walkers"] = 10
params["target_walker_number"] = 1000
params["d_tau"] = 1e-2
params["A"] = 10
params["xi"] = 0.1
params["zeta"] = 0.01
params["steps"] = 3000
params["initiator_threshold"] = 1


def main():
    header_message()
    profiler = Profiler()

    n = 4
    p_max = 4
    delta, g = 1.0, 1.0

    section_message("basis setting up")
    t1 = time.time()
    basis = Basis(p_max, delta, n)
    t2 = time.time()
    profiler.add_timing("set up basis", t2 - t1)

    t1 = time.time()
    section_message("hamiltonian setting up")
    hamiltonian = Hamiltonian(basis, g)
    t2 = time.time()
    profiler.add_timing("set up hamiltonian", t2 - t1)

    section_message("fci algorithm")
    t1 = time.time()
    fci = FCI(basis, hamiltonian, n)
    fci.build_configurations()
    fci.build_hamiltonian_matrix()
    fci.solve()
    print(f"E(FCI) = {fci.emin}")
    t2 = time.time()
    profiler.add_timing("fci algorithm", t2 - t1)

    section_message("fciqmc algorithm")
    t1 = time.time()
    fciqmc = FCIQMC(basis, hamiltonian, params, n)
    print(f"D0 = {repr(fciqmc.D0)}")
    print(f"E0 = {fciqmc.E0}")
    print(f"N0 = {fciqmc.get_number()}")
    fciqmc.warm()
    fciqmc.start()
    fciqmc.print_statistics(pos=0.5)
    t2 = time.time()
    profiler.add_timing("fciqmc algorithm", t2 - t1)

    section_message("plotting")
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
    plt.xlabel(r"$\tau\; \mathrm{[a.u.]}$", fontsize=9)
    plt.ylabel(r"$E\; \mathrm{[a.u.]}$", fontsize=9)
    plt.hlines(fci.emin, min(fciqmc.tau_trace), max(fciqmc.tau_trace), color="black", linestyle="--", linewidth=0.7, zorder=2, label=r"FCI")
    plt.plot(fciqmc.tau_trace, fciqmc.shift_trace, c="C2", linewidth=0.7, zorder=0, label=r"$S(\tau)$")
    plt.plot(fciqmc.tau_trace, fciqmc.energy_trace, c="C3", linewidth=0.7, zorder=1, label=r"$E(\tau)$")
    bwith = 0.7
    tk = plt.gca()
    tk.spines["bottom"].set_linewidth(bwith)
    tk.spines["top"].set_linewidth(bwith)
    tk.spines["left"].set_linewidth(bwith)
    tk.spines["right"].set_linewidth(bwith)
    plt.legend(loc="upper right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(f"./result/fig_fciqmc.png", bbox_inches="tight", transparent=False, dpi=1200)
    plt.show()

    section_message("Timings")
    profiler.print_timings()
    footer_message()


if __name__ == "__main__":
    main()
