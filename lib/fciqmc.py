import math
import random

from .profiler import *
from .basis import *
from .hamiltonian import *


class FCIQMC:
    def __init__(self, basis: Basis, hamil: Hamiltonian, params: dict, n: int):
        self.basis: Basis = basis
        self.hamiltonian: Hamiltonian = hamil
        self.NMO: int = self.basis.NMO
        self.particle_number: int = n
        self.is_initiator = True
        self.initial_walkers: float = params["initial_walkers"]
        self.target_walker_number: float = params["target_walker_number"]
        self.d_tau: float = params["d_tau"]
        self.A: int = params["A"]
        self.xi: float = params["xi"]
        self.zeta: float = params["zeta"]
        self.steps: int = params["steps"]
        self.initiator_threshold: int = params["initiator_threshold"]
        self.onebody_probability: float = 0.5
        self.twobody_probability: float = 1.0 - self.onebody_probability
        self.min_spawn_num: float = 0.01
        self.min_walker_num: float = 0.01
        self.D0: Det = basis.minimum_det(self.particle_number)
        self.E0: float = self.hamiltonian.Hmat0(self.D0)
        self.S: float = self.E0
        self.walkers: dict = {self.D0: (self.initial_walkers, self.E0)}
        self.new_walkers: List[Tuple[Det, bool, float]] = []
        self.tau_trace: List[float] = []
        self.number_trace: List[float] = []
        self.shift_trace: List[float] = []
        self.energy_trace: List[float] = []

    # get total walker number Nw
    def get_number(self) -> float:
        sum_walkers = sum(abs(value[0]) for value in self.walkers.values())
        return sum_walkers

    # get projected energy
    def get_energy(self) -> float:
        energy = 0.0
        D0_data = self.walkers.get(self.D0)
        if D0_data is None:
            raise ValueError("error: cannot find D0 in all configurations...")
        else:
            N0 = D0_data[0]
            if N0 == 0.0:
                raise ValueError("error: number of walkers on D0 is 0")
            for Di, (Ni, Hii) in self.walkers.items():
                if Ni == 0.0:
                    continue
                H0i = self.hamiltonian.Hmat(self.D0, Di)
                if H0i == 0.0:
                    continue
                energy += Ni * H0i
        return energy / N0

    # get projected energy and total walker number Nw
    def get_energy_and_number(self) -> Tuple[float, float]:
        energy = 0.0
        if self.D0 not in self.walkers:
            raise ValueError("error: cannot find D0 in all configurations...")
        else:
            N0, E0 = self.walkers[self.D0]
            Nsum = 0.0
            if N0 == 0.0:
                raise ValueError("error: number of walkers on D0 is 0")
            for Di, (Ni, Hii) in self.walkers.items():
                H0i = self.hamiltonian.Hmat(self.D0, Di)
                if (H0i == 0.0) and (Ni == 0.0):
                    continue
                energy += Ni * H0i
                Nsum += abs(Ni)
        return (energy / N0, Nsum)

    # get mean and error of S, E, Nw
    def get_statistics(self, pos: float):
        if pos <= 0 or pos >= 1:
            raise ValueError("error: pos must be : 0 < pos < 1")
        start_index = int(pos * len(self.energy_trace))
        stat_energy = self.energy_trace[start_index:]
        stat_shift = self.shift_trace[start_index:]
        stat_number = self.number_trace[start_index:]
        energy_mean = np.mean(stat_energy)
        energy_deviation = np.std(stat_energy, ddof=1)
        shift_mean = np.mean(stat_shift)
        shift_deviation = np.std(stat_shift, ddof=1)
        number_mean = np.mean(stat_number)
        number_deviation = np.std(stat_number, ddof=1)
        return shift_mean, shift_deviation, energy_mean, energy_deviation, number_mean, number_deviation

    # print statistics of S, E, Nw
    def print_statistics(self, pos: float):
        S_mean, S_std, E_mean, E_std, N_mean, N_std = self.get_statistics(pos)
        # print(f"S mean = {S_mean}")
        # print(f"S error = {S_std}")
        print(f"E mean = {E_mean}")
        print(f"E error = {E_std}")
        # print(f"N mean = {N_mean}")
        # print(f"N error = {N_std}")

    # cut the absolute value of num to target
    def abs_cut_to(self, num: float, target: float) -> float:
        abs_num = abs(num)
        if abs_num >= target:
            return num
        else:
            sign = 1.0 if num >= 0 else -1.0
            return sign * target * float(random.uniform(0, target) < abs_num)

    # cut walker number to min_walker_num
    def walker_num_cut(self, num: float) -> float:
        abs_num = abs(num)
        if abs_num >= self.min_walker_num:
            return num
        else:
            sign = 1.0 if num >= 0 else -1.0
            return sign * self.min_walker_num * float(random.uniform(0, self.min_walker_num) < abs_num)

    # walker evolution step
    def step(self):
        for Di, (ci, Hii) in list(self.walkers.items()):
            Di: Det
            ci: float
            Hii: float
            Ci = self.walker_num_cut(ci)
            if Ci == 0.0:
                del self.walkers[Di]
                continue
            Ni = math.floor(Ci + random.uniform(0, 1))
            pd = self.d_tau * (Hii - self.S)
            self.walkers[Di] = (ci - pd * Ni, Hii)  # diagonal step
            is_initiator = self.is_initiator and (abs(Ci) > self.initiator_threshold)
            for dummy in range(abs(Ni)):
                Df = Di.copy()
                invp = 0.0
                Hfi = 0.0
                if random.uniform(0, 1) < self.onebody_probability:
                    a, b, local_invp = self.basis.single_excite(Di)
                    if local_invp == 0:
                        continue
                    invp = local_invp / self.onebody_probability
                    Df.reset(a)
                    Hfi = self.hamiltonian.Hmat1(Df, a, b)
                    Df.set(b)
                else:
                    a, b, c, d, local_invp = self.basis.double_excite(Di)
                    if local_invp == 0:
                        continue
                    invp = local_invp / self.twobody_probability
                    Df.reset(a)
                    Df.reset(b)
                    Hfi = self.hamiltonian.Hmat2(Df, a, b, c, d)
                    Df.set(c)
                    Df.set(d)
                spawn_num = -sign(Ni) * self.d_tau * Hfi * invp
                spawn_num = self.abs_cut_to(spawn_num, self.min_spawn_num)
                if spawn_num == 0.0:
                    continue
                self.new_walkers.append((Df, is_initiator, spawn_num))

    # walker annihilation step
    def annihilation(self):
        for Df, is_initiator, spawn_num in self.new_walkers:
            if (Df not in self.walkers) and (not is_initiator):
                continue
            else:
                if Df in self.walkers:
                    current_num, current_energy = self.walkers[Df]
                    new_num = current_num + spawn_num
                    if new_num == 0.0:
                        del self.walkers[Df]
                        continue
                    else:
                        self.walkers[Df] = (new_num, current_energy)
                else:
                    new_num = spawn_num
                    current_energy = self.hamiltonian.Hmat0(Df)
                    self.walkers[Df] = (new_num, current_energy)
        self.new_walkers.clear()

    # warm up step
    def warm(self):
        warm_up_count = 0
        total_number = self.get_number()
        while total_number < self.target_walker_number:
            warm_up_count += 1
            self.step()
            self.annihilation()
            energy, total_number = self.get_energy_and_number()
            if warm_up_count % self.A == 0:
                print(f"{warm_up_count:>6}{total_number:>12.3f}{energy:>16.3f}")
        print(f"warm up steps: {warm_up_count}")

    # search optimal d_tau in warm up step
    def warm_search(self):
        warm_up_steps_max = 1e4
        for d_tau_now in np.logspace(-9, -1, num=100):
            self.d_tau = d_tau_now
            self.walkers = {self.D0: (self.initial_walkers, self.E0)}
            warm_up_count = 0
            total_number = self.get_number()
            while total_number < self.target_walker_number:
                warm_up_count += 1
                self.step()
                self.annihilation()
                energy, total_number = self.get_energy_and_number()
                if warm_up_count % self.A == 0:
                    print(f"{warm_up_count:>6}{total_number:>12.3f}{energy:>16.3f}")
                if warm_up_count > warm_up_steps_max:
                    break
            if warm_up_count < warm_up_steps_max:
                break
            print(f"warm up steps: {warm_up_count}")

    # start FCIQMC algorithm
    def start(self):
        print("! evolution begins")
        print(f"!{'step':>5}{'S':>16}{'E':>16}{'Nw':>16}")
        energy = self.get_energy()
        new_num = self.get_number()
        old_num = new_num
        for i in range(self.steps):
            self.step()
            self.annihilation()
            if i % self.A == 0:
                old_num = new_num
                energy, new_num = self.get_energy_and_number()
                print(f"{i:>6}{self.S:>16.3f}{energy:>16.3f}{new_num:>16.3e}")
                self.tau_trace.append(self.d_tau * i)
                self.number_trace.append(new_num)
                self.shift_trace.append(self.S)
                self.energy_trace.append(energy)
                self.S = self.S - self.xi / (self.A * self.d_tau) * math.log(new_num / old_num) - self.zeta / (self.A * self.d_tau) * math.log(new_num / self.target_walker_number)
        print("! evolution ends")
