import math
import numpy as np
import itertools

from .profiler import *
from .basis import *
from .hamiltonian import *


class FCI:
    def __init__(self, basis: Basis, hamil: Hamiltonian, n: int):
        self.basis = basis
        self.hamiltonian = hamil
        self.NMO = self.basis.NMO  # number of sp orbitals
        self.particle_number = n  # number of particles
        if self.particle_number > self.NMO:
            raise ValueError("error in init FCI...")
        self.dim = math.comb(self.NMO, self.particle_number)  # FCI dimension
        self.configs = []  # all possible FCI configurations
        self.hamil_matrix = np.zeros((self.dim, self.dim))  # FCI hamiltonian matrix
        self.eigenvalues = np.array([])  # all eigenvalues
        self.eigenvectors = np.array([])  # all eigenvectors
        self.emin = None  # ground-state energy

    # build all possible configurations
    def build_configurations(self):
        configs = list(itertools.combinations(range(self.NMO), self.particle_number))
        self.configs = configs

    # build FCI hamiltonian matrix
    def build_hamiltonian_matrix(self):
        for idx_f in range(self.dim):
            config_f = self.configs[idx_f]
            Df = Det(config_f, self.NMO)
            for idx_i in range(idx_f, self.dim):
                config_i = self.configs[idx_i]
                Di = Det(config_i, self.NMO)
                Hfi = self.hamiltonian.Hmat(Df, Di)
                self.hamil_matrix[idx_f][idx_i] = Hfi
                if idx_f < idx_i:
                    self.hamil_matrix[idx_i][idx_f] = Hfi

    def solve(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.hamil_matrix)
        self.emin = np.real(np.min(self.eigenvalues))

    def print_states(self, show_vectors=False):
        print(self.eigenvalues)
        if show_vectors:
            print(self.eigenvectors)
