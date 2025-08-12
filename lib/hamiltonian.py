import numpy as np

from .mymath import *
from .basis import *


class Hamiltonian:
    # basis: one-body basis structure
    # g: pairing-interaction strength
    def __init__(self, basis: Basis, g: float = 0.1, ini: bool = True):
        self.basis: Basis = basis
        self.NMO: int = self.basis.NMO
        self.g: float = g
        if ini:
            self.v0mat: float = 0.0
            self.ch_v1mat: np.ndarray = self.init_v1mat()
            self.ch_v2mat: List[np.ndarray] = self.init_v2mat()

    # initialize one-body matrix elements
    def init_v1mat(self) -> np.ndarray:
        ch_v1mat = []
        for orb_index in range(self.NMO):
            value = self.basis.get_orbit(orb_index).e
            ch_v1mat.append(value)
        return np.array(ch_v1mat)

    # two-body interactions for pairing model
    def cal_v2mat(self, orb_a: Orbital, orb_b: Orbital, orb_c: Orbital, orb_d: Orbital) -> float:
        pa = orb_a.p
        pb = orb_b.p
        pc = orb_c.p
        pd = orb_d.p
        sa = orb_a.s
        sb = orb_b.s
        sc = orb_c.s
        sd = orb_d.s
        if pa != pb or pc != pd:
            return 0.0
        if sa == sb or sc == sd:
            return 0.0
        if sa == sc and sb == sd:
            return -self.g / 2.0
        if sa == sd and sb == sc:
            return self.g / 2.0
        return 0.0

    # initialize two-body matrix elements
    def init_v2mat(self) -> List[np.ndarray]:
        ch_v2mat = []
        for channel_index in range(self.basis.two_body_channel_number):
            channel_size = len(self.basis.two_body_basis_channel[channel_index])
            ch_v2mat_this = [[0.0 for _ in range(channel_size)] for _ in range(channel_size)]
            for pos_left in range(channel_size):
                a, b = self.basis.two_body_basis_channel[channel_index][pos_left]
                orb_a = self.basis.get_orbit(a)
                orb_b = self.basis.get_orbit(b)
                for pos_right in range(channel_size):
                    c, d = self.basis.two_body_basis_channel[channel_index][pos_right]
                    orb_c = self.basis.get_orbit(c)
                    orb_d = self.basis.get_orbit(d)
                    if not check_two_body_symmetry(orb_a, orb_b, orb_c, orb_d):
                        raise ValueError("error in init_v2mat...")
                    value = self.cal_v2mat(orb_a, orb_b, orb_c, orb_d)
                    ch_v2mat_this[pos_left][pos_right] = value
            ch_v2mat.append(ch_v2mat_this)
        return [np.array(channel_data, dtype=float) for channel_data in ch_v2mat]

    # find two-body matrix elements
    def find_v2mat(self, a: int, b: int, c: int, d: int) -> float:
        channel_index_ab = self.basis.get_two_body_channel_index(a, b)
        channel_index_cd = self.basis.get_two_body_channel_index(c, d)
        channel_position_ab = self.basis.get_two_body_channel_position(a, b)
        channel_position_cd = self.basis.get_two_body_channel_position(c, d)
        if channel_index_ab != channel_index_cd:  # in FCI this will happen
            return 0.0
        return self.ch_v2mat[channel_index_ab][channel_position_ab][channel_position_cd]

    def Hmat0(self, D: Det) -> float:
        vsum = self.v0mat
        for k in range(self.NMO):
            if D.is_occupied(k):
                vsum += self.ch_v1mat[k]  # 1-body contribution h_{k}
                for l in range(k + 1, self.NMO):
                    if D.is_occupied(l):
                        vsum += self.find_v2mat(k, l, k, l)  # 2-body contribution v_{klkl}
        return vsum

    def Hmat1(self, D: Det, a: int, b: int) -> float:
        if b > a:
            a, b = b, a
        Dbits = D.bits
        bits_permute = (Dbits >> (b + 1)) << (self.NMO - a + b + 1)
        Dpermute = Det.from_int(bits_permute, self.NMO)
        permute = Dpermute.count_occupation()
        vsum = 0.0
        for k in range(self.NMO):
            if D.is_occupied(k):
                vsum += self.find_v2mat(b, k, a, k)  # 2-body contribution v_{bkak}
        return iphase_double(permute) * vsum

    def Hmat2(self, D: Det, a: int, b: int, c: int, d: int) -> float:
        Dbits = D.bits
        bits_permute_ab = (Dbits >> (a + 1)) << (self.NMO - b + a + 1)  # permute(a,b)
        bits_permute_cd = (Dbits >> (c + 1)) << (self.NMO - d + c + 1)  # permute(c,d)
        bits_permute = bits_permute_ab + bits_permute_cd
        Dpermute = Det.from_int(bits_permute, self.NMO)
        permute = Dpermute.count_occupation()
        return iphase_double(permute) * self.find_v2mat(c, d, a, b)  # 2-body contribution v_{cdab}

    # calculate <Df|H|Di>
    def Hmat(self, Df: Det, Di: Det) -> float:
        Ddiff = Df ^ Di
        diff = Ddiff.count_occupation()  # number of different orbitals of Df and Di
        Dsame = Df & Di
        if diff == 0:
            # diagonal part
            # <D|H|D>, with Df=Di=D
            return self.Hmat0(Dsame)
        elif diff == 2:
            # single-excitation part
            ai = (Ddiff & Di).get_occupied_indices()[0]
            af = (Ddiff & Df).get_occupied_indices()[0]
            if not check_one_body_symmetry(self.basis.get_orbit(ai), self.basis.get_orbit(af)):
                return 0.0
            return self.Hmat1(Dsame, ai, af)
        elif diff == 4:
            # double-excitation part
            i1, i2 = (Ddiff & Di).get_occupied_indices()
            f1, f2 = (Ddiff & Df).get_occupied_indices()
            if not check_two_body_symmetry(self.basis.get_orbit(i1), self.basis.get_orbit(i2), self.basis.get_orbit(f1), self.basis.get_orbit(f2)):
                return 0.0
            return self.Hmat2(Dsame, i1, i2, f1, f2)
        else:
            # cannot happen
            return 0
