import itertools
from typing import List, Tuple, Iterable
from collections import defaultdict
import random

from .orbit import *


class Det:
    def __init__(self, occupied_indices: Iterable[int], nmo: int):
        # occupied_indices: [0, 1, 5] for example
        # nmo: total length of Det
        self.nmo: int = nmo
        self.bits: int = 0
        for index in occupied_indices:
            if not (0 <= index < nmo):
                raise ValueError(f"index: {index} out of range: [0, {nmo-1}]!")
            self.bits |= 1 << index

    @classmethod
    def from_int(cls, bits: int, nmo: int) -> "Det":
        instance = cls([], nmo)
        instance.bits = bits
        return instance

    def is_occupied(self, index: int) -> bool:
        if not (0 <= index < self.nmo):
            raise IndexError(f"index: {index} out of range: [0, {self.nmo-1}]。")
        return bool((self.bits >> index) & 1)

    def get_occupied_indices(self) -> Tuple[int, ...]:
        indices = []
        bits = self.bits
        index = 0
        while bits > 0:
            if bits & 1:
                indices.append(index)
            bits >>= 1
            index += 1
        return tuple(indices)

    def get_unoccupied_indices(self) -> Tuple[int, ...]:
        unoccupied_indices = []
        for i in range(self.nmo):
            if not self.is_occupied(i):
                unoccupied_indices.append(i)
        return tuple(unoccupied_indices)

    def count_occupation(self) -> int:
        return self.bits.bit_count()

    def set(self, n: int) -> None:
        if not (0 <= n < self.nmo):
            raise IndexError(f"index: {n} out of range: [0, {self.nmo-1}]。")
        self.bits |= 1 << n

    def reset(self, n: int) -> None:
        if not (0 <= n < self.nmo):
            raise IndexError(f"index: {n} out of range: [0, {self.nmo-1}]。")
        self.bits &= ~(1 << n)

    def find_nth(self, n: int) -> int:
        if n <= 0:
            return self.nmo
        count = 0
        for i in range(self.nmo):
            if self.is_occupied(i) and (count := count + 1) == n:
                return i
        return self.nmo

    def copy(self) -> "Det":
        return Det(self.get_occupied_indices(), self.nmo)

    def __and__(self, other: "Det") -> "Det":
        return Det.from_int(self.bits & other.bits, self.nmo)

    def __or__(self, other: "Det") -> "Det":
        return Det.from_int(self.bits | other.bits, self.nmo)

    def __xor__(self, other: "Det") -> "Det":
        return Det.from_int(self.bits ^ other.bits, self.nmo)

    def __hash__(self) -> int:
        return hash((self.bits, self.nmo))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Det):
            return NotImplemented
        return self.bits == other.bits and self.nmo == other.nmo

    def __repr__(self) -> str:
        standard_bin = bin(self.bits)[2:]
        padded_bin = standard_bin.zfill(self.nmo)
        left_to_right_bin = padded_bin[::-1]
        return f"Det({left_to_right_bin})"

    def __str__(self) -> str:
        return f"Det(NMO={self.nmo}, occupation number={self.count_occupation()}, Indices={self.get_occupied_indices()})"


class Basis:
    def __init__(self, p_max: int, delta: float = 1, n: int = 4):
        self.p_max: int = p_max
        self.delta: float = delta
        self.sp_orbits: List[Orbital] = build_sp_orbits(p_max, delta)
        self.NMO: int = len(self.sp_orbits)
        self.particle_number: int = n
        # build one-body basis
        self.one_body_basis: List[int] = self.build_one_body_basis()
        self.one_body_basis_map: dict[int, List[int]] = self.build_one_body_basis_sorted()
        self.one_body_basis_channel: List[List[int]] = self.build_one_body_basis_channel()
        self.one_body_basis_number: int = len(self.one_body_basis)
        self.one_body_channel_number: int = len(self.one_body_basis_channel)
        self.one_body_basis_map.clear()
        self.one_body_state_channel_indices = []
        self.one_body_state_channel_positions = []
        # build two-body basis
        self.two_body_basis: List[Tuple[int, int]] = self.build_two_body_basis()
        self.two_body_basis_map: dict[int, List[Tuple[int, int]]] = self.build_two_body_basis_sorted()
        self.two_body_basis_channel: List[List[Tuple[int, int]]] = self.build_two_body_basis_channel()
        self.two_body_basis_number: int = len(self.two_body_basis)
        self.two_body_channel_number: int = len(self.two_body_basis_channel)
        self.two_body_basis_map.clear()
        self.two_body_state_channel_indices = []
        self.two_body_state_channel_positions = []
        # build states connections, which is important for fciqmc algorithm
        self.build_connections()

    def get_orbit(self, index: int) -> Orbital:
        return self.sp_orbits[index]

    def build_one_body_basis(self) -> List[int]:
        one_body_basis = []
        for orb in self.sp_orbits:
            index = orb.i
            one_body_basis.append(index)
        return one_body_basis

    # sort one-body states into map
    def build_one_body_basis_sorted(self) -> defaultdict[int, List[int]]:
        one_body_basis_map = defaultdict(list)
        for orb_index in self.one_body_basis:
            orb = self.get_orbit(orb_index)
            key = one_body_symmetry_key(orb)
            one_body_basis_map[key].append(orb_index)
        return one_body_basis_map

    # build one-body-states-vector in channels
    def build_one_body_basis_channel(self) -> List[List[int]]:
        one_body_basis_channel = []
        for key in self.one_body_basis_map:
            states_channel = self.one_body_basis_map[key]
            one_body_basis_channel.append(states_channel)
        return one_body_basis_channel

    # build two-body basis
    def build_two_body_basis(self) -> List[Tuple[int, int]]:
        return list(itertools.combinations(self.one_body_basis, 2))

    # sort two-body states into map
    def build_two_body_basis_sorted(self) -> defaultdict[int, List[Tuple[int, int]]]:
        two_body_basis_map = defaultdict(list)
        for index_orb_a, index_orb_b in self.two_body_basis:
            orb_a = self.get_orbit(index_orb_a)
            orb_b = self.get_orbit(index_orb_b)
            key = two_body_symmetry_key(orb_a, orb_b)
            two_body_basis_map[key].append((index_orb_a, index_orb_b))
        return two_body_basis_map

    # build two-body-states-vector in channels
    def build_two_body_basis_channel(self) -> List[List[Tuple[int, int]]]:
        two_body_basis_channel = []
        for key in self.two_body_basis_map:
            states_channel = self.two_body_basis_map[key]
            two_body_basis_channel.append(states_channel)
        return two_body_basis_channel

    def get_one_body_channel(self, index_i: int):
        channel_index = self.one_body_state_channel_indices[index_i]
        return self.one_body_basis_channel[channel_index]

    def get_two_body_channel(self, index_i: int, index_j: int):
        channel_index = self.get_two_body_channel_index(index_i, index_j)
        return self.two_body_basis_channel[channel_index]

    def get_two_body_connection_pos(self, index_i: int, index_j: int) -> int:
        n = self.NMO
        i = index_i
        j = index_j
        return int(i * (2 * n - i - 3) / 2 + j - 1)

    def build_connections(self):
        # build one-body connections
        self.one_body_state_channel_indices = [-1] * self.NMO
        self.one_body_state_channel_positions = [-1] * self.NMO
        for channel_index in range(self.one_body_channel_number):
            channel_size = len(self.one_body_basis_channel[channel_index])
            for channel_position in range(channel_size):
                orbit_index = self.one_body_basis_channel[channel_index][channel_position]
                self.one_body_state_channel_indices[orbit_index] = channel_index
                self.one_body_state_channel_positions[orbit_index] = channel_position
        # build two-body connections
        self.two_body_state_channel_indices = [-1] * int(self.NMO * (self.NMO - 1) / 2)
        self.two_body_state_channel_positions = [-1] * int(self.NMO * (self.NMO - 1) / 2)
        for channel_index in range(self.two_body_channel_number):
            channel_size = len(self.two_body_basis_channel[channel_index])
            for channel_position in range(channel_size):
                orbit_index_i, orbit_index_j = self.two_body_basis_channel[channel_index][channel_position]
                connection_pos = self.get_two_body_connection_pos(orbit_index_i, orbit_index_j)
                if connection_pos < 0 or connection_pos >= len(self.two_body_state_channel_indices) or self.two_body_state_channel_indices[connection_pos] != -1:
                    raise ValueError("error in build_connections()...")
                self.two_body_state_channel_indices[connection_pos] = channel_index
                self.two_body_state_channel_positions[connection_pos] = channel_position

    def get_two_body_channel_index(self, index_i: int, index_j: int) -> int:
        return self.two_body_state_channel_indices[self.get_two_body_connection_pos(index_i, index_j)]

    def get_two_body_channel_position(self, index_i: int, index_j: int) -> int:
        return self.two_body_state_channel_positions[self.get_two_body_connection_pos(index_i, index_j)]

    def minimum_det(self, n: int):
        if n > self.NMO:
            raise ValueError("error in minimum_det...")
        D0_indices = self.one_body_basis[:n]  # this is not very safe if one_body_basis is not sorted with sp energy
        D0 = Det(D0_indices, self.NMO)
        return D0

    def find_combination_2(self, n: int, x: int) -> int:
        for a in range(1, n):
            combinations_for_a = n - a
            if x <= combinations_for_a:
                b = a + x
                return (a, b)
            else:
                x -= combinations_for_a

        raise ValueError("Error in find_combination_2!!!")

    def single_excite(self, Di: Det) -> Tuple[int, int, int]:
        a = Di.find_nth(random.randint(1, self.particle_number))
        b = 0
        invp = 0
        channel = self.get_one_body_channel(a)
        for r in channel:
            if not Di.is_occupied(r):
                invp += 1
        if invp == 0:
            return (0, 0, 0)
        excite_to = random.randint(1, invp)
        for r in channel:
            if not Di.is_occupied(r):
                if excite_to > 1:
                    excite_to -= 1
                    continue
                else:
                    b = r
                    break
        invp = invp * self.particle_number
        return (a, b, invp)

    def double_excite(self, Di: Det) -> Tuple[int, int, int, int, int]:
        # a, b, c, d, invp = 0, 1, 2, 3, 1  # to delete
        num = self.particle_number
        two_body_conditions = int((num * (num - 1)) / 2)
        temp = random.randint(1, two_body_conditions)
        a_idx, b_idx = self.find_combination_2(num, temp)
        a = Di.find_nth(a_idx)
        b = Di.find_nth(b_idx)
        c = 0
        d = 0
        invp = 0
        channel = self.get_two_body_channel(a, b)
        for r, s in channel:
            if (not Di.is_occupied(r)) and (not Di.is_occupied(s)):
                invp += 1
        if invp == 0:
            return (0, 0, 0, 0, 0)
        excite_to = random.randint(1, invp)
        for r, s in channel:
            if (not Di.is_occupied(r)) and (not Di.is_occupied(s)):
                if excite_to > 1:
                    excite_to -= 1
                    continue
                else:
                    c = r
                    d = s
                    break
        invp = invp * two_body_conditions
        return (a, b, c, d, invp)
