from .profiler import *
from .basis import *
from .hamiltonian import *


class MBPT:
    def __init__(self, basis: Basis, hamil: Hamiltonian):
        self.basis: Basis = basis
        self.hamiltonian: Hamiltonian = hamil
        self.NMO: int = self.basis.NMO
        self.particle_number: int = basis.particle_number
        self.hole_states: List[int] = []
        self.particle_states: List[int] = []
        self.build_hole_particle()

    def build_hole_particle(self):
        all_states = self.basis.one_body_basis.copy()
        all_states.sort(key=lambda a: self.basis.get_orbit(a).e)
        self.hole_states = all_states[: self.particle_number]
        self.particle_states = all_states[self.particle_number :]

    def h0(self, p: int, q: int) -> float:
        if p == q:
            return self.basis.get_orbit(p).e
        else:
            return 0.0

    def assym(self, p: int, q: int, r: int, s: int) -> float:
        orbit_p = self.basis.get_orbit(p)
        orbit_q = self.basis.get_orbit(q)
        orbit_r = self.basis.get_orbit(r)
        orbit_s = self.basis.get_orbit(s)
        return self.hamiltonian.cal_v2mat(orbit_p, orbit_q, orbit_r, orbit_s)

    def f(self, p: int, q: int) -> float:
        s = self.h0(p, q)
        for i in self.hole_states:
            s += self.assym(p, i, q, i)
        return s

    def eps(self, holes, particles) -> float:
        E = 0.0
        for h in holes:
            E += self.f(h, h)
        for p in particles:
            E -= self.f(p, p)
        return E

    def cal_s1(self) -> float:
        s1 = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for i in self.hole_states:
                    for j in self.hole_states:
                        s1 += 0.25 * self.assym(a, b, i, j) * self.assym(i, j, a, b) / self.eps((i, j), (a, b))
        return s1

    def cal_s3(self) -> float:
        s3 = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for i in self.hole_states:
                        for j in self.hole_states:
                            for k in self.hole_states:
                                s3 += self.assym(i, j, a, b) * self.assym(a, c, j, k) * self.assym(b, k, c, i) / self.eps((i, j), (a, b)) / self.eps((k, j), (a, c))
        return s3

    def cal_s4(self) -> float:
        s4 = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                s4 += 0.125 * self.assym(i, j, a, b) * self.assym(a, b, c, d) * self.assym(c, d, i, j) / self.eps((i, j), (a, b)) / self.eps((i, j), (c, d))
        return s4

    def cal_s5(self) -> float:
        s5 = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for i in self.hole_states:
                    for j in self.hole_states:
                        for k in self.hole_states:
                            for l in self.hole_states:
                                s5 += 0.125 * self.assym(i, j, a, b) * self.assym(k, l, i, j) * self.assym(a, b, k, l) / self.eps((i, j), (a, b)) / self.eps((k, l), (a, b))
        return s5

    def cal_q5(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for e in self.particle_states:
                            for f in self.particle_states:
                                for i in self.hole_states:
                                    for j in self.hole_states:
                                        q += 1.0 / 16.0 * self.assym(i, j, a, b) * self.assym(a, b, c, d) * self.assym(c, d, e, f) * self.assym(e, f, i, j) / self.eps((i, j), (a, b)) / self.eps((i, j), (c, d)) / self.eps((i, j), (e, f))
        return q

    def cal_q6(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q += 1.0 / 16.0 * self.assym(i, j, a, b) * self.assym(a, b, c, d) * self.assym(k, l, i, j) * self.assym(c, d, k, l) / self.eps((i, j), (a, b)) / self.eps((i, j), (c, d)) / self.eps((k, l), (c, d))
        return q

    def cal_q7(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q += 1.0 / 16.0 * self.assym(i, j, a, b) * self.assym(k, l, i, j) * self.assym(a, b, c, d) * self.assym(c, d, k, l) / self.eps((i, j), (a, b)) / self.eps((k, l), (a, b)) / self.eps((k, l), (c, d))
        return q

    def cal_q8(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for i in self.hole_states:
                    for j in self.hole_states:
                        for k in self.hole_states:
                            for l in self.hole_states:
                                for m in self.hole_states:
                                    for n in self.hole_states:
                                        q += 1.0 / 16.0 * self.assym(i, j, a, b) * self.assym(k, l, i, j) * self.assym(m, n, k, l) * self.assym(a, b, m, n) / self.eps((i, j), (a, b)) / self.eps((k, l), (a, b)) / self.eps((m, n), (a, b))
        return q

    def cal_q9(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for e in self.particle_states:
                            for i in self.hole_states:
                                for j in self.hole_states:
                                    for k in self.hole_states:
                                        q -= 0.5 * self.assym(i, j, a, b) * self.assym(a, b, c, d) * self.assym(k, d, i, e) * self.assym(c, e, k, j) / self.eps((i, j), (a, b)) / self.eps((i, j), (c, d)) / self.eps((j, k), (c, e))
        return q

    def cal_q10(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for e in self.particle_states:
                            for i in self.hole_states:
                                for j in self.hole_states:
                                    for k in self.hole_states:
                                        q -= 0.5 * self.assym(i, j, a, b) * self.assym(k, b, i, c) * self.assym(a, c, d, e) * self.assym(d, e, k, j) / self.eps((i, j), (a, b)) / self.eps((j, k), (a, c)) / self.eps((j, k), (d, e))
        return q

    def cal_q11(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for i in self.hole_states:
                        for j in self.hole_states:
                            for k in self.hole_states:
                                for l in self.hole_states:
                                    for m in self.hole_states:
                                        q -= 0.5 * self.assym(i, j, a, b) * self.assym(k, l, i, j) * self.assym(a, m, c, l) * self.assym(c, b, k, m) / self.eps((i, j), (a, b)) / self.eps((k, l), (a, b)) / self.eps((k, m), (b, c))
        return q

    def cal_q12(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for i in self.hole_states:
                        for j in self.hole_states:
                            for k in self.hole_states:
                                for l in self.hole_states:
                                    for m in self.hole_states:
                                        q -= 0.5 * self.assym(i, j, a, b) * self.assym(a, k, c, j) * self.assym(l, m, i, k) * self.assym(c, b, l, m) / self.eps((i, j), (a, b)) / self.eps((i, k), (b, c)) / self.eps((l, m), (b, c))
        return q

    def cal_q13(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q += self.assym(i, j, a, b) * self.assym(a, k, c, j) * self.assym(c, l, d, k) * self.assym(d, b, i, l) / self.eps((i, j), (a, b)) / self.eps((i, k), (b, c)) / self.eps((i, l), (b, d))
        return q

    def cal_q14(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q -= self.assym(i, j, a, b) * self.assym(k, b, c, j) * self.assym(c, l, i, d) * self.assym(a, d, k, l) / self.eps((i, j), (a, b)) / self.eps((i, k), (a, c)) / self.eps((k, l), (a, d))
        return q

    def cal_q15(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q -= self.assym(i, j, a, b) * self.assym(k, b, c, j) * self.assym(a, l, k, d) * self.assym(c, d, i, l) / self.eps((i, j), (a, b)) / self.eps((i, k), (a, c)) / self.eps((i, l), (c, d))
        return q

    def cal_q16(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q += self.assym(i, j, a, b) * self.assym(k, b, i, c) * self.assym(a, l, d, j) * self.assym(d, c, k, l) / self.eps((i, j), (a, b)) / self.eps((j, k), (a, c)) / self.eps((k, l), (c, d))
        return q

    def cal_q17(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for e in self.particle_states:
                            for i in self.hole_states:
                                for j in self.hole_states:
                                    for k in self.hole_states:
                                        q -= 0.5 * self.assym(i, j, a, b) * self.assym(a, k, c, d) * self.assym(c, b, e, k) * self.assym(e, d, i, j) / self.eps((i, j), (a, b)) / self.eps((i, j, k), (b, c, d)) / self.eps((i, j), (d, e))
        return q

    def cal_q18(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for e in self.particle_states:
                            for i in self.hole_states:
                                for j in self.hole_states:
                                    for k in self.hole_states:
                                        q -= 0.5 * self.assym(i, j, a, b) * self.assym(a, k, c, d) * self.assym(c, d, e, j) * self.assym(e, b, i, k) / self.eps((i, j), (a, b)) / self.eps((i, j, k), (b, c, d)) / self.eps((i, k), (b, e))
        return q

    def cal_q19(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for i in self.hole_states:
                        for j in self.hole_states:
                            for k in self.hole_states:
                                for l in self.hole_states:
                                    for m in self.hole_states:
                                        q -= 0.5 * self.assym(i, j, a, b) * self.assym(k, l, i, c) * self.assym(m, b, k, l) * self.assym(a, c, m, j) / self.eps((i, j), (a, b)) / self.eps((j, k, l), (a, b, c)) / self.eps((j, m), (a, c))
        return q

    def cal_q20(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for i in self.hole_states:
                        for j in self.hole_states:
                            for k in self.hole_states:
                                for l in self.hole_states:
                                    for m in self.hole_states:
                                        q -= 0.5 * self.assym(i, j, a, b) * self.assym(k, l, i, c) * self.assym(m, c, k, j) * self.assym(a, b, m, l) / self.eps((i, j), (a, b)) / self.eps((j, k, l), (a, b, c)) / self.eps((m, l), (a, b))
        return q

    def cal_q21(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for e in self.particle_states:
                            for i in self.hole_states:
                                for j in self.hole_states:
                                    for k in self.hole_states:
                                        q += self.assym(i, j, a, b) * self.assym(a, k, c, d) * self.assym(c, b, e, j) * self.assym(e, d, i, k) / self.eps((i, j), (a, b)) / self.eps((i, j, k), (b, c, d)) / self.eps((i, k), (d, e))
        return q

    def cal_q22(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for e in self.particle_states:
                            for i in self.hole_states:
                                for j in self.hole_states:
                                    for k in self.hole_states:
                                        q += 0.25 * self.assym(i, j, a, b) * self.assym(a, k, c, d) * self.assym(c, d, e, k) * self.assym(e, b, i, j) / self.eps((i, j), (a, b)) / self.eps((i, j, k), (b, c, d)) / self.eps((i, j), (b, e))
        return q

    def cal_q23(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for i in self.hole_states:
                        for j in self.hole_states:
                            for k in self.hole_states:
                                for l in self.hole_states:
                                    for m in self.hole_states:
                                        q += 0.25 * self.assym(i, j, a, b) * self.assym(k, l, i, c) * self.assym(m, c, k, l) * self.assym(a, b, m, j) / self.eps((i, j), (a, b)) / self.eps((j, k, l), (a, b, c)) / self.eps((j, m), (a, b))
        return q

    def cal_q24(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for i in self.hole_states:
                        for j in self.hole_states:
                            for k in self.hole_states:
                                for l in self.hole_states:
                                    for m in self.hole_states:
                                        q += self.assym(i, j, a, b) * self.assym(k, l, i, c) * self.assym(m, b, k, j) * self.assym(a, c, m, l) / self.eps((i, j), (a, b)) / self.eps((j, k, l), (a, b, c)) / self.eps((l, m), (a, c))
        return q

    def cal_q26(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q -= self.assym(i, j, a, b) * self.assym(a, k, c, d) * self.assym(l, d, i, k) * self.assym(c, b, l, j) / self.eps((i, j), (a, b)) / self.eps((i, j, k), (b, c, d)) / self.eps((j, l), (b, c))
        return q

    def cal_q27(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q -= self.assym(i, j, a, b) * self.assym(k, l, i, c) * self.assym(a, c, d, l) * self.assym(d, b, k, j) / self.eps((i, j), (a, b)) / self.eps((j, k, l), (a, b, c)) / self.eps((j, k), (b, d))
        return q

    def cal_q29(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q += 0.5 * self.assym(i, j, a, b) * self.assym(a, k, c, d) * self.assym(l, b, i, k) * self.assym(c, d, l, j) / self.eps((i, j), (a, b)) / self.eps((i, j, k), (b, c, d)) / self.eps((j, l), (c, d))
        return q

    def cal_q30(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q += 0.5 * self.assym(i, j, a, b) * self.assym(k, l, i, c) * self.assym(a, b, d, l) * self.assym(d, c, k, j) / self.eps((i, j), (a, b)) / self.eps((j, k, l), (a, b, c)) / self.eps((j, k), (c, d))
        return q

    def cal_q31(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q += 0.5 * self.assym(i, j, a, b) * self.assym(k, l, i, c) * self.assym(a, c, d, j) * self.assym(d, b, k, l) / self.eps((i, j), (a, b)) / self.eps((j, k, l), (a, b, c)) / self.eps((k, l), (b, d))
        return q

    def cal_q32(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q += 0.5 * self.assym(i, j, a, b) * self.assym(a, k, c, d) * self.assym(l, d, i, j) * self.assym(c, b, l, k) / self.eps((i, j), (a, b)) / self.eps((i, j, k), (b, c, d)) / self.eps((k, l), (b, c))
        return q

    def cal_q40(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q += 0.5 * self.assym(i, j, a, b) * self.assym(k, l, c, d) * self.assym(c, b, i, l) * self.assym(a, d, k, j) / self.eps((i, j), (a, b)) / self.eps((i, l), (b, c)) / self.eps((j, k), (a, d))
        return q

    def cal_q41(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q += 1.0 / 16.0 * self.assym(i, j, a, b) * self.assym(k, l, c, d) * self.assym(c, d, i, j) * self.assym(a, b, k, l) / self.eps((i, j), (a, b)) / self.eps((i, j), (c, d)) / self.eps((k, l), (a, b))
        return q

    def cal_q42(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q -= 1.0 / 4.0 * self.assym(i, j, a, b) * self.assym(k, l, c, d) * self.assym(c, b, i, j) * self.assym(a, d, k, l) / self.eps((i, j), (a, b)) / self.eps((i, j), (b, c)) / self.eps((k, l), (a, d))
        return q

    def cal_q43(self) -> float:
        q = 0.0
        for a in self.particle_states:
            for b in self.particle_states:
                for c in self.particle_states:
                    for d in self.particle_states:
                        for i in self.hole_states:
                            for j in self.hole_states:
                                for k in self.hole_states:
                                    for l in self.hole_states:
                                        q -= 1.0 / 4.0 * self.assym(i, j, a, b) * self.assym(k, l, c, d) * self.assym(c, d, i, l) * self.assym(a, b, k, j) / self.eps((i, j), (a, b)) / self.eps((j, k), (a, b)) / self.eps((i, l), (c, d))
        return q

    def cal_coor2(self, g: float):
        self.hamiltonian.g = g
        return self.cal_s1()

    def cal_coor3(self, g: float):
        self.hamiltonian.g = g
        return self.cal_s3() + self.cal_s4() + self.cal_s5()

    def cal_coor4(self, g: float):
        self.hamiltonian.g = g
        return (
            self.cal_q5()
            + self.cal_q6()
            + self.cal_q7()
            + self.cal_q8()
            + self.cal_q9()
            + self.cal_q10()
            + self.cal_q11()
            + self.cal_q12()
            + self.cal_q13()
            + self.cal_q14()
            + self.cal_q15()
            + self.cal_q16()
            + self.cal_q17()
            + self.cal_q18()
            + self.cal_q19()
            + self.cal_q20()
            + self.cal_q21()
            + self.cal_q22()
            + self.cal_q23()
            + self.cal_q24()
            + self.cal_q26()
            + self.cal_q27()
            + self.cal_q29()
            + self.cal_q30()
            + self.cal_q31()
            + self.cal_q32()
            + self.cal_q40()
            + self.cal_q41()
            + self.cal_q42()
            + self.cal_q43()
        )
