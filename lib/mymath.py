import numpy as np


# i^(2n)
def iphase_double(n: int) -> float:
    if n % 2 == 0:
        return 1.0
    else:
        return -1.0


def sign(x: float) -> float:
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    else:
        return 0.0


def delta(i: int, j: int) -> float:
    if i == j:
        return 1.0
    else:
        return 0.0
