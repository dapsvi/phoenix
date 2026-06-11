"""
Shared stress-computation utilities.
"""

import numpy as np


def build_B_centroid(h):
    """6 by 24 strain-displacement matrix at the centroid of a unit-cube hex element with side length h"""
    signs = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ], dtype=np.float64)
    B = np.zeros((6, 24), dtype=np.float64)
    for i in range(8):
        xi, eta, zeta = signs[i]
        j = i * 3
        inv4h = 1.0 / (4.0 * h)
        B[0, j]     = xi * inv4h
        B[1, j + 1] = eta * inv4h
        B[2, j + 2] = zeta * inv4h
        B[3, j]     = eta * inv4h
        B[3, j + 1] = xi * inv4h
        B[4, j + 1] = zeta * inv4h
        B[4, j + 2] = eta * inv4h
        B[5, j]     = zeta * inv4h
        B[5, j + 2] = xi * inv4h
    return B


def build_D(E, nu):
    """6 by 6 isotropic linear-elastic constitutive matrix"""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6), dtype=np.float64)
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def von_mises(stress_voigt):
    """Compute stress from (n, 6) Voigt stress array"""
    s = stress_voigt
    return np.sqrt(
        s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2
        - s[:, 0]*s[:, 1] - s[:, 1]*s[:, 2] - s[:, 2]*s[:, 0]
        + 3.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)
    )
