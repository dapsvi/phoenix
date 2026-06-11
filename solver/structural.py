"""
Structural (linear-elastic) topology optimisation.

Minimises compliance subject to a volume constraint.
"""

import numpy as np
from .base import BaseSolver, _voxel_nodes


# 8-node hexahedron stiffness matrix (SIMP, unit cube)

def _lk_h8(nu=0.3):
    """Return the (24,24) element stiffness matrix for an 8-node hex."""
    A = np.array([
        [32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
        [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12],
    ], dtype=float)
    kvec = (1.0 / 144.0) * A.T @ np.array([1.0, nu])
    k = kvec.ravel()

    K1 = np.array([
        [k[0], k[1], k[1], k[2], k[4], k[4]],
        [k[1], k[0], k[1], k[3], k[5], k[6]],
        [k[1], k[1], k[0], k[3], k[6], k[5]],
        [k[2], k[3], k[3], k[0], k[7], k[7]],
        [k[4], k[5], k[6], k[7], k[0], k[1]],
        [k[4], k[6], k[5], k[7], k[1], k[0]],
    ])
    K2 = np.array([
        [k[8], k[7], k[11], k[5], k[3], k[6]],
        [k[7], k[8], k[11], k[4], k[2], k[4]],
        [k[9], k[9], k[12], k[6], k[3], k[5]],
        [k[5], k[4], k[10], k[8], k[1], k[9]],
        [k[3], k[2], k[4], k[1], k[8], k[11]],
        [k[10], k[3], k[5], k[11], k[9], k[12]],
    ])
    K3 = np.array([
        [k[5], k[6], k[3], k[8], k[11], k[7]],
        [k[6], k[5], k[3], k[9], k[12], k[9]],
        [k[4], k[4], k[2], k[7], k[11], k[8]],
        [k[8], k[9], k[1], k[5], k[10], k[4]],
        [k[11], k[12], k[9], k[10], k[5], k[3]],
        [k[1], k[11], k[8], k[3], k[4], k[2]],
    ])
    K4 = np.array([
        [k[13], k[10], k[10], k[12], k[9], k[9]],
        [k[10], k[13], k[10], k[11], k[8], k[7]],
        [k[10], k[10], k[13], k[11], k[7], k[8]],
        [k[12], k[11], k[11], k[13], k[6], k[6]],
        [k[9], k[8], k[7], k[6], k[13], k[10]],
        [k[9], k[7], k[8], k[6], k[10], k[13]],
    ])
    K5 = np.array([
        [k[0], k[1], k[7], k[2], k[4], k[3]],
        [k[1], k[0], k[7], k[3], k[5], k[10]],
        [k[7], k[7], k[0], k[4], k[10], k[5]],
        [k[2], k[3], k[4], k[0], k[7], k[1]],
        [k[4], k[5], k[10], k[7], k[0], k[7]],
        [k[3], k[10], k[5], k[1], k[7], k[0]],
    ])
    K6 = np.array([
        [k[13], k[10], k[6], k[12], k[9], k[11]],
        [k[10], k[13], k[6], k[11], k[8], k[1]],
        [k[6], k[6], k[13], k[9], k[1], k[8]],
        [k[12], k[11], k[9], k[13], k[6], k[10]],
        [k[9], k[8], k[1], k[6], k[13], k[6]],
        [k[11], k[1], k[8], k[10], k[6], k[13]],
    ])
    KE = np.vstack([
        np.hstack([K1,  K2,      K3,      K4]),
        np.hstack([K2.T, K5,     K6,      K3.T]),
        np.hstack([K3.T, K6,     K5.T,    K2.T]),
        np.hstack([K4,   K3,     K2,      K1]),
    ])
    return KE / ((nu + 1.0) * (1.0 - 2.0 * nu))


class StructuralSolver(BaseSolver):
    """
    Linear-elastic SIMP topology optimisation

    Extra parameter
    ---------------
    nu : float
        Poisson ratio (default 0.3).
    """

    ndof_per_node = 3

    def __init__(self, grid, volfrac=0.3, penal=3.0, rmin=2.5,
                 move=0.2, eta=0.5, nu=0.3, solver_kind="direct",
                 E=1.0, material_density=1.0):
        self.nu = nu
        super().__init__(grid, volfrac=volfrac, penal=penal,
                         rmin=rmin, move=move, eta=eta,
                         solver_kind=solver_kind,
                         E=E, material_density=material_density)

    # element matrix

    def _element_matrix(self):
        return _lk_h8(self.nu)

    # RHS: force vector

    def _build_rhs(self):
        """Return a list of force vectors, one per load case"""
        g = self.grid
        nxy = (g.nx + 1) * (g.ny + 1)
        rhs_list = []
        for case in g.load_cases:
            f = np.zeros(self.ndof)
            for mask, direction in case:
                fx, fy, fz = map(float, direction)
                if abs(fx) + abs(fy) + abs(fz) < 1e-30:
                    continue
                ix, iy, iz = np.where(mask)
                if len(ix) == 0:
                    continue
                all_nodes = _voxel_nodes(ix, iy, iz, g.nx, nxy).ravel()
                w = 1.0 / 8.0
                np.add.at(f, 3 * all_nodes,     fx * w)
                np.add.at(f, 3 * all_nodes + 1, fy * w)
                np.add.at(f, 3 * all_nodes + 2, fz * w)
            rhs_list.append(f)
        return rhs_list

    # BCs: fixed displacements

    def _build_fixed(self):
        """Fixed DOFs from the support mask (per-voxel constraints)"""
        g = self.grid
        if not g.support_mask.any():
            return np.array([], dtype=np.int32)

        nxy = (g.nx + 1) * (g.ny + 1)
        constraints = g.support_constraint_voxel[g.support_mask]
        unique = np.unique(constraints)

        all_fixed = []
        for c in unique:
            mask = g.support_mask & (g.support_constraint_voxel == c)
            ix, iy, iz = np.where(mask)
            if len(ix) == 0:
                continue
            nodes = _voxel_nodes(ix, iy, iz, g.nx, nxy).ravel()
            c_str = str(c)
            if c_str == "fix":
                fixed = np.column_stack([
                    3 * nodes, 3 * nodes + 1, 3 * nodes + 2,
                ]).ravel()
            elif c_str == "fix_z":
                fixed = 3 * nodes + 2
            elif c_str == "fix_xy":
                fixed = np.column_stack([
                    3 * nodes, 3 * nodes + 1,
                ]).ravel()
            elif c_str == "fix_x":
                fixed = 3 * nodes
            elif c_str == "fix_y":
                fixed = 3 * nodes + 1
            else:
                raise ValueError(f"Unknown constraint: {c_str}")
            all_fixed.append(fixed)

        return np.unique(np.concatenate(all_fixed)).astype(np.int32)
