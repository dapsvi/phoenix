"""
Thermal (heat-conduction) topology optimisation.

Minimises thermal compliance (maximises heat dissipation) subject
to a volume constraint.

The underlying element is the same 8-node hex; the DOF is a single
temperature per node, and the element "stiffness" is the 8 by 8
conductivity matrix.
"""

import numpy as np
from .base import BaseSolver, _voxel_nodes


# 8-node hex conductivity matrix (unit cube, k = 1)

def _lk_h8_thermal():
    """
    Return the 8 by 8 thermal conductivity matrix for a unit-cube
    8-node hex element with unit isotropic conductivity (k = 1).

    Computed with 2 by 2 by 2 Gaussian quadrature
    """
    gp = 1.0 / np.sqrt(3.0)           # Gauss-point coordinate
    gauss = np.array([-gp, gp])
    weights = np.ones(2)               # both = 1.0 for 2-pt rule

    KE = np.zeros((8, 8))

    for xi, wi in zip(gauss, weights):
        for eta, wj in zip(gauss, weights):
            for zeta, wk in zip(gauss, weights):
                # Shape-function derivatives in reference space
                dN_dxi = np.array([
                    -(1 - eta) * (1 - zeta),
                     (1 - eta) * (1 - zeta),
                     (1 + eta) * (1 - zeta),
                    -(1 + eta) * (1 - zeta),
                    -(1 - eta) * (1 + zeta),
                     (1 - eta) * (1 + zeta),
                     (1 + eta) * (1 + zeta),
                    -(1 + eta) * (1 + zeta),
                ]) / 8.0

                dN_deta = np.array([
                    -(1 - xi) * (1 - zeta),
                    -(1 + xi) * (1 - zeta),
                     (1 + xi) * (1 - zeta),
                     (1 - xi) * (1 - zeta),
                    -(1 - xi) * (1 + zeta),
                    -(1 + xi) * (1 + zeta),
                     (1 + xi) * (1 + zeta),
                     (1 - xi) * (1 + zeta),
                ]) / 8.0

                dN_dzeta = np.array([
                    -(1 - xi) * (1 - eta),
                    -(1 + xi) * (1 - eta),
                    -(1 + xi) * (1 + eta),
                    -(1 - xi) * (1 + eta),
                     (1 - xi) * (1 - eta),
                     (1 + xi) * (1 - eta),
                     (1 + xi) * (1 + eta),
                     (1 - xi) * (1 + eta),
                ]) / 8.0

                dN_dx = dN_dxi * 2.0
                dN_dy = dN_deta * 2.0
                dN_dz = dN_dzeta * 2.0

                B = np.column_stack([dN_dx, dN_dy, dN_dz])  # (8, 3)

                detJ = 1.0 / 8.0

                KE += (B @ B.T) * detJ * wi * wj * wk

    return KE



class ThermalSolver(BaseSolver):
    """
    Heat-conduction SIMP topology optimisation.

    The objective is thermal compliance, which the OC updater already
    minimises.  A lower thermal compliance means the structure
    conducts heat more efficiently.
    """

    ndof_per_node = 1

    def _element_matrix(self):
        return _lk_h8_thermal()

    # RHS: heat-flux vector

    def _build_rhs(self):
        """Distribute heat flux: returns a list, one vector per load case"""
        g = self.grid
        nxy = (g.nx + 1) * (g.ny + 1)
        rhs_list = []
        for case in g.load_cases:
            q = np.zeros(self.ndof)
            for mask, direction in case:
                qx, qy, qz = map(float, direction)
                if abs(qx) + abs(qy) + abs(qz) < 1e-30:
                    continue
                ix, iy, iz = np.where(mask)
                if len(ix) == 0:
                    continue
                nodes = _voxel_nodes(ix, iy, iz, g.nx, nxy).ravel()
                w = 1.0 / 8.0
                np.add.at(q, nodes, (qx + qy + qz) * w)
            rhs_list.append(q)
        return rhs_list

    # BCs: fixed temperatures

    def _build_fixed(self):
        """Fixed-temperature DOFs from the support mask."""
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
            # All three constraint types fix the single temperature DOF (only one DOF per node)
            all_fixed.append(nodes)

        return np.unique(np.concatenate(all_fixed)).astype(np.int32)
