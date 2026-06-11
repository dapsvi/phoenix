"""
Gravity (self-weight) solver.
"""

import numpy as np
from .structural import StructuralSolver


class GravitySolver(StructuralSolver):
    """Structural solver with self-weight (body force)"""

    ndof_per_node = 3

    # pre-defined directions (mm/s²)
    _DIRECTIONS = {
        "down":     (0, 0, -9810),  "-z": (0, 0, -9810),
        "up":       (0, 0,  9810),  "+z": (0, 0,  9810),
        "left":     (-9810, 0, 0),  "-x": (-9810, 0, 0),
        "right":    ( 9810, 0, 0),  "+x": ( 9810, 0, 0),
        "forward":  (0, -9810, 0),  "-y": (0, -9810, 0),
        "backward": (0,  9810, 0),  "+y": (0,  9810, 0),
    }

    def __init__(self, grid, volfrac=0.3, penal=3.0, rmin=2.5,
                 move=0.2, eta=0.5, nu=0.3, solver_kind="direct",
                 E=1.0, material_density=1.0,
                 gravity_direction=None, gravity_vector=None,
                 gravity_penal=1.0):
        self.nu = nu
        # Resolve gravity vector
        if gravity_vector is not None:
            self._g = tuple(float(x) for x in gravity_vector)
        elif gravity_direction is not None:
            key = gravity_direction.lower()
            if key not in self._DIRECTIONS:
                raise ValueError(
                    f"Unknown gravity direction {gravity_direction!r}.  "
                    f"Choose from: {list(self._DIRECTIONS)}")
            self._g = self._DIRECTIONS[key]
        else:
            self._g = None
        self._g_penal = gravity_penal

        # precompute per-voxel body force
        if self._g is not None:
            h = float(grid.spacing[0])
            self._voxel_weight = np.array(self._g) * (material_density * h**3)
        else:
            self._voxel_weight = np.zeros(3)

        super().__init__(grid, volfrac=volfrac, penal=penal,
                         rmin=rmin, move=move, eta=eta,
                         nu=nu, solver_kind=solver_kind,
                         E=E, material_density=material_density)

    # add body force to any existing loads

    def _build_rhs(self):
        """Point loads + self-weight contribution"""
        rhs_list = super()._build_rhs() # base point loads
        if np.allclose(self._voxel_weight, 0):
            return rhs_list

        # Add body force to each case
        g = self.grid
        nxy = (g.nx + 1) * (g.ny + 1)
        w = self._voxel_weight / 8.0 # distributed to 8 corner nodes

        ix = np.arange(g.nx)
        iy = np.arange(g.ny)
        iz = np.arange(g.nz)
        all_nodes = np.array(np.meshgrid(ix, iy, iz, indexing="ij")).reshape(3, -1)
        ix, iy, iz = all_nodes
        from .base import _voxel_nodes
        node_mat = _voxel_nodes(ix, iy, iz, g.nx, nxy)  # (nele, 8)

        for case_idx in range(len(rhs_list)):
            f_body = np.zeros(3 * (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
            np.add.at(f_body, 3 * node_mat.ravel(),     np.tile(w[0], g.nele * 8))
            np.add.at(f_body, 3 * node_mat.ravel() + 1, np.tile(w[1], g.nele * 8))
            np.add.at(f_body, 3 * node_mat.ravel() + 2, np.tile(w[2], g.nele * 8))
            rhs_list[case_idx] = rhs_list[case_idx] + f_body

        return rhs_list

    # evaluate: structural compliance (body force handled by _build_rhs)

    def evaluate(self, density, rho_min=1e-3):
        """Structural compliance including self-weight"""
        return super().evaluate(density, rho_min=rho_min)
