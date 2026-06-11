"""
Grid: central data object for topology optimization.
"""

import numpy as np


class Grid:
    def __init__(self, nx, ny, nz, initial_density=0.3,
                 spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
                 symmetry=None):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.spacing = tuple(spacing)
        self.origin = tuple(origin)
        self.density = np.full((nx, ny, nz), initial_density, dtype=np.float64)

        # load_cases: list of [(mask, direction), ...], one list per case
        self.load_cases = [[]]

        self.support_mask = np.zeros((nx, ny, nz), dtype=bool)
        self.support_constraint_voxel = np.full((nx, ny, nz), "", dtype=object)

        self.solid_mask = np.zeros((nx, ny, nz), dtype=bool)
        self.void_mask = np.zeros((nx, ny, nz), dtype=bool)

        # Symmetry: None, "x", "y", "z", "xy", "xz", "yz", "xyz"
        self.symmetry = symmetry


    @property
    def loads(self):
        """Backward-compat alias for the first load case"""
        if not self.load_cases:
            self.load_cases.append([])
        return self.load_cases[0]

    @loads.setter
    def loads(self, value):
        self.load_cases[0] = list(value)

    @property
    def load_mask(self):
        first = self.loads
        if not first:
            return np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        mask = first[0][0].copy()
        for m, _ in first[1:]:
            mask |= m
        return mask

    @property
    def load_direction(self):
        """Direction of the first load"""
        return self.loads[0][1] if self.loads else (0, -1, 0)


    @property
    def shape(self):
        return (self.nx, self.ny, self.nz)

    @property
    def nele(self):
        return self.nx * self.ny * self.nz

    @property
    def ndof(self):
        return 3 * (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    @property
    def volume_fraction(self):
        return float(self.density.sum() / self.nele)

    def add_load(self, load, case=0):
        """Add a Load to the grid, optionally into a specific case"""
        while len(self.load_cases) <= case:
            self.load_cases.append([])
        self.load_cases[case].append(
            (load.to_mask(self.nx, self.ny, self.nz), load.direction))

    def add_support(self, support):
        mask, constraint = support.to_mask(self.nx, self.ny, self.nz)
        self.support_mask |= mask
        self.support_constraint_voxel[mask] = constraint

    def upsample(self, factor):
        new = Grid(self.nx * factor, self.ny * factor, self.nz * factor,
                   initial_density=0.0,
                   spacing=self.spacing, origin=self.origin,
                   symmetry=self.symmetry)
        new.density = np.repeat(
            np.repeat(np.repeat(self.density, factor, axis=0), factor, axis=1),
            factor, axis=2,
        )
        # Replicate each load case
        new.load_cases = []
        for case in self.load_cases:
            new_case = []
            for mask, direction in case:
                up_mask = np.repeat(
                    np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1),
                    factor, axis=2,
                )
                new_case.append((up_mask, direction))
            new.load_cases.append(new_case)
        new.support_mask = np.repeat(
            np.repeat(np.repeat(self.support_mask, factor, axis=0), factor, axis=1),
            factor, axis=2,
        )
        new.support_constraint_voxel = np.repeat(
            np.repeat(np.repeat(self.support_constraint_voxel, factor, axis=0), factor, axis=1),
            factor, axis=2,
        )
        new.solid_mask = np.repeat(
            np.repeat(np.repeat(self.solid_mask, factor, axis=0), factor, axis=1),
            factor, axis=2,
        )
        new.void_mask = np.repeat(
            np.repeat(np.repeat(self.void_mask, factor, axis=0), factor, axis=1),
            factor, axis=2,
        )
        return new

    def filled(self, value=0.3):
        self.density.fill(value)
        return self

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def __repr__(self):
        sx, sy, sz = self.spacing
        return (f"Grid(nx={self.nx}, ny={self.ny}, nz={self.nz}, "
                f"size={self.nx*sx:.0f}×{self.ny*sy:.0f}×{self.nz*sz:.0f} mm, "
                f"vol={self.volume_fraction:.3f}, "
                f"loads={len(self.loads)}, "
                f"supports={self.support_mask.sum()})")