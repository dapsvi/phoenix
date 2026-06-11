"""
MaxLengthSolver: local-volume constraint wrapper
"""

import numpy as np
from scipy.ndimage import convolve


# Ball kernel: 1 inside radius, 0 outside

def _ball_kernel(radius):
    """Binary ball/sphere kernel of radius radius"""
    r = int(np.ceil(radius))
    kx, ky, kz = np.meshgrid(
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        indexing="ij",
    )
    dist = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    return (dist <= radius).astype(np.float64)


class MaxLengthSolver:
    """
    Wrapper that adds a local-volume (max-length-scale) constraint to any solver's sensitivity gradient

    Parameters
    ----------
    grid : Grid
    solver : BaseSolver
        The wrapped physics solver
    r_max : float or None
        Ball-kernel radius in voxels
        None disables the penalty
    r_max_alpha : float
        Local volume cap multiplier
    r_max_weight : float
        Penalty strength: higher = more aggressive thinning
    """

    def __init__(self, grid, solver, *,
                 r_max=4.0, r_max_alpha=1.5, r_max_weight=5.0):
        self.grid = grid
        self.solver = solver
        self.r_max = r_max
        self.r_max_alpha = r_max_alpha
        self.r_max_weight = r_max_weight

        # Build kernel once
        self._max_kernel = _ball_kernel(r_max) if r_max is not None else None

        # Forward common properties from the wrapped solver
        for attr in ("volfrac", "rmin", "move", "eta",
                      "solver_kind", "E", "material_density",
                      "_kernel", "ndof_per_node", "ndof",
                      "_edof", "_rhs", "_fixed", "_KE",
                      "_history", "_snapshots"):
            if hasattr(solver, attr):
                setattr(self, attr, getattr(solver, attr))

        if hasattr(solver, "history"):
            self._history = solver._history

    # penal proxy to keep wrapper and inner solver in sync

    @property
    def penal(self):
        return self.solver.penal

    @penal.setter
    def penal(self, v):
        self.solver.penal = v

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    # physics hooks to delegate to wrapped solver

    def _element_matrix(self):
        return self.solver._element_matrix()

    def _build_rhs(self):
        return self.solver._build_rhs()

    def _build_fixed(self):
        return self.solver._build_fixed()

    def _assemble_stiffness(self, density, rho_min=1e-3):
        return self.solver._assemble_stiffness(density, rho_min=rho_min)

    def _solve(self, K, f, active_mask):
        return self.solver._solve(K, f, active_mask)

    def _compute_sensitivity(self, density, u, rho_min=1e-3):
        return self.solver._compute_sensitivity(density, u, rho_min=rho_min)

    # evaluate: base physics + local-volume penalty

    def evaluate(self, density, rho_min=1e-3):
        """Base physics sensitivity, multiplicatively suppressed where the neighbourhood-average density exceeds the cap"""
        obj, sens = self.solver.evaluate(density, rho_min=rho_min)

        if self._max_kernel is None:
            return obj, sens

        # compute local density
        ones = np.ones_like(density)
        num = convolve(density, self._max_kernel, mode="constant", cval=0.0)
        den = convolve(ones, self._max_kernel, mode="constant", cval=0.0)
        local_vol = num / np.maximum(den, 1e-10)

        # cap: volfrac × alpha
        alpha = self.volfrac * self.r_max_alpha
        excess = np.maximum(0.0, local_vol - alpha)

        # multiplicative suppression
        self._suppression = np.exp(-self.r_max_weight * excess)
        sens = sens * self._suppression

        return obj, sens

    def compute_suppression(self, density):
        """Compute the multiplicative suppression field WITHOUT running the full FE solve"""
        if self._max_kernel is None:
            self._suppression = None
            return None

        ones = np.ones_like(density)
        num = convolve(density, self._max_kernel, mode="constant", cval=0.0)
        den = convolve(ones, self._max_kernel, mode="constant", cval=0.0)
        local_vol = num / np.maximum(den, 1e-10)

        alpha = self.volfrac * self.r_max_alpha
        excess = np.maximum(0.0, local_vol - alpha)

        self._suppression = np.exp(-self.r_max_weight * excess)
        return self._suppression

    # optimize: delegate to shared SIMP loop

    def optimize(self, max_iter=60, tol=0.01, verbose=True,
                 callback=None, rho_min=1e-3, final="best",
                 snapshot_interval=0, history_key="compliance"):
        from .base import run_simp_loop
        return run_simp_loop(
            self, max_iter=max_iter, tol=tol, verbose=verbose,
            callback=callback, rho_min=rho_min, final=final,
            snapshot_interval=snapshot_interval,
            history_key=history_key, lower_is_better=True,
        )
