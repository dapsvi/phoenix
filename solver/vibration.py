"""
Vibration (natural-frequency) topology optimisation.

Maximises the first natural frequency of the structure.
"""

import numpy as np
from scipy.sparse import coo_matrix, block_diag

from .base import BaseSolver
from .structural import StructuralSolver


#  8-node hex consistent mass matrix

def _m_h8():
    """
    24 by 24 consistent mass matrix for an 8-node hexahedron.

    The 8 by 8 nodal mass sub-matrix (in units of 1/216) is repeated
    for each of the three displacement DOFs  (x, y, z)
    """
    m8 = (1.0 / 216.0) * np.array([
        [8, 4, 2, 4, 4, 2, 1, 2],
        [4, 8, 4, 2, 2, 4, 2, 1],
        [2, 4, 8, 4, 1, 2, 4, 2],
        [4, 2, 4, 8, 2, 1, 2, 4],
        [4, 2, 1, 2, 8, 4, 2, 4],
        [2, 4, 2, 1, 4, 8, 4, 2],
        [1, 2, 4, 2, 2, 4, 8, 4],
        [2, 1, 2, 4, 4, 2, 4, 8],
    ])
    return np.kron(np.eye(3), m8)



class VibrationSolver(StructuralSolver):
    """Maximise the first natural frequency."""

    ndof_per_node = 3

    def __init__(self, grid, volfrac=0.3, penal=3.0, rmin=2.5,
                 move=0.2, eta=0.5, nu=0.3, solver_kind="direct",
                 E=1.0, material_density=1.0,
                 min_frequency=None, freq_penalty_weight=1e-6,
                 activate_after_hz=None):
        self.nu = nu
        self.min_frequency = min_frequency           # (Hz) lower bound constraint
        self.freq_penalty_weight = freq_penalty_weight  # penalty strength
        self.activate_after_hz = activate_after_hz   # (Hz) only contribute above this frequency
        super().__init__(grid, volfrac=volfrac, penal=penal,
                         rmin=rmin, move=move, eta=eta,
                         nu=nu, solver_kind=solver_kind,
                         E=E, material_density=material_density)
        h = float(grid.spacing[0])
        self._ME = _m_h8() * (material_density * h**3)

    def optimize(self, max_iter=60, tol=0.01, verbose=True,
                 callback=None, rho_min=1e-3, final="best",
                 snapshot_interval=0):
        from .base import run_simp_loop
        return run_simp_loop(
            self, max_iter=max_iter, tol=tol, verbose=verbose,
            callback=callback, rho_min=rho_min, final=final,
            snapshot_interval=snapshot_interval,
            history_key="frequency", lower_is_better=True,
        )

    def report_metric(self):
        try:
            obj, _ = self.evaluate(self.grid.density)
            omega2 = max(-obj, 1e-30)
            f_hz = float(np.sqrt(omega2) / (2.0 * np.pi))
            return ("f₁", f_hz, "Hz")
        except Exception:
            return ("f₁", 0.0, "Hz")

    def _assemble_mass(self, density, rho_min=1e-3):
        """Assemble SIMP-interpolated mass matrix"""
        g = self.grid; ndof = self.ndof
        rho_flat = density.transpose(2, 1, 0).ravel()
        active = rho_flat >= rho_min
        edof_a = self._edof[active]
        rho_a = rho_flat[active]; n_a = active.sum()

        rows = np.broadcast_to(edof_a[:, :, None], (n_a, 24, 24)).ravel()
        cols = np.broadcast_to(edof_a[:, None, :], (n_a, 24, 24)).ravel()
        vals = (self._ME[None] * rho_a[:, None, None]).ravel()

        return coo_matrix((vals, (rows, cols)),
                          shape=(ndof, ndof)).tocsr()

    def _solve_eigenvalue(self, K, M, fixed, active_mask, k=4):
        """Solve for the k smallest eigenvalues"""
        from .eigen import generalized_eigen

        ndof = K.shape[0]
        active_dofs = np.unique(self._edof[active_mask].ravel())
        free = np.zeros(ndof, dtype=bool)
        free[active_dofs] = True; free[fixed] = False
        free_idx = np.where(free)[0]
        n_free = len(free_idx)
        if n_free == 0:
            return np.zeros(k), np.zeros((ndof, k))

        K_f = K[free_idx][:, free_idx]
        M_f = M[free_idx][:, free_idx]

        try:
            omega2_vals, phi_f = generalized_eigen(
                K_f, M_f, k=k, max_iter=30, tol=1e-6)
            omega2_vals = np.maximum(omega2_vals, 1e-30)
            phi = np.zeros((ndof, k))
            phi[free_idx, :] = phi_f
            for j in range(k):
                phi[:, j] /= max(np.sqrt(phi[:, j] @ M @ phi[:, j]), 1e-15)
        except Exception as exc:
            from warnings import warn
            warn(f'vibration eigen failed: {exc}')
            omega2_vals = np.full(k, 1e-30)
            phi = np.zeros((ndof, k))

        return omega2_vals, phi

    def _freq_sensitivity(self, density, phi, omega2, rho_min=1e-3):
        g = self.grid
        rho_flat = density.transpose(2, 1, 0).ravel()
        active = rho_flat >= rho_min
        edof_a = self._edof[active]
        rho_a = rho_flat[active]
        ME = _m_h8()

        phi_e = phi[edof_a]
        stiff = np.einsum('ij,jk,ik->i', phi_e, self._KE, phi_e)
        mass  = np.einsum('ij,jk,ik->i', phi_e, self._ME, phi_e)
        sens_e = self.penal * rho_a ** (self.penal - 1) * stiff - omega2 * mass

        sens = np.zeros(g.nele)
        sens[active] = np.maximum(sens_e, 1e-30)

        return sens.reshape((g.nx, g.ny, g.nz), order='F')

    def evaluate(self, density, rho_min=1e-3, ks_rho=10.0):
        """
        Full vibration analysis with KS-mode aggregation

        KS-aggregates the lowest k eigenfrequencies so the objective
        stays smooth when eigenvalues cross (prevents mode-switching
        2-cycles on near-symmetric domains)
        """
        rho_flat = density.transpose(2, 1, 0).ravel()
        active_mask = rho_flat >= rho_min

        K = self._assemble_stiffness(density, rho_min=rho_min)
        M = self._assemble_mass(density, rho_min=rho_min)

        omega2_vals, phi_mat = self._solve_eigenvalue(
            K, M, self._fixed, active_mask, k=4)

        # KS aggregate
        shifted = omega2_vals - omega2_vals.min()
        exp_terms = np.exp(-ks_rho * shifted)
        omega2_ks = omega2_vals.min() - np.log(exp_terms.mean()) / ks_rho

        # gate: don't contribute until the structure is stiff enough
        if self.activate_after_hz is not None:
            f_hz = np.sqrt(max(omega2_ks, 0.0)) / (2.0 * np.pi)
            if f_hz < self.activate_after_hz:
                return 0.0, np.zeros_like(density)

        # frequency constraint penalty
        penalty = 0.0
        if self.min_frequency is not None:
            f_target_hz = self.min_frequency
            omega2_target = (2 * np.pi * f_target_hz) ** 2
            if omega2_ks < omega2_target:
                deficit = omega2_target - omega2_ks
                penalty = self.freq_penalty_weight * deficit ** 2

        # KS weights for sensitivity blending
        weights = exp_terms / exp_terms.sum()

        # blended sensitivity
        sens = np.zeros_like(density)
        for j in range(len(omega2_vals)):
            if weights[j] < 1e-6:
                continue
            phi_j = phi_mat[:, j]
            w2_j = omega2_vals[j]
            sens_j = self._freq_sensitivity(density, phi_j, w2_j,
                                            rho_min=rho_min)
            sens += weights[j] * sens_j

        # apply constraint penalty to objective and sensitivity
        obj = -omega2_ks
        if penalty > 0:
            obj += penalty
            # Cap the factor to avoid NaN when the structure is very flexible
            factor = min(1.0 + 2.0 * self.freq_penalty_weight * (omega2_target - omega2_ks), 100.0)
            sens *= factor

        return obj, sens
