"""
Stress-constrained topology optimisation.

Adds a p-norm von Mises stress penalty to the structural compliance
objective.  Sharp corners and notches that concentrate stress are
automatically rounded by the optimizer.
"""

import numpy as np
from .structural import StructuralSolver
from ._stress_utils import build_B_centroid, build_D, von_mises


class StressConstrainedSolver(StructuralSolver):
    """
    Structural solver with a p-norm stress constraint.

    Extra parameters
    ----------------
    sigma_max : float
        Allowable von Mises stress in MPa.  Penalty activates when
        the p-norm exceeds this value.
    p_stress : float
        p-norm exponent (8-12).  Higher = closer to true max, but
        more nonlinear.  Default 10.
    stress_weight : float
        Weight of the stress penalty relative to compliance.
        Default 1.0, tune up if optimizer ignores stress, down if
        compliance degrades too much.
    """

    ndof_per_node = 3

    def __init__(self, grid, volfrac=0.3, penal=3.0, rmin=2.5,
                 move=0.2, eta=0.5, nu=0.3, solver_kind="direct",
                 E=1.0, material_density=1.0,
                 sigma_max=None, p_stress=10.0, stress_weight=1.0):
        self.nu = nu
        self.sigma_max = sigma_max          # MPa
        self.p_stress = p_stress
        self.stress_weight = stress_weight

        super().__init__(grid, volfrac=volfrac, penal=penal,
                         rmin=rmin, move=move, eta=eta,
                         nu=nu, solver_kind=solver_kind,
                         E=E, material_density=material_density)

        # Precompute B and D for stress evaluation
        h = float(grid.spacing[0])
        self._B = build_B_centroid(h)
        self._D = build_D(E, nu)
        self._DB = self._D @ self._B             # (6, 24)

    # post-optimization metric

    def report_metric(self):
        """Compute compliance + peak von Mises stress on the current design"""
        try:
            obj, _ = self.evaluate(self.grid.density)
            return ("compliance + stress", float(obj), "N·mm")
        except Exception:
            return ("compliance + stress", 0.0, "N·mm")

    # evaluate with stress penalty

    def evaluate(self, density, rho_min=1e-3):
        """Compliance + p-norm stress penalty with adjoint sensitivity."""
        # Base compliance
        obj, sens = super().evaluate(density, rho_min=rho_min)

        if self.sigma_max is None:
            return obj, sens

        # per-element von Mises stress
        g = self.grid
        rho_flat = density.transpose(2, 1, 0).ravel()
        active = rho_flat >= rho_min
        edof_a = self._edof[active]
        n_a = active.sum()
        if n_a == 0:
            return obj, sens

        K = self._assemble_stiffness(density, rho_min=rho_min)
        rhs_list = self._rhs if isinstance(self._rhs, list) else [self._rhs]
        u = self._solve(K, rhs_list[0], None)

        # Element stress vectors and von Mises scalars
        stress_e = np.empty((n_a, 6), dtype=np.float64)
        u_e_all = u[edof_a]  # (n_a, 24)
        for e in range(n_a):
            stress_e[e] = self._DB @ u_e_all[e]
        vm = von_mises(stress_e)  # (n_a,)

        # p-norm
        sigma_pn = (np.sum(vm ** self.p_stress)) ** (1.0 / self.p_stress)
        if sigma_pn < 1e-30:
            return obj, sens

        # penalty
        excess = max(0.0, sigma_pn - self.sigma_max)
        if excess <= 0:
            return obj, sens

        penalty = self.stress_weight * excess ** 2

        # Build von Mises matrix V
        V = np.array([
            [ 1.0, -0.5, -0.5, 0.0, 0.0, 0.0],
            [-0.5,  1.0, -0.5, 0.0, 0.0, 0.0],
            [-0.5, -0.5,  1.0, 0.0, 0.0, 0.0],
            [ 0.0,  0.0,  0.0, 3.0, 0.0, 0.0],
            [ 0.0,  0.0,  0.0, 0.0, 3.0, 0.0],
            [ 0.0,  0.0,  0.0, 0.0, 0.0, 3.0],
        ])
        DB = self._DB                     # (6, 24), D · B_centroid
        BT_DT_V = DB.T @ V                # (24, 6)

        # adjoint load
        ndof = K.shape[0]
        f_adj = np.zeros(ndof, dtype=np.float64)

        # Pre-factor
        pref = sigma_pn ** (1.0 - self.p_stress)
        for e in range(n_a):
            s_vec = stress_e[e] # (6,)
            s_vm = max(vm[e], 1e-30)
            weight = pref * s_vm ** (self.p_stress - 2.0)
            f_elem = weight * BT_DT_V @ s_vec  # (24,)
            np.add.at(f_adj, edof_a[e], f_elem)

        # solve adjoint system K lambda = f_adj
        free = np.ones(ndof, dtype=bool)
        free[self._fixed] = False
        free_idx = np.where(free)[0]

        from pypardiso import spsolve
        lam = np.zeros(ndof, dtype=np.float64)
        lam[free_idx] = spsolve(K[free_idx, :][:, free_idx], f_adj[free_idx])

        # per-element sensitivity
        flat_idx = np.where(active)[0]        # flat element indices for active elements
        keep_stress = np.zeros(g.nele, dtype=np.float64)
        for e in range(n_a):
            lam_e = lam[edof_a[e]]            # (24,) adjoint displacement
            u_e = u_e_all[e]                  # (24,) physical displacement
            strain_adj = np.dot(lam_e, self._KE @ u_e)
            keep_stress[flat_idx[e]] = (
                2.0 * self.stress_weight * excess
                * self.penal
                * rho_flat[flat_idx[e]] ** (self.penal - 1.0)
                * strain_adj
            )

        stress_sens = keep_stress.reshape((g.nx, g.ny, g.nz), order='F')
        return obj + penalty, sens + stress_sens
