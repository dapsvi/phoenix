"""
Convective-dissipation topology optimisation
"""

import numpy as np
from scipy.sparse import coo_matrix

from .base import _has_pyamg
from .thermal import ThermalSolver, _lk_h8_thermal


class ConvectionSolver(ThermalSolver):
    """
    Thermal topology optimisation with volumetric convection

    Parameters
    ----------
    h_conv : float
        Convection coefficient (higher = stronger cooling of void)
    penal_conv : float
        Penalisation exponent for the void / convection term
    """

    ndof_per_node = 1

    def __init__(self, grid, volfrac=0.3, penal=3.0, rmin=2.5,
                 move=0.2, eta=0.5, h_conv=1.0, penal_conv=3.0,
                 solver_kind="direct"):
        self.h_conv = h_conv
        self.penal_conv = penal_conv
        super().__init__(grid, volfrac=volfrac, penal=penal,
                         rmin=rmin, move=move, eta=eta,
                         solver_kind=solver_kind)

    def _solve(self, K, f, active_mask):
        """Solve without dead-DOF pruning because convection adds stiffness everywhere"""
        import numpy as np
        from scipy.sparse.linalg import cg as sp_cg
        ndof = K.shape[0]
        if isinstance(f, list):
            f = f[0]
        free = np.ones(ndof, dtype=bool)
        free[self._fixed] = False
        free_idx = np.where(free)[0]
        K_free = K[free_idx, :][:, free_idx]
        if self.solver_kind == "cg_amg" and _has_pyamg:
            import pyamg
            ml = pyamg.smoothed_aggregation_solver(K_free)
            M = ml.aspreconditioner()
            u_free, info = sp_cg(K_free, f[free_idx], M=M,
                                 atol=1e-8, rtol=1e-8)
            if info != 0:
                import warnings
                warnings.warn(f"CG did not converge (info={info})")
        else:
            from pypardiso import spsolve
            u_free = spsolve(K_free, f[free_idx])
        u = np.zeros(ndof)
        u[free_idx] = u_free
        return u

    # element matrix (conduction part only)

    def _element_matrix(self):
        """Return the 8 by 8 conduction matrix (same as ThermalSolver)"""
        return _lk_h8_thermal()

    # assembly: conduction + convection

    def _assemble_stiffness(self, density, rho_min=1e-3):
        """Assemble  K_cond(rho) + C_conv(rho)"""
        g = self.grid
        ndof = self.ndof

        # conduction part (inherited)
        K = super()._assemble_stiffness(density, rho_min=rho_min)

        # convection part
        rho_flat = density.transpose(2, 1, 0).ravel()
        void = rho_flat < (1.0 - rho_min) # "void" for convection
        if not void.any():
            return K

        rho_void = rho_flat[void]
        conv_factor = (1.0 - rho_void) ** self.penal_conv
        edof_v = self._edof[void] # (n_void, 8)

        # Lumped convection: each node gets h_conv / 8 * (1−rho_void)^q
        node_val = (self.h_conv / 8.0) * conv_factor  # (n_void,)
        rows = edof_v.ravel()
        cols = edof_v.ravel()
        vals = np.repeat(node_val, 8)

        # COO with duplicate entries: sums contributions from multiple elements to the same node automatically
        C = coo_matrix((vals, (rows, cols)),
                       shape=(ndof, ndof)).tocsr()

        return K + C

    # sensitivity: conduction + convection

    def _compute_sensitivity(self, density, u, rho_min=1e-3):
        """Combined sensitivity for conduction and convection terms"""
        g = self.grid
        rho_flat = density.transpose(2, 1, 0).ravel()

        # conduction term (inherited logic)
        active = rho_flat > rho_min
        edof_a = self._edof[active]
        u_e = u[edof_a]
        strain = np.sum(u_e * (self._KE @ u_e.T).T, axis=1)

        sens_flat = np.zeros(g.nele)
        sens_flat[active] = (self.penal
                             * (rho_flat[active] ** (self.penal - 1))
                             * strain)

        # convection term (opposite sign)
        void = rho_flat < (1.0 - rho_min)
        if void.any():
            edof_v = self._edof[void]
            u_v = u[edof_v] # (n_void, 8)
            conv_strain = (self.h_conv / 8.0) * np.sum(u_v ** 2, axis=1)

            sens_flat[void] -= (self.penal_conv
                                * ((1.0 - rho_flat[void]) ** (self.penal_conv - 1))
                                * conv_strain)

        # clip: the OC updater raises sens to a fractional power and cannot handle negative values
        # Elements where convection dominates stay at the lower bound
        sens_flat = np.maximum(sens_flat, 1e-30)

        return sens_flat.reshape((g.nx, g.ny, g.nz), order='F')
