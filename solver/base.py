"""
Base solver : shared optimization loop, filtering, and OC update.

To add a new physics, subclass BaseSolver and implement the following:

* ndof_per_node : int (3 for structural, 1 for thermal, etc)
* _element_matrix() : return the element matrix KE
* _build_rhs() : return the global right-hand-side vector
* _build_fixed() : return the array of fixed-DOF indices

Everything else (assembly, solve, sensitivity, filtering, OC update,
optimisation loop, continuation) is inherited
"""

import numpy as np
from pypardiso import spsolve
from scipy.sparse import coo_matrix
import scipy.sparse as sparse
from scipy.sparse.linalg import cg
from scipy.ndimage import convolve

try:
    import pyamg
    _has_pyamg = True
except ImportError:
    pyamg = None
    _has_pyamg = False


def _voxel_nodes(ix, iy, iz, nx, nxy):
    """Return (N, 8) array of the 8 corner-node indices for N voxels"""
    n0 = iz * nxy + iy * (nx + 1) + ix
    return np.column_stack([
        n0,
        n0 + 1,
        n0 + (nx + 1) + 1,
        n0 + (nx + 1),
        n0 + nxy,
        n0 + 1 + nxy,
        n0 + (nx + 1) + 1 + nxy,
        n0 + (nx + 1) + nxy,
    ])


def _cone_kernel(rmin):
    """Cone-shaped density filter kernel of radius rmin"""
    r = int(np.ceil(rmin))
    kx, ky, kz = np.meshgrid(
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        indexing="ij",
    )
    dist = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    return np.maximum(0.0, rmin - dist)


def _filter_density(density, kernel):
    """Density filter of formula ρ̄ = H ∗ ρ / H ∗ 1"""
    num = convolve(density, kernel, mode="constant", cval=0.0)
    den = convolve(np.ones_like(density), kernel, mode="constant", cval=0.0)
    return num / np.maximum(den, 1e-10)


def _heaviside(rho, beta, eta=0.5):
    """Smoothed Heaviside projection to push filtered density toward 0 or 1"""
    t1 = np.tanh(beta * eta)
    t2 = np.tanh(beta * (rho - eta))
    t3 = np.tanh(beta * (1.0 - eta))
    return (t1 + t2) / (t1 + t3)


def _heaviside_deriv(rho, beta, eta=0.5):
    """Derivative for the Heaviside projection obtained using the chain rule"""
    t1 = np.tanh(beta * eta)
    t3 = np.tanh(beta * (1.0 - eta))
    sech2 = 1.0 / np.cosh(beta * (rho - eta)) ** 2
    return beta * sech2 / (t1 + t3)



_HEAVISIDE_BETA_MAX = 8
_HEAVISIDE_BETA_STEP = 8
_HEAVISIDE_ETA = 0.5


def _volume_preserving_heaviside(rho, beta, target_volume, eta0=0.5):
    """Heaviside projection with beta chosen so sum(rho) is approximately target_volume"""
    rho_flat = rho.ravel()
    lo, hi = 0.01, 0.99
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        proj = _heaviside(rho_flat, beta, mid)
        total = proj.sum()
        if abs(total - target_volume) / max(target_volume, 1) < 1e-4:
            break
        if total > target_volume:
            lo = mid   # need higher threshold : fewer ones
        else:
            hi = mid   # need lower threshold : more ones
    return _heaviside(rho, beta, mid), mid


def _oc_update(density, filtered_sens, volfrac, move=0.2, eta=0.5, rho_min=0.001, free_mask=None, oc_damping=0.5):
    """
    Optimality-criteria update with optional free-element mask
    and damping.

    When free_mask is supplied, only those elements participate
    in the volume-target bisection and fixed-region elements are
    left untouched.
    """
    nele = density.size
    rho = density.ravel()
    sens = filtered_sens.ravel()

    if sens.max() < 1e-30:
        return density.copy()

    if free_mask is not None:
        active = free_mask.ravel()
        n_active = int(active.sum())
        if n_active == 0:
            return density.copy()
        target_volume = volfrac * n_active
    else:
        active = slice(None)
        target_volume = volfrac * nele

    rho_act = rho[active]
    sens_act = sens[active]

    l1, l2 = 0.0, sens_act.max() * 2.0
    rho_new_act = rho_act.copy()

    for _ in range(50):
        lmid = 0.5 * (l1 + l2)
        factor = (sens_act / max(lmid, 1e-10)) ** eta
        rho_try = rho_act * factor
        rho_try = np.clip(rho_try,
                          np.maximum(rho_min, rho_act - move),
                          np.minimum(1.0, rho_act + move))
        total = float(rho_try.sum())
        if abs(total - target_volume) < 1e-6:
            rho_new_act = rho_try
            break
        if total > target_volume:
            l1 = lmid
        else:
            l2 = lmid
        rho_new_act = rho_try

    rho_new = rho.copy()
    rho_new[active] = (1.0 - oc_damping) * rho[active] + oc_damping * rho_new_act
    # Re-clip to move limits after damping
    lo = np.maximum(rho_min, rho - move)
    hi = np.minimum(1.0, rho + move)
    rho_new = np.clip(rho_new, lo, hi)

    return rho_new.reshape(density.shape)


def _apply_fixed_regions(density, grid):
    density[grid.solid_mask] = 1.0
    density[grid.void_mask] = 0.0


def _apply_symmetry(density, grid):
    sym = grid.symmetry
    if not sym:
        return
    if isinstance(sym, dict):
        axes = [ch for ch, on in sym.items() if on]
    else:
        axes = list(sym)
    for ch in axes:
        ax = {"x": 0, "y": 1, "z": 2}[ch]
        density[:] = 0.5 * (density + np.flip(density, axis=ax))


def _effective_volfrac(grid, volfrac):
    """Return volfrac for free elements, accounting for fixed regions."""
    nele = grid.nele
    n_solid = int(grid.solid_mask.sum())
    n_void = int(grid.void_mask.sum())
    n_free = nele - n_solid - n_void
    if n_free <= 0:
        return volfrac  # degenerate
    return max(0.0, min(1.0, (volfrac * nele - n_solid) / n_free))


# Shared SIMP loop

def run_simp_loop(solver, max_iter=60, tol=0.01, verbose=True,
                  callback=None, rho_min=1e-3, final="best",
                  snapshot_interval=0,
                  evaluate_fn=None, history_key="compliance",
                  lower_is_better=True):
    """
    Single shared SIMP optimisation loop.

    All solvers (structural, thermal, vibration, multi) run through
    this function and only the evaluate callback varies.

    Parameters :
    solver : BaseSolver
        Provides grid, volfrac, penal, rmin, move, eta, _kernel
    evaluate_fn : callable
        (density, iteration) -> (objective, sensitivity)
        If None, solver.evaluate(density, rho_min=rho_min) is used
    history_key : str
        Key name for the objective in the history dict
        ("compliance", "frequency", "objective", etc)
    lower_is_better : bool
        True for compliance, False only if the objective is framed as maximisation.
    """
    if evaluate_fn is None:
        def evaluate_fn(d, it):
            return solver.evaluate(d, rho_min=rho_min)

    g = solver.grid
    design = g.density.copy()
    nele = g.nele
    history = {history_key: [], "volume": [], "change": []}
    snapshots = []

    volfrac_free = _effective_volfrac(g, solver.volfrac)

    free_mask = ~(g.solid_mask | g.void_mask)
    n_free = int(free_mask.sum())

    p_final = solver.penal
    best_obj = np.inf if lower_is_better else -np.inf
    best_design = None
    iters_at_beta_max = 0 # counter for convergence gating

    for it in range(max_iter):
        p_cur = min(p_final, 1.0 + it * (p_final - 1.0) / 10.0)
        solver.penal = p_cur

        beta = min(_HEAVISIDE_BETA_MAX, 2.0 ** (it // _HEAVISIDE_BETA_STEP))
        if beta >= _HEAVISIDE_BETA_MAX:
            iters_at_beta_max += 1
        else:
            iters_at_beta_max = 0

        # filter + volume-preserving projection
        rho_filt = _filter_density(design, solver._kernel)
        # Target free-element volume only
        target_vol = volfrac_free * n_free
        rho_phys, used_eta = _volume_preserving_heaviside(
            rho_filt, beta, target_vol)
        g.density = rho_phys.reshape((g.nx, g.ny, g.nz), order='F')
        _apply_fixed_regions(g.density, g)

        # snapshot
        if snapshot_interval > 0 and (it + 1) % snapshot_interval == 0:
            snapshots.append(g.density.copy())

        # evaluate with physical density
        obj, sens = evaluate_fn(g.density, it)

        # chain-rule through Heaviside
        dproj = _heaviside_deriv(rho_filt, beta, used_eta)
        sens = (sens.ravel(order='F') * dproj.ravel(order='F'))
        sens = sens.reshape((g.nx, g.ny, g.nz), order='F')

        # density-filter adjoint (NOT sensitivity filter)
        ones = np.ones_like(design)
        denom = convolve(ones, solver._kernel, mode="constant", cval=0.0)
        sens_filt = convolve(sens / np.maximum(denom, 1e-10),
                              solver._kernel, mode="constant", cval=0.0)

        # OC on design variable
        design_new = _oc_update(design, sens_filt, volfrac_free,
                                move=solver.move, eta=solver.eta,
                                free_mask=free_mask)

        # clamp fixed regions + enforce symmetry
        _apply_fixed_regions(design_new, g)
        _apply_symmetry(design_new, g)

        # convergence: mean absolute change
        change = np.abs(design_new - design).mean()
        phys_vol = rho_phys.sum() / nele

        # keep-best
        better = obj < best_obj if lower_is_better else obj > best_obj
        if better:
            best_obj = obj
            best_design = design.copy()

        history[history_key].append(obj)
        history["volume"].append(phys_vol)
        history["change"].append(change)

        if callback is not None:
            callback(g.density, it + 1, obj)

        if verbose:
            # Pick a unit label based on the physics
            if history_key == "frequency":
                unit = "Hz"
            elif history_key == "objective":
                unit = " "
            else:
                unit = "N·mm"   # compliance (N for forces, mm for spacing, etc)
            label = history_key[:5].rstrip("_")
            print(f"  iter {it + 1:3d}  |  p={p_cur:.1f}  β={beta:.1f}  "
                  f"{label} {obj:.4e} {unit}  |  "
                  f"vol {phys_vol:.3f}  |  Δρ̄ {change:.4f}")

        # stop when the design stops changing
        if iters_at_beta_max >= _HEAVISIDE_BETA_STEP // 2:
            recent = history["change"][-5:]
            if len(recent) >= 5 and max(recent) < tol:
                if verbose:
                    print(f"  Converged at iteration {it + 1}  "
                          f"(Δρ̄ < {tol:.3f} @ β={beta:.0f})")
                design = design_new
                break

        design = design_new

    solver.penal = p_final   # restore
    # final physical density
    if final == "latest":
        source = design
    else:
        source = best_design if best_design is not None else design
    rho_filt = _filter_density(source, solver._kernel)
    g.density, _ = _volume_preserving_heaviside(
        rho_filt, _HEAVISIDE_BETA_MAX, volfrac_free * n_free)
    g.density = g.density.reshape((g.nx, g.ny, g.nz), order='F')
    _apply_fixed_regions(g.density, g)
    if verbose and final == "latest":
        print(f"  Final physical density from last-iteration design")
    solver._history = history
    solver._snapshots = snapshots
    return g.density, history



class BaseSolver:
    """
    SIMP topology-optimisation loop that is independent of the
    underlying physics.

    Subclass attributes
    -------------------
    ndof_per_node : int
        Degrees of freedom per grid node (3 = structural, 1 = thermal)

    Subclass methods (must be overriden)
    ------------------------------------
    _element_matrix()
        Return the square element matrix KE (shape depends on physics)
    _build_rhs()
        Return the global right-hand-side vector (forces, flux, etc)
    _build_fixed()
        Return an array of fixed-DOF indices

    Inherited methods
    -----------------
    optimize(max_iter, tol, callback, rho_min)
    continuation(factor, coarse_iter, fine_iter, etc)
    """

    ndof_per_node = None # set by subclass

    def __init__(self, grid, volfrac=0.3, penal=3.0, rmin=2.5,
                 move=0.2, eta=0.5, solver_kind="direct",
                 E=1.0, material_density=1.0):
        if self.ndof_per_node is None:
            raise TypeError("subclass must set ndof_per_node")

        self.grid = grid
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.move = move
        self.eta = eta
        self.solver_kind = solver_kind
        self.E = E
        self.material_density = material_density

        # precompute and cache
        # KE_scaled = KE_unit × E × h
        spacing = float(grid.spacing[0])
        self._KE = self._element_matrix() * (E * spacing)
        self._edof = self._build_edof()
        self._rhs = self._build_rhs()
        self._fixed = self._build_fixed()
        self._kernel = _cone_kernel(rmin)
        self._history = {"compliance": [], "volume": [], "change": []}

        g = self.grid
        g.density[g.solid_mask] = 1.0
        g.density[g.void_mask] = 0.0

    # properties

    @property
    def history(self):
        return self._history

    @property
    def ndof(self):
        g = self.grid
        return self.ndof_per_node * (g.nx + 1) * (g.ny + 1) * (g.nz + 1)

    # element-DOF table
    # Generic (nx, ny, nz) maps to (nele, 8 × ndof_per_node)
    # Override this if your physics needs a different layout

    def _build_edof(self):
        """(nele, 8 by ndof_per_node) table mapping elements -> global DOFs"""
        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz
        d = self.ndof_per_node
        nxy = (nx + 1) * (ny + 1)

        ez = np.repeat(np.arange(nz), nx * ny)
        ey = np.tile(np.repeat(np.arange(ny), nx), nz)
        ex = np.tile(np.arange(nx), ny * nz)

        n0 = ez * nxy + ey * (nx + 1) + ex
        nodes = np.column_stack([
            n0, n0 + 1, n0 + (nx + 1) + 1, n0 + (nx + 1),
            n0 + nxy, n0 + 1 + nxy,
            n0 + (nx + 1) + 1 + nxy, n0 + (nx + 1) + nxy,
        ])  # (nele, 8)

        edof = np.zeros((len(n0), 8 * d), dtype=np.int32)
        for j in range(8):                     # for each corner node
            base_dof = d * nodes[:, j]          # first DOF of this node
            for i in range(d):                  # all d DOFs
                edof[:, j * d + i] = base_dof + i
        return edof

    # abstract methods

    def _element_matrix(self):
        raise NotImplementedError

    def _build_rhs(self):
        raise NotImplementedError

    def _build_fixed(self):
        raise NotImplementedError

    # assembly

    def _assemble_stiffness(self, density, rho_min=1e-3):
        E_MIN = 1e-6
        g = self.grid
        ndof = self.ndof
        d = self.ndof_per_node
        edof_size = 8 * d

        rho_flat = density.transpose(2, 1, 0).ravel()
        active = rho_flat >= rho_min
        rho_penal = (E_MIN + rho_flat[active] ** self.penal * (1.0 - E_MIN))
        edof_a = self._edof[active]
        n_a = active.sum()

        rows = np.broadcast_to(edof_a[:, :, None],
                               (n_a, edof_size, edof_size)).ravel()
        cols = np.broadcast_to(edof_a[:, None, :],
                               (n_a, edof_size, edof_size)).ravel()
        vals = (self._KE[None] * rho_penal[:, None, None]).ravel()

        K = coo_matrix((vals, (rows, cols)),
                       shape=(ndof, ndof)).tocsr()
        # E_min diagonal baseline prevents singularity
        K += E_MIN * sparse.eye(ndof, format='csr')
        return K

    # solve (with dead-DOF elimination)

    def _solve(self, K, f, active_mask):
        """Solve K·u = f with fixed BCs"""
        ndof = K.shape[0]
        if isinstance(f, list):
            f = f[0]

        free = np.ones(ndof, dtype=bool)
        free[self._fixed] = False
        free_idx = np.where(free)[0]
        K_free = K[free_idx, :][:, free_idx]

        # external preconditioner (pre-built for multi-case)
        M_ext = getattr(self, '_M_precond', None)
        if M_ext is not None:
            u_free, info = cg(K_free, f[free_idx], M=M_ext,
                              atol=1e-8, rtol=1e-8)
            if info != 0:
                import warnings
                warnings.warn(f"CG did not converge (info={info})")
        elif self.solver_kind == "cg_amg" and _has_pyamg:
            ml = pyamg.smoothed_aggregation_solver(K_free)
            M = ml.aspreconditioner()
            u_free, info = cg(K_free, f[free_idx], M=M,
                              atol=1e-8, rtol=1e-8)
            if info != 0:
                import warnings
                warnings.warn(f"CG did not converge (info={info})")
        elif self.solver_kind == "cg_amg" and not _has_pyamg:
            raise ImportError(
                "solver_kind='cg_amg' requires pyamg.  "
                "Install it with:  pip install pyamg")
        else:
            u_free = spsolve(K_free, f[free_idx])

        u = np.zeros(ndof)
        u[free_idx] = u_free
        return u

    # sensitivity

    def _compute_sensitivity(self, density, u, rho_min=1e-3):
        """Per-element sensitivity: penal * rho^(penal-1) * u_e^T KE u_e"""
        g = self.grid
        rho_flat = density.transpose(2, 1, 0).ravel()
        active = rho_flat >= rho_min
        edof_a = self._edof[active]

        u_e = u[edof_a]
        KE_u = self._KE @ u_e.T
        strain_energy = np.sum(u_e * KE_u.T, axis=1)

        sens_flat = np.zeros(g.nele)
        sens_flat[active] = (self.penal
                             * (rho_flat[active] ** (self.penal - 1))
                             * strain_energy)

        return sens_flat.reshape((g.nx, g.ny, g.nz), order='F')

    # evaluate

    def evaluate(self, density, rho_min=1e-3):
        """
        Run the full per-physics analysis

        The default implementation does a static compliance
        analysis. Override for eigenvalue, thermal, etc

        Returns
        -------
        objective : float
            Scalar objective value (compliance, etc)
        sensitivity : ndarray (nx, ny, nz)
            Element sensitivity field (unfiltered)
        """
        K = self._assemble_stiffness(density, rho_min=rho_min)

        # pre-build AMG preconditioner once per call
        if self.solver_kind == "cg_amg" and _has_pyamg:
            free = np.ones(K.shape[0], dtype=bool)
            free[self._fixed] = False
            free_idx = np.where(free)[0]
            K_free = K[free_idx, :][:, free_idx]
            ml = pyamg.smoothed_aggregation_solver(K_free)
            self._M_precond = ml.aspreconditioner()
        else:
            self._M_precond = None

        rhs_list = self._rhs if isinstance(self._rhs, list) else [self._rhs]
        n_cases = len(rhs_list)
        total_obj = 0.0
        combined_sens = np.zeros_like(density)

        for f in rhs_list:
            u = self._solve(K, f, None)
            total_obj += float(f @ u) / n_cases
            sens = self._compute_sensitivity(density, u, rho_min=rho_min)
            combined_sens += sens / n_cases

        self._M_precond = None  # clear
        return total_obj, combined_sens

    # optimisation loop

    def optimize(self, max_iter=60, tol=0.01, verbose=True,
                 callback=None, rho_min=1e-3, final="best",
                 snapshot_interval=0):
        """Thin wrapper that delegates to the shared SIMP loop."""
        return run_simp_loop(
            self, max_iter=max_iter, tol=tol, verbose=verbose,
            callback=callback, rho_min=rho_min, final=final,
            snapshot_interval=snapshot_interval,
            history_key="compliance", lower_is_better=True,
        )

    # post-optimization metric

    def report_metric(self):
        """Compute the solver's native metric on the current design"""
        try:
            obj, _ = self.evaluate(self.grid.density)
            return ("compliance", float(obj), "N·mm")
        except Exception:
            return ("compliance", 0.0, "N·mm")

    # continuation

    def continuation(self, factor=2, coarse_iter=40, fine_iter=30, rescale_rmin=True, verbose=True):
        """Two-stage refinement: coarse solve > upsample > fine solve"""
        g = self.grid
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  STAGE 1 : Coarse ({g.nx}×{g.ny}×{g.nz})")
            print(f"{'=' * 60}")

        coarse_result, coarse_hist = self.optimize(
            max_iter=coarse_iter, verbose=verbose)

        if verbose:
            print(f"\n  Upsampling by {factor} ...")
        fine_grid = g.upsample(factor)
        fine_rmin = self.rmin * factor if rescale_rmin else self.rmin

        fine_solver = type(self)(
            fine_grid,
            volfrac=self.volfrac, penal=self.penal, rmin=fine_rmin,
            move=self.move, eta=self.eta,
            nu=getattr(self, 'nu', 0.3),
            solver_kind=self.solver_kind,
        )

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  STAGE 2 : Fine "
                  f"({fine_grid.nx}×{fine_grid.ny}×{fine_grid.nz})")
            print(f"{'=' * 60}")

        fine_result, fine_hist = fine_solver.optimize(
            max_iter=fine_iter, verbose=verbose)

        # Update self to the fine state
        self.grid = fine_grid
        self.rmin = fine_rmin
        self._rebuild()
        # Merge histories
        merged = {}
        for key in set(coarse_hist.keys()) & set(fine_hist.keys()):
            merged[key] = coarse_hist[key] + fine_hist[key]
        self._history = merged if merged else {
            "compliance": (coarse_hist.get("compliance", [])
                           + fine_hist.get("compliance", [])),
            "volume": coarse_hist.get("volume", []) + fine_hist.get("volume", []),
            "change": coarse_hist.get("change", []) + fine_hist.get("change", []),
        }
        return fine_result, {"coarse": coarse_hist, "fine": fine_hist}

    # internal

    def _rebuild(self):
        """Recompute cached arrays after the grid changes size"""
        g = self.grid
        self._edof = self._build_edof()
        self._rhs = self._build_rhs()
        self._fixed = self._build_fixed()
        self._kernel = _cone_kernel(self.rmin)

    def visualize(self, threshold=0.5, cmap="plasma", mesh_overlay=True):
        """Open a static 3D viewer for the current density"""
        from visualizer import quick_view
        quick_view(self.grid, threshold=threshold, cmap=cmap,
                   mesh_overlay=mesh_overlay)
