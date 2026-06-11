"""
Multi-physics topology optimisation with three modes of coupling
"""

import numpy as np
from .base import _apply_fixed_regions, _apply_symmetry


def _normalize(sens):
    m = np.abs(sens).max()
    return sens / max(m, 1e-30)


class MultiPhysicsSolver:
    """
    Parameters
    ----------
    grid : Grid
    solvers : list of BaseSolver
    weights : list of float   (simultaneous mode only)
    mode : "simultaneous" | "alternating" | "pipeline"
    iterations : list of int  (pipeline mode only)
    normalize : bool
        Normalise each sensitivity field before combining (default True).
    """

    def __init__(self, grid, solvers, weights=None,
                 mode="simultaneous", iterations=None,
                 normalize=True):
        if not solvers:
            raise ValueError("need at least one solver")
        self.grid = grid
        self.solvers = list(solvers)
        self.mode = mode
        self._normalize = normalize and mode == "simultaneous"

        if mode == "simultaneous":
            if weights is None or len(weights) != len(solvers):
                raise ValueError("simultaneous needs weights")
            self.weights = list(weights)
        else:
            self.weights = None

        self.iterations = (list(iterations) if iterations
                           else [40] * len(solvers))

        primary = solvers[0]
        self.volfrac = primary.volfrac
        self.penal = primary.penal
        self.move = primary.move
        self.eta = primary.eta
        self.rmin = primary.rmin
        self._kernel = primary._kernel
        self._history = {}

        # apply fixed regions + symmetry to initial density
        _apply_fixed_regions(grid.density, grid)
        _apply_symmetry(grid.density, grid)

    @property
    def history(self):
        return self._history

    def report_metric(self):
        """Return the blended objective on the current design."""
        try:
            total = 0.0
            for s, w in zip(self.solvers, self.weights):
                obj, _ = s.evaluate(self.grid.density)
                total += w * obj
            return ("objective", float(total), "")
        except Exception:
            return ("objective", 0.0, "")

    # simultaneous (weighted sum with normalisation)
    def _optimize_simultaneous(self, max_iter=60, tol=0.01,
                                verbose=True, callback=None,
                                rho_min=1e-3, final="best",
                                snapshot_interval=0):
        from .base import run_simp_loop

        # per-solver scale state (captured by closure, set on iter 0)
        scale_state = {"obj_scale": None, "sens_scale": None,
                       "p_finals": [s.penal for s in self.solvers]}

        # Separate maxlength solvers
        reg_solvers = []
        reg_weights = []
        maxlen_solvers = []
        for i, (s, w) in enumerate(zip(self.solvers, self.weights)):
            is_maxlen = hasattr(s, 'r_max') and s.r_max is not None
            if is_maxlen:
                maxlen_solvers.append((i, s, w))
            else:
                reg_solvers.append((i, s, w))
                reg_weights.append(w)

        def _evaluate(density, iteration):
            combined_sens = np.zeros_like(density)
            total_obj = 0.0

            # regular solvers: weighted sum
            for i, solver, w in reg_solvers:
                pf = scale_state["p_finals"][i]
                p_cur = min(pf, 1.0 + iteration * (pf - 1.0) / 10.0)
                solver.penal = p_cur

                obj, sens = solver.evaluate(density, rho_min=rho_min)

                if iteration == 0:
                    if scale_state["obj_scale"] is None:
                        scale_state["obj_scale"] = []
                        scale_state["sens_scale"] = []
                    scale_state["obj_scale"].append(max(abs(obj), 1e-30))
                    scale_state["sens_scale"].append(max(np.abs(sens).max(), 1e-30))

                if self._normalize and scale_state["sens_scale"] is not None:
                    sens = sens / scale_state["sens_scale"][i]

                total_obj += w * obj / scale_state["obj_scale"][i]
                combined_sens += w * sens

            # maxlength solvers: multiplicative post-blend
            for i, solver, w in maxlen_solvers:
                pf = scale_state["p_finals"][i]
                solver.penal = min(pf, 1.0 + iteration * (pf - 1.0) / 10.0)
                factor = solver.compute_suppression(density)
                if factor is not None:
                    combined_sens *= factor ** w

            return total_obj, combined_sens

        result = run_simp_loop(
            self, max_iter=max_iter, tol=tol, verbose=verbose,
            callback=callback, rho_min=rho_min, final=final,
            snapshot_interval=snapshot_interval,
            evaluate_fn=_evaluate, history_key="objective",
            lower_is_better=True,   # blended objective is lower/better
        )
        # restore sub-solver penal
        for solver, pf in zip(self.solvers, scale_state["p_finals"]):
            solver.penal = pf
        return result

    # alternating (cycle: A, B, A, B, etc)
    def _optimize_alternating(self, max_iter=60, tol=0.01,
                               verbose=True, callback=None,
                               rho_min=1e-3, final="best",
                               snapshot_interval=0):
        from .base import run_simp_loop

        n = len(self.solvers)
        p_finals = [s.penal for s in self.solvers]

        def _evaluate(density, iteration):
            solver = self.solvers[iteration % n]
            pf = p_finals[iteration % n]
            p_cur = min(pf, 1.0 + iteration * (pf - 1.0) / 10.0)
            solver.penal = p_cur
            return solver.evaluate(density, rho_min=rho_min)

        result = run_simp_loop(
            self, max_iter=max_iter, tol=tol, verbose=verbose,
            callback=callback, rho_min=rho_min, final=final,
            snapshot_interval=snapshot_interval,
            evaluate_fn=_evaluate, history_key="objective",
            lower_is_better=True,
        )
        for solver, pf in zip(self.solvers, p_finals):
            solver.penal = pf
        return result

    # pipeline (A, B, C, ... fully)
    def _optimize_pipeline(self, tol=0.01, verbose=True,
                            callback=None, rho_min=1e-3,
                            final="best"):
        g = self.grid
        all_histories = {}
        n_stages = len(self.solvers)
        for i, (solver, n_iter) in enumerate(
                zip(self.solvers, self.iterations)):
            label = f"stage_{i}_{type(solver).__name__}"
            if verbose:
                print(f"\n--- {label} ({n_iter} iters) ---")
            # Pass final only to the last stage
            stage_final = final if i == n_stages - 1 else "best"
            _, h = solver.optimize(max_iter=n_iter, tol=tol,
                                   verbose=verbose,
                                   callback=callback,
                                   rho_min=rho_min,
                                   final=stage_final)
            all_histories[label] = h
        self._history = all_histories
        return g.density, all_histories

    # dispatch
    def optimize(self, max_iter=60, tol=0.01, verbose=True,
                 callback=None, rho_min=1e-3, final="best",
                 snapshot_interval=0):
        if self.mode == "alternating":
            return self._optimize_alternating(
                max_iter=max_iter, tol=tol, verbose=verbose,
                callback=callback, rho_min=rho_min, final=final,
                snapshot_interval=snapshot_interval)
        elif self.mode == "pipeline":
            return self._optimize_pipeline(
                tol=tol, verbose=verbose,
                callback=callback, rho_min=rho_min, final=final)
        else:
            return self._optimize_simultaneous(
                max_iter=max_iter, tol=tol, verbose=verbose,
                callback=callback, rho_min=rho_min, final=final,
                snapshot_interval=snapshot_interval)
