"""
Live 3-D viewer orchestration for any BaseSolver subclass
"""

import threading
from visualizer import Viewer, Scene


def live_optimize(solver, max_iter=60, tol=0.01, verbose=True,
                  threshold=0.5, cmap="viridis",
                  mesh_overlay=True, mesh_opacity=0.3,
                  poll_interval_ms=100,
                  final="best",
                  snapshot_interval=0):
    """Run a solver with a live 3-D viewer"""
    grid = solver.grid

    # shared state
    state = {
        "density": grid.density.copy(),
        "iteration": 0,
        "compliance": 0.0,
        "running": True,
    }

    def _cb(density, iteration, compliance):
        state["density"] = density.copy()
        state["iteration"] = iteration
        state["compliance"] = float(compliance)

    def _solve_thread():
        try:
            solver.optimize(max_iter=max_iter, tol=tol,
                            verbose=verbose, callback=_cb,
                            final=final,
                            snapshot_interval=snapshot_interval)
        finally:
            # Push final density as the last rendered frame
            _cb(solver.grid.density, state["iteration"] + 1, state["compliance"])
            state["running"] = False

    thread = threading.Thread(target=_solve_thread, daemon=True)
    thread.start()

    # viewer on main thread
    scene = Scene(
        grid.density,
        load_mask=grid.load_mask,
        load_direction=grid.load_direction,
        load_cases=grid.load_cases,
        support_mask=grid.support_mask,
        spacing=grid.spacing, origin=grid.origin,
        threshold=threshold, cmap=cmap,
        mesh_overlay=mesh_overlay, mesh_opacity=mesh_opacity,
    )
    viewer = Viewer(scene, poll_interval_ms=poll_interval_ms)
    viewer.start_live(state)
    thread.join()

    return solver.grid.density, solver.history, getattr(solver, '_snapshots', [])


def live_continuation(solver, factor=2, coarse_iter=40, fine_iter=30,
                      rescale_rmin=True, verbose=True,
                      threshold=0.5, cmap="viridis",
                      mesh_overlay=True, mesh_opacity=0.3,
                      poll_interval_ms=100):
    """Live two-stage refinement"""
    grid = solver.grid

    # stage 1: coarse optimization
    density, coarse_hist, _ = live_optimize(
        solver, max_iter=coarse_iter, tol=0.01, verbose=verbose,
        threshold=threshold, cmap=cmap,
        mesh_overlay=mesh_overlay, mesh_opacity=mesh_opacity,
        poll_interval_ms=poll_interval_ms,
    )

    # upsample grid and solver for stage 2
    fine_grid = grid.upsample(factor)
    fine_rmin = solver.rmin * factor if rescale_rmin else solver.rmin

    # MultiPhysicsSolver has a different constructor signature
    from solver.multi import MultiPhysicsSolver
    if isinstance(solver, MultiPhysicsSolver):
        # Re-create sub-solvers on the fine grid
        fine_solvers = []
        for s in solver.solvers:
            fs = type(s)(
                fine_grid,
                volfrac=s.volfrac, penal=s.penal, rmin=fine_rmin,
                move=s.move, eta=s.eta,
                nu=getattr(s, 'nu', 0.3),
                solver_kind=getattr(s, 'solver_kind', 'direct'),
            )
            fine_solvers.append(fs)
        fine_solver = MultiPhysicsSolver(
            fine_grid, fine_solvers,
            weights=solver.weights,
            mode=solver.mode,
            iterations=solver.iterations,
        )
    else:
        fine_solver = type(solver)(
            fine_grid,
            volfrac=solver.volfrac, penal=solver.penal, rmin=fine_rmin,
            move=solver.move, eta=solver.eta,
            nu=getattr(solver, 'nu', 0.3),
            solver_kind=getattr(solver, 'solver_kind', 'direct'),
        )

    # stage 2: fine optimization
    fine_density, fine_hist, _ = live_optimize(
        fine_solver, max_iter=fine_iter, tol=0.01, verbose=verbose,
        threshold=threshold, cmap=cmap,
        mesh_overlay=mesh_overlay, mesh_opacity=mesh_opacity,
        poll_interval_ms=poll_interval_ms,
    )

    return fine_density, {"coarse": coarse_hist, "fine": fine_hist}
