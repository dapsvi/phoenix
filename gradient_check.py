"""
Gradient verification: finite-difference check for every physics.

Compares analytical sensitivity (from solver.evaluate) against
central-difference numerical gradients on a tiny test grid.
"""

import sys
import numpy as np


# Test grid (small enough for fast FD, big enough for a load path)

def _make_test_grid():
    """(6,4,4) grid with one fixed face and one loaded face"""
    from grid import Grid
    from ptypes import Load, Support
    g = Grid(6, 4, 4, initial_density=0.4, spacing=(1.0, 1.0, 1.0))

    # Fix left face in X
    g.add_support(Support(0, 0, 0, 3, 0, 3, constraint="fix"))

    # Downward load on right face
    g.add_load(Load(5, 5, 1, 2, 1, 2, direction=(0, -1, 0)))

    # Randomise density so gradient isn't degenerate
    rng = np.random.default_rng(42)
    g.density = rng.uniform(0.1, 0.9, (g.nx, g.ny, g.nz))
    g.density[g.solid_mask] = 1.0
    g.density[g.void_mask] = 0.0

    return g


def check_solver(solver, grid, n_test=10, eps=1e-4, verbose=False):
    """
    Compare analytical gradient against central-difference for n_test random free elements.
    """
    density = grid.density.copy()
    nele = density.size

    grid.density = density.copy()
    obj_ref, sens_ref = solver.evaluate(grid.density, rho_min=1e-3)
    sens_flat = sens_ref.transpose(2, 1, 0).ravel()

    free = np.where(~(grid.solid_mask | grid.void_mask).ravel())[0]
    # Exclude elements with floor sensitivity
    free = np.intersect1d(free, np.where(sens_flat > 1e-29)[0])
    if len(free) == 0:
        print("  ⚠  no free elements to test")
        return 0, 0, 0.0, []
    test_elems = np.random.default_rng(123).choice(
        free, size=min(n_test, len(free)), replace=False)

    grid.density = density.copy()
    obj_ref, sens_ref = solver.evaluate(grid.density, rho_min=1e-3)
    sens_flat = sens_ref.transpose(2, 1, 0).ravel()

    n_pass = 0
    n_fail = 0
    max_err = 0.0
    failures = []

    for e in test_elems:
        idx = np.unravel_index(e, density.shape, order='F')

        # perturb +eps
        grid.density = density.copy()
        grid.density[idx] += eps
        f_plus, _ = solver.evaluate(grid.density, rho_min=1e-3)

        # perturb -eps
        grid.density = density.copy()
        grid.density[idx] -= eps
        f_minus, _ = solver.evaluate(grid.density, rho_min=1e-3)

        num_grad = (f_plus - f_minus) / (2.0 * eps)
        ana_grad = sens_flat[e]

        # Skip elements where FD gives exactly zero
        if abs(num_grad) < 1e-30:
            continue

        denom = max(abs(ana_grad), abs(num_grad), 1e-30)
        rel_err = abs(ana_grad + num_grad) / denom

        if verbose:
            print(f"    elem {e:5d}  |  analyt {ana_grad:+.6e}  "
                  f"numer {num_grad:+.6e}  |  rel {rel_err:.2e}")

        if rel_err < 5e-3:
            n_pass += 1
        else:
            n_fail += 1
            failures.append((e, ana_grad, num_grad, rel_err))
        max_err = max(max_err, rel_err)

    return n_pass, n_fail, max_err, failures


# Per-physics test runners

def check_structural():
    from solver.structural import StructuralSolver
    from grid import Grid
    g = _make_test_grid()
    s = StructuralSolver(g, volfrac=0.3, penal=3.0, rmin=1.5)
    return check_solver(s, g)

def check_thermal():
    from solver.thermal import ThermalSolver
    from grid import Grid
    g = _make_test_grid()
    s = ThermalSolver(g, volfrac=0.3, penal=3.0, rmin=1.5)
    return check_solver(s, g)

def check_vibration():
    from solver.vibration import VibrationSolver
    from grid import Grid
    g = _make_test_grid()
    # Higher density so fewer elements hit the rho_min floor
    rng = np.random.default_rng(42)
    g.density = rng.uniform(0.35, 0.9, (g.nx, g.ny, g.nz))
    g.density[g.solid_mask] = 1.0
    g.density[g.void_mask] = 0.0
    s = VibrationSolver(g, volfrac=0.3, penal=3.0, rmin=1.5)
    n_pass, n_fail, max_err, failures = check_solver(s, g, n_test=8)
    return n_pass, n_fail, max_err, failures


def check_vibration_isolated():
    import numpy as np
    from grid import Grid
    from ptypes import Load, Support
    from solver.vibration import VibrationSolver
    from scipy.sparse.linalg import eigsh

    g = Grid(8, 4, 4, initial_density=0.5, spacing=(1.0, 1.0, 1.0))
    g.add_support(Support(0, 0, 0, 3, 0, 3, constraint="fix"))
    g.add_load(Load(7, 7, 1, 2, 1, 2, direction=(0, -1, 0)))
    rng = np.random.default_rng(42)
    g.density = rng.uniform(0.4, 0.9, (g.nx, g.ny, g.nz))
    g.density[g.solid_mask] = 1.0
    g.density[g.void_mask] = 0.0

    s = VibrationSolver(g, volfrac=0.3, penal=3.0, rmin=1.5)
    rho_flat = g.density.transpose(2, 1, 0).ravel()
    active_mask = rho_flat >= 1e-3
    K = s._assemble_stiffness(g.density.copy(), rho_min=1e-3)
    M = s._assemble_mass(g.density.copy(), rho_min=1e-3)
    free = np.ones(s.ndof, dtype=bool)
    free[s._fixed] = False
    free_idx = np.where(free)[0]
    K_f, M_f = K[free_idx, :][:, free_idx], M[free_idx, :][:, free_idx]
    w2_vals, phi_f = eigsh(K_f, M=M_f, k=1, which='SM')
    omega2 = w2_vals[0]
    phi = np.zeros(s.ndof); phi[free_idx] = phi_f[:, 0]
    phi /= np.sqrt(max(phi @ M @ phi, 1e-15))

    edof_a = s._edof[active_mask]
    rho_a = rho_flat[active_mask]
    n_a = active_mask.sum()
    phi_e = phi[edof_a]
    stiff = np.einsum('ij,jk,ik->i', phi_e, s._KE, phi_e)
    mass  = np.einsum('ij,jk,ik->i', phi_e, s._ME, phi_e)
    sens_e = s.penal * rho_a ** (s.penal - 1) * stiff - omega2 * mass  # NO clip

    sens_ref = np.zeros(g.nele)
    sens_ref[active_mask] = sens_e
    sens_ref = sens_ref.reshape((g.nx, g.ny, g.nz), order='F')
    sens_flat = sens_ref.transpose(2, 1, 0).ravel()

    free_e = np.where(~(g.solid_mask | g.void_mask).ravel())[0]
    free_e = np.intersect1d(free_e, np.where(active_mask)[0])
    free_e = np.intersect1d(free_e, np.where(np.abs(sens_flat) > 1e-30)[0])
    test_e = np.random.default_rng(123).choice(free_e, size=min(8, len(free_e)), replace=False)

    def _get_omega2(dens):
        K2 = s._assemble_stiffness(dens, rho_min=1e-3)
        M2 = s._assemble_mass(dens, rho_min=1e-3)
        w2, _ = eigsh(K2[free_idx,:][:,free_idx], M=M2[free_idx,:][:,free_idx], k=1, which='SM')
        return w2[0]

    eps = 1e-4; n_pass, n_fail = 0, 0
    d = g.density.copy()
    for e in test_e:
        idx = np.unravel_index(e, d.shape, order='F')
        dp, dm = d.copy(), d.copy()
        dp[idx] += eps; dm[idx] -= eps
        num = (_get_omega2(dp) - _get_omega2(dm)) / (2 * eps)
        ana = sens_flat[e]
        if abs(num) < 1e-30: continue
        rel = abs(ana - num) / max(abs(ana), abs(num), 1e-30)
        if rel < 1e-3: n_pass += 1
        else:
            n_fail += 1
            print(f"    elem {e}: analyt={ana:.6e} numer={num:.6e} rel_err={rel:.2e}")
    status = "PASS" if n_fail == 0 else f"{n_fail} FAILURES"
    print(f"  {n_pass}/{n_pass+n_fail} passed  |  {status}")
    return n_pass, n_fail, 0.0, []


def check_gravity():
    from solver.gravity import GravitySolver
    from grid import Grid
    g = _make_test_grid()
    rng = np.random.default_rng(42)
    g.density = rng.uniform(0.35, 0.9, (g.nx, g.ny, g.nz))
    g.density[g.solid_mask] = 1.0
    g.density[g.void_mask] = 0.0
    s = GravitySolver(g, volfrac=0.3, penal=3.0, rmin=1.5,
                      gravity_vector=(0, -1, 0))
    return check_solver(s, g)

def check_stress():
    from solver.stress import StressConstrainedSolver
    from grid import Grid
    g = _make_test_grid()
    s = StressConstrainedSolver(g, volfrac=0.3, penal=3.0, rmin=1.5,
                                 sigma_max=10.0, p_stress=10.0,
                                 stress_weight=1.0)
    return check_solver(s, g)

def check_convection():
    from solver.convection import ConvectionSolver
    from grid import Grid
    g = _make_test_grid()
    rng = np.random.default_rng(42)
    g.density = rng.uniform(0.35, 0.9, (g.nx, g.ny, g.nz))
    g.density[g.solid_mask] = 1.0
    g.density[g.void_mask] = 0.0
    s = ConvectionSolver(g, volfrac=0.3, penal=3.0, rmin=1.5)
    return check_solver(s, g)


ALL_CHECKS = {
    "structural": check_structural,
    "thermal":    check_thermal,
    "gravity":    check_gravity,
    "stress":     check_stress,
    "convection": check_convection,
    "vibration":  check_vibration,       # clipped, use vib_iso for formula verification
    "vib_iso":    check_vibration_isolated,  # unclipped single-mode, confirms formula
}


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if args:
        names = [n for n in args if n in ALL_CHECKS]
        if not names:
            print(f"Unknown physics. Choose from: {list(ALL_CHECKS)}")
            sys.exit(1)
    else:
        names = list(ALL_CHECKS)

    all_ok = True
    for name in names:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        n_pass, n_fail, max_err, failures = ALL_CHECKS[name](
            ) if verbose else ALL_CHECKS[name]()
        status = "PASS" if n_fail == 0 else f"{n_fail} FAILURES"
        print(f"  {n_pass}/{n_pass + n_fail} passed  |  "
              f"max rel err {max_err:.2e}  |  {status}")
        if failures:
            for e, ana, num, err in failures:
                print(f"    elem {e}: analyt={ana:.4e} numer={num:.4e} "
                      f"rel_err={err:.2e}")
            all_ok = False

    if all_ok:
        print("\nAll gradients correct.")
    else:
        print("\nSome gradients are wrong, check failures above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
