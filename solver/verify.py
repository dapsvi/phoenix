"""
Post-optimization verification
"""

import numpy as np
from dataclasses import dataclass
from ._stress_utils import build_B_centroid, build_D, von_mises



@dataclass
class StressReport:
    """Results of a multi-case structural verification"""
    mass_g: float
    n_solid: int
    n_total: int
    compliance: float
    max_displacement: float
    max_vm_stress: float
    safety_factor: float | None
    max_stress_location: tuple
    per_element_vm: np.ndarray
    per_element_disp: np.ndarray
    displacement: np.ndarray | None = None
    governing_case: int = 0

    def summary(self, mat=None):
        lines = [
            f"  Mass:           {self.mass_g:.1f} g",
            f"  Solid elements: {self.n_solid} / {self.n_total} "
            f"({100*self.n_solid/self.n_total:.1f}%)",
            f"  Compliance:     {self.compliance:.4e} N·mm",
            f"  Max displacement: {self.max_displacement:.4f} mm",
        ]
        loc = self.max_stress_location
        lines.append(
            f"  Max von Mises:  {self.max_vm_stress:.1f} MPa  "
            f"(element {loc[0]},{loc[1]},{loc[2]})  "
            f"[case {self.governing_case}]")
        if self.safety_factor is not None:
            sf = self.safety_factor
            status = ("[OK]" if sf >= 2.0 else
                      "[LOW]" if sf >= 1.2 else "[FAIL]")
            lines.append(f"  Safety factor:  {sf:.2f}  [{status}]")
        return "\n".join(lines)


@dataclass
class FrequencyReport:
    """Results of a vibration verification"""
    f1_hz: float
    f2_hz: float = 0.0
    f3_hz: float = 0.0
    n_modes: int = 1

    def summary(self, mat=None):
        lines = [f"  f₁ = {self.f1_hz:.1f} Hz"]
        if self.n_modes >= 2:
            lines.append(f"  f₂ = {self.f2_hz:.1f} Hz")
        if self.n_modes >= 3:
            lines.append(f"  f₃ = {self.f3_hz:.1f} Hz")
        return "\n".join(lines)



def verify_stress(solver, density, threshold=0.5, mat=None):
    """Multi-case structural verification: stress + displacement"""
    g = solver.grid
    h = float(g.spacing[0])
    E_val = solver.E
    nu_val = getattr(solver, 'nu', 0.3)
    yield_stress = getattr(mat, 'yield_stress', None) if mat else None

    binary = (density >= threshold).astype(np.float64)
    n_solid = int(binary.sum())
    n_total = int(binary.size)

    mass_g = 0.0
    if mat is not None:
        mass_g = n_solid * h**3 * mat.density * 1e6

    save_penal = solver.penal
    solver.penal = 1.0
    K = solver._assemble_stiffness(binary, rho_min=0.5)
    solver.penal = save_penal

    B = build_B_centroid(h)
    D_mat = build_D(E_val, nu_val)
    DB = D_mat @ B

    rho_flat = binary.transpose(2, 1, 0).ravel()
    active = rho_flat >= 0.5
    edof_a = solver._edof[active]
    n_a = active.sum()

    rhs_list = solver._rhs if isinstance(solver._rhs, list) else [solver._rhs]
    vm_field_worst = np.zeros(g.nele, dtype=np.float64)
    disp_field_worst = np.zeros(g.nele, dtype=np.float64)
    max_vm, max_disp, governing_vm = 0.0, 0.0, 0
    compliance = 0.0
    active_idx = np.where(active)[0]

    for ci, f in enumerate(rhs_list):
        u = solver._solve(K, f, None)
        if ci == 0:
            compliance = float(f @ u)
        disp = float(np.abs(u).max())
        if disp > max_disp:
            max_disp = disp
        u_xyz = u.reshape(-1, 3)
        for i, e_flat in enumerate(active_idx):
            node_indices = solver._edof[active][i, ::3] // 3
            ue_mag = np.linalg.norm(u_xyz[node_indices], axis=1).max()
            if ue_mag > disp_field_worst[e_flat]:
                disp_field_worst[e_flat] = ue_mag
        stress_e = np.empty((n_a, 6), dtype=np.float64)
        for e in range(n_a):
            stress_e[e] = DB @ u[edof_a[e]]
        vm = von_mises(stress_e)
        vm_flat = np.zeros(g.nele, dtype=np.float64)
        vm_flat[active] = vm
        vm_field_worst = np.maximum(vm_field_worst, vm_flat)
        if vm.max() > max_vm:
            max_vm = float(vm.max())
            governing_vm = ci

    vm_field = vm_field_worst.reshape((g.nx, g.ny, g.nz), order='F')
    disp_field = disp_field_worst.reshape((g.nx, g.ny, g.nz), order='F')
    peak_flat = np.argmax(vm_field.ravel(order='F'))
    safety = yield_stress / max_vm if yield_stress else None

    return StressReport(
        mass_g=mass_g, n_solid=n_solid, n_total=n_total,
        compliance=compliance, max_displacement=max_disp,
        max_vm_stress=max_vm, safety_factor=safety,
        max_stress_location=(peak_flat % g.nx, (peak_flat // g.nx) % g.ny,
                             peak_flat // (g.nx * g.ny)),
        per_element_vm=vm_field, per_element_disp=disp_field,
        governing_case=governing_vm,
    )


def verify_vibration(solver, density, threshold=0.5, mat=None):
    """Vibration verification: natural frequencies on binary density"""
    from .vibration import VibrationSolver
    g = solver.grid

    if hasattr(solver, '_ME'):
        vs = solver
    else:
        rho = getattr(mat, 'density', 1.0) if mat else 1.0
        vs = VibrationSolver(g, volfrac=solver.volfrac, penal=1.0,
                              rmin=solver.rmin, nu=getattr(solver, 'nu', 0.3),
                              E=solver.E, material_density=rho)

    binary = (density >= threshold).astype(np.float64)
    save_penal = vs.penal
    vs.penal = 1.0
    try:
        K = vs._assemble_stiffness(binary, rho_min=0.5)
        M = vs._assemble_mass(binary, rho_min=0.5)
        from scipy.sparse.linalg import eigsh
        free = np.ones(vs.ndof, dtype=bool)
        free[vs._fixed] = False
        free_idx = np.where(free)[0]
        n_free = len(free_idx)
        k = min(3, max(1, n_free - 1))
        w2, _ = eigsh(K[free_idx, :][:, free_idx],
                       M=M[free_idx, :][:, free_idx], k=k, which='SM')
        f_hz = np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)
        return FrequencyReport(
            f1_hz=float(f_hz[0]),
            f2_hz=float(f_hz[1]) if len(f_hz) > 1 else 0.0,
            f3_hz=float(f_hz[2]) if len(f_hz) > 2 else 0.0,
            n_modes=len(f_hz))
    finally:
        vs.penal = save_penal



VERIFIERS = [
    ("structural", verify_stress),
    ("vibration",  verify_vibration),
]


def get_verifier(name):
    """Look up a verifier by name"""
    for n, fn in VERIFIERS:
        if n == name:
            return fn
    raise ValueError(f"Unknown verifier {name!r}. Available: "
                     f"{[n for n, _ in VERIFIERS]}")


def run_all(solver, density, threshold=0.5, mat=None, names=None):
    """Run all registered verifiers and return [(name, report), ...]"""
    if names is None:
        names = [n for n, _ in VERIFIERS]
    results = []
    for name in names:
        try:
            fn = get_verifier(name)
            results.append((name, fn(solver, density, threshold, mat)))
        except Exception as e:
            results.append((name, f"FAILED: {e}"))
    return results