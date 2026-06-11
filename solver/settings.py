"""
Solver factory that reads a JSON from settings/ and creates the
configured solver(s) for a given grid
"""

import json

from .structural import StructuralSolver
from .thermal import ThermalSolver
from .convection import ConvectionSolver
from .vibration import VibrationSolver
from .gravity import GravitySolver
from .stress import StressConstrainedSolver
from .maxlength import MaxLengthSolver
from .multi import MultiPhysicsSolver


def create_solver(grid, settings):
    """Create a configured solver from settings"""
    if isinstance(settings, dict):
        cfg = settings
    elif isinstance(settings, str):
        # Resolve short name -> full path for backward compat
        if not (settings.startswith("settings/") or "/" in settings or settings.endswith(".json")):
            path = f"settings/{settings}.json"
        else:
            path = settings
        with open(path) as f:
            cfg = json.load(f)
    else:
        raise TypeError(
            f"settings must be str or dict, got {type(settings).__name__}")

    solver_type = cfg["type"]

    if solver_type == "structural":
        return _make_structural(grid, cfg)

    elif solver_type == "thermal":
        return _make_thermal(grid, cfg)

    elif solver_type == "convection":
        return _make_convection(grid, cfg)

    elif solver_type == "vibration":
        return _make_vibration(grid, cfg)

    elif solver_type == "gravity":
        return _make_gravity(grid, cfg)

    elif solver_type == "stress":
        return _make_stress(grid, cfg)

    elif solver_type == "maxlength":
        return _make_maxlength(grid, cfg)

    elif solver_type == "multi":
        return _make_multi(grid, cfg)

    else:
        raise ValueError(
            f"Unknown solver type {solver_type!r} in {path}. "
            f"Expected 'structural', 'thermal', or 'multi'."
        )


# internal builders
def _material_kwargs(cfg):
    """Extract material parameters from a settings dict."""
    from ptypes import resolve_material
    mat = resolve_material(cfg.get("material", "aluminum"))
    return {"E": mat.E, "material_density": mat.density}


def _common_kwargs(cfg):
    """Extract parameters shared by all solvers."""
    kw = {k: cfg[k] for k in
          ("volfrac", "penal", "rmin", "move", "eta", "solver_kind")
          if k in cfg}
    kw.update(_material_kwargs(cfg))
    return kw


def _make_structural(grid, cfg):
    kwargs = _common_kwargs(cfg)
    kwargs["nu"] = cfg.get("nu", 0.3)
    return StructuralSolver(grid, **kwargs)


def _make_thermal(grid, cfg):
    kwargs = _common_kwargs(cfg)
    return ThermalSolver(grid, **kwargs)


def _make_convection(grid, cfg):
    kwargs = _common_kwargs(cfg)
    kwargs["h_conv"] = cfg.get("h_conv", 10.0)
    kwargs["penal_conv"] = cfg.get("penal_conv", 3.0)
    return ConvectionSolver(grid, **kwargs)


def _make_vibration(grid, cfg):
    kwargs = _common_kwargs(cfg)
    kwargs["nu"] = cfg.get("nu", 0.3)
    kwargs["min_frequency"] = cfg.get("min_frequency")
    kwargs["freq_penalty_weight"] = cfg.get("freq_penalty_weight", 1e-6)
    kwargs["activate_after_hz"] = cfg.get("activate_after_hz")
    return VibrationSolver(grid, **kwargs)


def _make_gravity(grid, cfg):
    kwargs = _common_kwargs(cfg)
    kwargs["nu"] = cfg.get("nu", 0.3)
    kwargs["gravity_direction"] = cfg.get("gravity_direction")
    kwargs["gravity_vector"] = cfg.get("gravity_vector")
    kwargs["gravity_penal"] = cfg.get("gravity_penal", 1.0)
    return GravitySolver(grid, **kwargs)


def _make_stress(grid, cfg):
    kwargs = _common_kwargs(cfg)
    kwargs["nu"] = cfg.get("nu", 0.3)
    kwargs["sigma_max"] = cfg.get("sigma_max")
    kwargs["p_stress"] = cfg.get("p_stress", 10.0)
    kwargs["stress_weight"] = cfg.get("stress_weight", 1.0)
    return StressConstrainedSolver(grid, **kwargs)


def _make_maxlength(grid, cfg):
    """Build a MaxLengthSolver wrapping a sub-solver"""
    sub_cfg = dict(cfg)                 # inherit volfrac, rmin, move, etc
    sub_cfg.update(cfg["solver"])       # inner block overrides
    sub_cfg.pop("solver", None)
    base = create_solver(grid, sub_cfg)
    return MaxLengthSolver(
        grid, base,
        r_max=cfg.get("r_max", 4.0),
        r_max_alpha=cfg.get("r_max_alpha", 1.5),
        r_max_weight=cfg.get("r_max_weight", 5.0),
    )


def _make_multi(grid, cfg):
    mode = cfg.get("mode", "simultaneous")
    solvers = []
    weights = []
    iterations = []

    for sc in cfg["solvers"]:
        sub_type = sc["type"]
        sub_cfg = dict(cfg)
        sub_cfg.update(sc)

        if sub_type == "structural":
            solvers.append(_make_structural(grid, sub_cfg))
        elif sub_type == "thermal":
            solvers.append(_make_thermal(grid, sub_cfg))
        elif sub_type == "convection":
            solvers.append(_make_convection(grid, sub_cfg))
        elif sub_type == "vibration":
            solvers.append(_make_vibration(grid, sub_cfg))
        elif sub_type == "gravity":
            solvers.append(_make_gravity(grid, sub_cfg))
        elif sub_type == "stress":
            solvers.append(_make_stress(grid, sub_cfg))
        elif sub_type == "maxlength":
            solvers.append(_make_maxlength(grid, sub_cfg))
        else:
            raise ValueError(f"Unknown sub-solver {sub_type!r}")

        if mode == "simultaneous":
            weights.append(sc.get("weight", 1.0))
        elif mode == "pipeline":
            iterations.append(sc.get("iterations", 40))

    if mode == "simultaneous":
        return MultiPhysicsSolver(grid, solvers, weights=weights,
                                  mode="simultaneous")
    elif mode == "pipeline":
        return MultiPhysicsSolver(grid, solvers,
                                  mode="pipeline",
                                  iterations=iterations)
    else:  # alternating
        return MultiPhysicsSolver(grid, solvers,
                                  mode="alternating")
