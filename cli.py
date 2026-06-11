#!/usr/bin/env python3
"""Phoenix topology optimisation CLI."""

import argparse
import sys
from pathlib import Path


# helpers

def _load_input(path):
    """Load a scene/preset/result, return (grid, preset, suggested_settings)."""
    from ptypes import Preset, Result

    if path.endswith(".npz"):
        result = Result.load(path)
        preset = result.preset
        grid = preset.to_grid()
        grid.density = result.density
        return grid, preset, preset.settings_name

    if path.endswith(".py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_preset_mod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "scene"):
            raise SystemExit(f"Error: {path} must export a `scene` object.")
        from scene import Scene
        scene = mod.scene
        grid = scene.to_grid()
        preset = scene.to_preset()
        return grid, preset, scene.suggested_settings

    import json
    with open(path) as f:
        raw = json.load(f)

    if "objects" in raw:
        from scene import Scene
        scene = Scene.from_dict(raw)
        grid = scene.to_grid()
        preset = scene.to_preset()
        suggested = scene.suggested_settings
    else:
        preset = Preset.from_dict(raw)
        grid = preset.to_grid()
        suggested = preset.settings_name
    return grid, preset, suggested


def _resolve_settings_obj(cli_flag, suggested):
    """Resolve settings with CLI-flag priority."""
    if cli_flag is not None:
        return cli_flag
    if suggested is not None:
        return suggested
    print("Error: no settings provided.  Use -s/--settings or add "
          "'suggested_settings' to the scene JSON.")
    raise SystemExit(1)


def _load_settings_dict(settings):
    """Normalise settings to a dict, loading from file if needed."""
    if isinstance(settings, dict):
        return settings
    import json
    if settings.startswith("settings/") or "/" in settings or settings.endswith(".json"):
        path = settings
    else:
        path = f"settings/{settings}.json"
    with open(path) as f:
        return json.load(f)


# optimize

def cmd_optimize(args):
    """Run a topology optimisation with live viewer."""
    from ptypes import Preset, Result
    from solver.settings import create_solver
    from live import live_optimize

    grid, preset, suggested = _load_input(args.preset)
    print(f"Loaded: {preset}")

    settings = _resolve_settings_obj(args.settings, suggested)
    cfg = _load_settings_dict(settings)
    if isinstance(settings, str):
        print(f"Settings: {settings}")
    else:
        print(f"Settings: inline ({cfg.get('type', '?')})")

    volfrac = cfg.get("volfrac", 0.1)
    initial_density = cfg.get("initial_density", volfrac)
    if grid.density.sum() == 0:
        grid.density[:] = initial_density

    sp = float(args.spacing or cfg.get("spacing", 1.0))
    grid.spacing = (sp, sp, sp)
    print(grid, "\n")

    solver = create_solver(grid, cfg)
    snap_interval = cfg.get("snapshot_interval", 0)
    density, history, snapshots = live_optimize(
        solver,
        max_iter=cfg.get("max_iter", 40),
        tol=cfg.get("tol", 0.01),
        threshold=args.threshold,
        cmap=args.cmap,
        final=args.final,
        snapshot_interval=snap_interval,
    )

    if hasattr(solver, 'solvers'):
        for s in solver.solvers:
            if hasattr(s, 'report_metric'):
                label, value, unit = s.report_metric()
                print(f"final {label}: {value:.4e} {unit}")
    elif hasattr(solver, 'report_metric'):
        label, value, unit = solver.report_metric()
        print(f"final {label}: {value:.4e} {unit}")

    from ptypes import resolve_material
    mat = resolve_material(cfg.get("material", "aluminum"))
    sp = float(grid.spacing[0])
    vol_mm3 = density.sum() * sp**3
    mass_g = vol_mm3 * mat.density * 1e6
    size_mm = (grid.nx * sp, grid.ny * sp, grid.nz * sp)
    print(f"Material: {mat.name}  |  {size_mm[0]:.0f}x{size_mm[1]:.0f}x{size_mm[2]:.0f} mm  "
          f"|  mass {mass_g:.1f} g  |  vol frac {density.sum()/density.size:.3f}")
    grid.spacing = (sp, sp, sp)

    preset_stem = Path(args.preset).stem
    result = Result(density, preset, history, density_snapshots=snapshots)
    out_npz = args.output or f"results/{preset_stem}_latest.npz"
    result.save(out_npz)
    print(result)

    if not args.no_stl:
        from export import to_stl
        stl_path = f"exports/{preset_stem}.stl"
        to_stl(density, stl_path,
               threshold=args.threshold,
               method=args.method,
               smooth_type=args.smooth,
               smooth_iterations=10,
               spacing=grid.spacing, origin=grid.origin)

    if not args.no_view:
        from visualizer import quick_view
        quick_view(grid, threshold=args.threshold, cmap=args.cmap,
                   mesh_overlay=not args.no_mesh)

    if not args.no_verify:
        _run_verification(solver, density, cfg, preset_stem)


def _run_verification(solver, density, cfg, name):
    """Run all registered verifiers on the final design."""
    from solver.verify import run_all
    from ptypes import resolve_material

    mat = resolve_material(cfg.get("material", "aluminum"))
    print(f"\n{'=' * 50}")
    print(f"  Verification for {name}")
    print(f"{'=' * 50}")
    print(f"  Material: {mat.name} (E={mat.E:.0f} MPa, "
          f"yield={mat.yield_stress or 'N/A'} MPa)")
    sp = float(solver.grid.spacing[0])
    print(f"  Grid:    {solver.grid.nx}x{solver.grid.ny}x{solver.grid.nz} "
          f"voxels @ {sp} mm")

    results = run_all(solver, density, threshold=0.5, mat=mat)
    for vname, report in results:
        if isinstance(report, str):
            print(f"\n  [{vname}] {report}")
        else:
            print(f"\n  [{vname}]")
            print(report.summary(mat))
    print(f"{'=' * 50}\n")


# export

def cmd_export(args):
    """Convert a result .npz file to STL."""
    from ptypes import Result
    from export import to_stl

    result = Result.load(args.result)
    print(f"Loaded: {result}")

    out = args.output or Path(args.result).with_suffix(".stl").name
    sp = (args.spacing,) * 3 if args.spacing else getattr(
        getattr(result, 'preset', None), 'spacing', (1, 1, 1))
    to_stl(result.density, out,
           threshold=args.threshold,
           method=args.method,
           smooth_type=args.smooth,
           smooth_iterations=args.smooth_iters,
           spacing=sp,
           origin=getattr(getattr(result, 'preset', None), 'origin', (0, 0, 0)),
           decimate=args.decimate)


# plot

def cmd_plot(args):
    """Plot history curves from a result file."""
    import matplotlib.pyplot as plt
    from ptypes import Result

    result = Result.load(args.result)
    hist = result.history

    if not hist:
        print("No history data in result.")
        return

    keys = [k for k in hist if k not in ("density_snapshots",)
            and isinstance(hist.get(k), list) and len(hist[k]) > 0]

    if not keys:
        print("No plottable history keys found.")
        return

    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        ax.plot(hist[key], marker=".", markersize=2, linewidth=0.8)
        ax.set_ylabel(key.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Iteration")
    fig.suptitle(f"Optimisation History : {Path(args.result).stem}",
                 fontsize=11)
    fig.tight_layout()

    out = args.output or f"plots/{Path(args.result).stem}_history.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")

    if not args.no_show:
        plt.show()


# view

def cmd_view(args):
    """Open the 3D viewer on the final density of a result."""
    from ptypes import Result
    from visualizer import Scene, Viewer

    result = Result.load(args.result)
    print(f"Loaded: {result}")

    d = result.density
    visible = (d > args.threshold).sum()
    if visible == 0:
        print(f"Warning: no voxels above threshold {args.threshold}. "
              f"density in [{d.min():.3f}, {d.max():.3f}]. "
              f"Try --threshold {max(0.1, d.max() * 0.7):.2f}")
        return

    grid = result.to_grid()
    scene = Scene(
        grid.density,
        load_mask=grid.load_mask,
        load_direction=grid.load_direction,
        load_cases=grid.load_cases,
        support_mask=grid.support_mask,
        spacing=grid.spacing, origin=grid.origin,
        threshold=args.threshold,
        cmap=args.cmap,
        mesh_overlay=not args.no_mesh,
        mesh_opacity=args.mesh_opacity,
    )
    Viewer(scene).show()


# animate

def cmd_animate(args):
    """Interactive viewer that cycles through snapshots."""
    from ptypes import Result
    from visualizer import Scene, Viewer

    result = Result.load(args.result)
    print(f"Loaded: {result}")

    snapshots = result.density_snapshots
    if not snapshots:
        print("No snapshots in this result.  Run an optimization with "
              "snapshot_interval > 0 in the settings JSON.")
        return

    print(f"  {len(snapshots)} snapshots: cycling every {args.duration}s")

    grid = result.to_grid()
    scene = Scene(
        grid.density,
        load_mask=grid.load_mask,
        load_direction=grid.load_direction,
        load_cases=grid.load_cases,
        support_mask=grid.support_mask,
        spacing=grid.spacing, origin=grid.origin,
        threshold=args.threshold,
        cmap=args.cmap,
        mesh_overlay=not args.no_mesh,
        mesh_opacity=args.mesh_opacity,
    )

    viewer = Viewer(scene)
    viewer.show_with_snapshots(snapshots, play_interval=args.duration)


# verify

def cmd_verify(args):
    """Run verification on a result file."""
    from ptypes import Result, resolve_material
    from solver.settings import create_solver
    from solver.verify import run_all

    result = Result.load(args.result)
    print(f"Loaded: {result}")

    preset = result.preset
    grid = preset.to_grid()
    grid.spacing = (1.0, 1.0, 1.0)

    cfg = {}
    if args.settings:
        import json
        with open(args.settings) as f:
            cfg = json.load(f)
    elif preset.settings_name:
        try:
            cfg = preset.solver_settings or {}
        except Exception:
            pass
    mat = resolve_material(cfg.get("material", args.material or "aluminum"))
    sp = float(args.spacing or cfg.get("spacing", 1.0))
    grid.spacing = (sp, sp, sp)

    solver = create_solver(grid, cfg if cfg else {
        "type": "structural", "material": mat.name, "spacing": sp})
    if hasattr(solver, 'solvers'):
        structural = next(
            (s for s in solver.solvers if hasattr(s, 'nu')),
            solver.solvers[0])
    else:
        structural = solver

    reports = run_all(structural, result.density,
                      threshold=args.threshold, mat=mat)
    report = None
    for vname, r in reports:
        if isinstance(r, str):
            print(f"[{vname}] {r}")
        else:
            print(f"\n-- {vname} --")
            print(r.summary(mat))
            if vname == "structural":
                report = r

    if not args.no_view and report is not None:
        _show_verify_viewer(result, report, grid, mat, args)


def _show_verify_viewer(result, report, grid, mat, args):
    """3D viewer with stress / displacement / safety factor heatmaps."""
    import numpy as np
    from visualizer import Scene, grid_to_heatmap_unstructured
    import pyvista as pv

    binary_solid = (result.density >= args.threshold).astype(np.float64)

    vm = report.per_element_vm.copy()
    disp = report.per_element_disp.copy()
    vm_max = max(vm.max(), 1e-10)
    disp_max = max(disp.max(), 1e-10)
    vm_norm = binary_solid * np.clip(vm / vm_max, 0, 1)
    disp_norm = binary_solid * np.clip(disp / disp_max, 0, 1)

    sf_field = np.zeros_like(vm_norm)
    if report.safety_factor is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            sf_raw = binary_solid * mat.yield_stress / np.maximum(
                report.per_element_vm, 1e-10)
            sf_field = np.clip(sf_raw / 5.0, 0, 1)

    fields = {"stress": vm_norm, "displacement": disp_norm, "safety": sf_field}
    labels = {
        "stress": f"von Mises (max {report.max_vm_stress:.0f} MPa)",
        "displacement": f"Displacement (max {report.max_displacement:.3f} mm)",
        "safety": "Safety factor (red < 1, yellow ~2, green >= 5)",
    }
    cmaps = {"stress": "hot", "displacement": "hot", "safety": "RdYlGn"}
    current = ["stress"]

    plotter = pv.Plotter(window_size=(1024, 768), lighting="three lights")
    plotter.set_background("white")

    s = Scene(grid.density, load_mask=None, load_direction=None,
              support_mask=None, spacing=grid.spacing, origin=grid.origin,
              threshold=0.0, cmap="hot", mesh_overlay=False)
    nx, ny, nz = s.density.shape
    sx, sy, sz = s.spacing
    ox, oy, oz = s.origin
    plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z",
                     line_width=2, labels_off=False)
    box = pv.Box(bounds=[
        ox, ox + nx * sx, oy, oy + ny * sy, oz, oz + nz * sz,
    ]).extract_all_edges()
    plotter.add_mesh(box, color="black", line_width=2, name="bbox")

    heatmap_actor = None

    def _show_field(name):
        nonlocal heatmap_actor
        field = fields[name]
        visible_mask = binary_solid > 0.0
        n_visible = int(visible_mask.sum())
        if args.debug:
            print(f"[verify] switching to {name}: {n_visible} / {field.size} "
                  f"solid voxels (field min={field.min():.4f}, "
                  f"max={field.max():.4f})")
        if heatmap_actor:
            plotter.remove_actor(heatmap_actor, render=False)
            heatmap_actor = None
        plotter.render()
        render_field = np.where(visible_mask, field, -1.0)
        ug = grid_to_heatmap_unstructured(
            render_field, threshold=-0.5,
            cmap=cmaps.get(name, "hot"),
            spacing=s.spacing, origin=s.origin)
        if args.debug:
            print(f"[verify]   heatmap cells: {ug.n_cells}")
        if ug.n_cells > 0:
            heatmap_actor = plotter.add_mesh(
                ug, scalars="color", rgb=True,
                opacity=1.0, show_edges=False, name="heatmap")
        sf_str = f"SF {report.safety_factor:.1f}  |  " if report.safety_factor else ""
        hint = hints.get(name, "")
        plotter.add_title(f"{labels[name]}  |  {sf_str}{hint}", font_size=11)
        plotter.render()

    def _on_key(obj, event):
        key = plotter.iren.interactor.GetKeySym().lower()
        if key == "s":
            current[0] = "stress"; _show_field("stress")
        elif key == "d":
            current[0] = "displacement"; _show_field("displacement")
        elif key == "f":
            current[0] = "safety"; _show_field("safety")

    def _block_vtk_chars(obj, event):
        if obj.GetKeySym().lower() in ("s", "w", "d", "f"):
            obj.SetKeySym("")

    hints = {
        "stress": "D=displacement  F=safety",
        "displacement": "S=stress  F=safety",
        "safety": "S=stress  D=displacement",
    }

    plotter.iren.add_observer("KeyPressEvent", _on_key)
    import warnings
    from pyvista import PyVistaDeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyVistaDeprecationWarning)
        plotter.iren.add_observer("CharEvent", _block_vtk_chars, 1.0)
    _show_field("stress")
    plotter.show()


# main

def main():
    parser = argparse.ArgumentParser(
        prog="phoenix",
        description="Phoenix topology optimisation CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_opt = sub.add_parser("optimize", help="Run topology optimisation")
    p_opt.add_argument("preset", help="Path to preset JSON")
    p_opt.add_argument("--settings", "-s", default=None,
                       help="Settings JSON path")
    p_opt.add_argument("--final", choices=["best", "latest"], default="latest",
                       help="Final result: best-design or last-iteration")
    p_opt.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Density threshold")
    p_opt.add_argument("--cmap", default="plasma", help="Colormap")
    p_opt.add_argument("--no-stl", action="store_true",
                       help="Skip STL export")
    p_opt.add_argument("--no-view", action="store_true",
                       help="Skip the 3D viewer")
    p_opt.add_argument("--no-verify", action="store_true",
                       help="Skip post-optimisation verification")
    p_opt.add_argument("--no-mesh", action="store_true",
                       help="Hide the mesh overlay")
    p_opt.add_argument("--output", "-o", default=None,
                       help="Output .npz path")
    p_opt.add_argument("--spacing", type=float, default=None,
                       help="Voxel size in mm")
    p_opt.add_argument("--method", choices=["marching_cubes", "voxel"],
                       default="marching_cubes", help="STL meshing method")
    p_opt.add_argument("--smooth", choices=["laplacian", "taubin", "none"],
                       default="taubin", help="STL smoothing")

    p_exp = sub.add_parser("export", help="Convert .npz to STL")
    p_exp.add_argument("result", help="Path to result .npz")
    p_exp.add_argument("--output", "-o", default=None, help="Output STL path")
    p_exp.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Density threshold")
    p_exp.add_argument("--smooth", choices=["laplacian", "taubin", "none"],
                       default="taubin", help="Smoothing type")
    p_exp.add_argument("--smooth-iters", type=int, default=10,
                       help="Smoothing iterations")
    p_exp.add_argument("--decimate", type=float, default=None,
                       help="Decimate factor or target face count")
    p_exp.add_argument("--method", choices=["marching_cubes", "voxel"],
                       default="marching_cubes", help="Meshing method")
    p_exp.add_argument("--spacing", type=float, default=None,
                       help="Voxel size in mm")

    p_plot = sub.add_parser("plot", help="Plot history curves")
    p_plot.add_argument("result", help="Path to result .npz")
    p_plot.add_argument("--output", "-o", default=None,
                        help="Output image path")
    p_plot.add_argument("--no-show", action="store_true",
                        help="Save only, don't display")

    p_view = sub.add_parser("view", help="3D viewer")
    p_view.add_argument("result", help="Path to result .npz")
    p_view.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Density threshold")
    p_view.add_argument("--cmap", default="plasma", help="Colormap")
    p_view.add_argument("--no-mesh", action="store_true",
                        help="Hide mesh overlay")
    p_view.add_argument("--mesh-opacity", type=float, default=0.3,
                        help="Mesh overlay opacity")

    p_anim = sub.add_parser("animate", help="Cycle through snapshots")
    p_anim.add_argument("result", help="Path to result .npz")
    p_anim.add_argument("--duration", "-d", type=float, default=0.3,
                        help="Seconds per snapshot")
    p_anim.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Density threshold")
    p_anim.add_argument("--cmap", default="plasma", help="Colormap")
    p_anim.add_argument("--no-mesh", action="store_true",
                        help="Hide mesh overlay")
    p_anim.add_argument("--mesh-opacity", type=float, default=0.3,
                        help="Mesh overlay opacity")

    p_ver = sub.add_parser("verify", help="Stress-verify a result")
    p_ver.add_argument("result", help="Path to result .npz")
    p_ver.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Density threshold")
    p_ver.add_argument("--settings", "-s", default=None,
                       help="Settings JSON")
    p_ver.add_argument("--material", "-m", default=None,
                       help="Material name")
    p_ver.add_argument("--spacing", type=float, default=1.0,
                       help="Voxel spacing in mm")
    p_ver.add_argument("--no-view", action="store_true",
                       help="Skip the 3D stress viewer")
    p_ver.add_argument("--debug", action="store_true",
                       help="Print diagnostic info")

    args = parser.parse_args()

    if args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "view":
        cmd_view(args)
    elif args.command == "animate":
        cmd_animate(args)
    elif args.command == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
