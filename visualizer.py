"""PyVista-based visualization framework for topology optimization.

Scene: plain data object, testable in isolation.
Viewer: holds the VTK plotter and pipeline, handles static and live modes.
quick_view: one-liner convenience.
"""

import time
import numpy as np
import pyvista as pv
import vtk


# Scene

class Scene:
    """Everything needed to render a topology optimization result."""

    def __init__(self, density, *,
                 load_mask=None,
                 load_direction=None,
                 load_cases=None,
                 support_mask=None,
                 spacing=(1.0, 1.0, 1.0),
                 origin=(0.0, 0.0, 0.0),
                 threshold=0.5,
                 cmap="viridis",
                 mesh_overlay=True,
                 mesh_opacity=0.3):
        self.density = density
        self.load_mask = load_mask
        self.load_direction = load_direction
        self.load_cases = load_cases
        self.support_mask = support_mask
        self.spacing = tuple(spacing)
        self.origin = tuple(origin)
        self.threshold = threshold
        self.cmap = cmap
        self.mesh_overlay = mesh_overlay
        self.mesh_opacity = mesh_opacity

    @classmethod
    def from_grid(cls, grid, threshold=0.5, cmap="viridis",
                  mesh_overlay=True, mesh_opacity=0.3):
        return cls(
            grid.density,
            load_mask=grid.load_mask,
            load_direction=grid.load_direction,
            load_cases=grid.load_cases,
            support_mask=grid.support_mask,
            spacing=grid.spacing,
            origin=grid.origin,
            threshold=threshold,
            cmap=cmap,
            mesh_overlay=mesh_overlay,
            mesh_opacity=mesh_opacity,
        )

    @classmethod
    def from_result(cls, result, threshold=0.5, cmap="viridis",
                    mesh_overlay=True, mesh_opacity=0.3):
        from ptypes import Result
        preset = result.preset
        grid = preset.to_grid()
        grid.density = result.density
        return cls.from_grid(
            grid, threshold=threshold, cmap=cmap,
            mesh_overlay=mesh_overlay, mesh_opacity=mesh_opacity,
        )


# mesh builders

def grid_to_mesh(density, threshold=0.5, spacing=(1.0, 1.0, 1.0),
                 origin=(0.0, 0.0, 0.0), smooth=True):
    """Convert a density grid to a smooth isosurface via marching cubes."""
    nx, ny, nz = density.shape
    grid_pv = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=spacing, origin=origin,
    )
    point_density = np.zeros((nx + 1, ny + 1, nz + 1), dtype=density.dtype)
    point_density[:nx, :ny, :nz] = density
    grid_pv.point_data["density"] = point_density.ravel(order="F")
    mesh = grid_pv.contour(
        isosurfaces=[threshold], scalars="density",
        method="marching_cubes",
    )
    if smooth and mesh.n_points > 0:
        mesh = mesh.smooth(n_iter=30)
    return mesh


def grid_to_heatmap_unstructured(density, threshold=0.5, cmap="viridis",
                                  spacing=(1.0, 1.0, 1.0),
                                  origin=(0.0, 0.0, 0.0)):
    """Convert a density grid to a single UnstructuredGrid of coloured cubes."""
    from matplotlib import colormaps
    cmap_obj = colormaps[cmap]

    nx, ny, nz = density.shape
    sx, sy, sz = spacing
    ox, oy, oz = origin

    xs, ys, zs = np.where(density > threshold)
    n_visible = len(xs)
    if n_visible == 0:
        return pv.UnstructuredGrid()

    values = density[xs, ys, zs]
    rgba = cmap_obj(np.clip(values, 0.0, 1.0))

    corners = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ])

    centers = np.column_stack([
        ox + (xs + 0.5) * sx,
        oy + (ys + 0.5) * sy,
        oz + (zs + 0.5) * sz,
    ])
    points = (corners[np.newaxis] * [sx, sy, sz]
              + centers[:, np.newaxis]).reshape(-1, 3)

    celltypes = np.full(n_visible, 12, dtype=np.uint8)
    cells = np.zeros((n_visible, 9), dtype=np.int64)
    cells[:, 0] = 8
    offsets = (np.arange(n_visible)[:, np.newaxis] * 8
               + np.arange(8)[np.newaxis])
    cells[:, 1:] = offsets

    ug = pv.UnstructuredGrid(cells.ravel(), celltypes, points)
    ug.cell_data["color"] = rgba[:, :3]
    return ug


# static element builders: one actor, not N actors

def _add_force_arrows(plotter, load_mask, direction, spacing, origin,
                       color="red", name="forces"):
    """Add vectorized force arrows via Glyph."""
    ix, iy, iz = np.where(load_mask)
    if len(ix) == 0:
        return
    sx, sy, sz = spacing
    ox, oy, oz = origin
    centers = np.column_stack([
        ox + (ix + 0.5) * sx,
        oy + (iy + 0.5) * sy,
        oz + (iz + 0.5) * sz,
    ])
    fx, fy, fz = direction
    mag = np.sqrt(fx**2 + fy**2 + fz**2)
    if mag < 1e-30:
        return
    norm = np.array([fx, fy, fz]) / mag
    dirs = np.tile(norm, (len(ix), 1))
    cloud = pv.PolyData(centers)
    cloud["direction"] = dirs
    cloud["magnitude"] = np.full(len(ix), mag)
    glyphs = cloud.glyph(
        orient="direction", scale="magnitude",
        factor=max(sx, sy, sz) * 3.0,
        geom=pv.Arrow(shaft_radius=0.08, tip_radius=0.2),
    )
    plotter.add_mesh(glyphs, color=color, name=name)


def _add_support_markers(plotter, support_mask, spacing, origin,
                          color="dimgray", opacity=0.6):
    """Add support cubes as a single UnstructuredGrid."""
    ix, iy, iz = np.where(support_mask)
    if len(ix) == 0:
        return
    sx, sy, sz = spacing
    ox, oy, oz = origin
    n = len(ix)
    s = min(sx, sy, sz) * 0.6 / 2.0

    corners = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ]) * s

    centers = np.column_stack([
        ox + (ix + 0.5) * sx,
        oy + (iy + 0.5) * sy,
        oz + (iz + 0.5) * sz,
    ])
    points = (corners[np.newaxis] + centers[:, np.newaxis]).reshape(-1, 3)

    celltypes = np.full(n, 12, dtype=np.uint8)
    cells = np.zeros((n, 9), dtype=np.int64)
    cells[:, 0] = 8
    offsets = (np.arange(n)[:, np.newaxis] * 8 + np.arange(8)[np.newaxis])
    cells[:, 1:] = offsets

    ug = pv.UnstructuredGrid(cells.ravel(), celltypes, points)
    plotter.add_mesh(ug, color=color, opacity=opacity, name="supports")


# Viewer

class Viewer:
    """Interactive 3D viewer for topology optimization.

    static:
        viewer = Viewer(scene); viewer.show()

    live:
        state = {"density": ..., "iteration": 0, "compliance": 0.0, "running": True}
        viewer = Viewer(scene)
        viewer.start_live(state)
    """

    def __init__(self, scene, *,
                 window_size=(1024, 768),
                 background="white",
                 title="Topology Optimization",
                 poll_interval_ms=100):
        self.scene = scene
        self._poll_interval = poll_interval_ms

        self.plotter = pv.Plotter(
            window_size=window_size, lighting="three lights",
        )
        self.plotter.set_background(background)
        self.plotter.add_title(title, font_size=12)

    def show(self, screenshot=None):
        self._add_static_elements()
        self._add_heatmap(self.scene.density)
        if self.scene.mesh_overlay:
            self._add_mesh_overlay(self.scene.density)
        if screenshot:
            self.plotter.show(screenshot=screenshot)
        else:
            self.plotter.show()

    def start_live(self, shared_state):
        """Open a live-updating viewer fed by a solver on another thread."""
        self._state = shared_state
        self._render_version = -1
        self._heatmap_actor = None
        self._mesh_actor = None

        self._add_static_elements()
        self._poll_state()

        self.plotter.show(interactive_update=True)

        try:
            last_poll = time.perf_counter()
            while True:
                self.plotter.update()

                now = time.perf_counter()
                if now - last_poll >= self._poll_interval / 1000.0:
                    last_poll = now
                    self._poll_state()

                if not self._state.get("running", True) \
                        and self._render_version == self._state.get("iteration", 0):
                    deadline = time.perf_counter() + 2.0
                    while time.perf_counter() < deadline:
                        self.plotter.update()
                    break
        finally:
            self.plotter.close()

    def _add_static_elements(self):
        s = self.scene
        nx, ny, nz = s.density.shape
        sx, sy, sz = s.spacing
        ox, oy, oz = s.origin

        if s.load_cases:
            import matplotlib
            cmap = matplotlib.colormaps["tab10"]
            for ci, case in enumerate(s.load_cases):
                color = cmap(ci % 10)[:3]
                for li, (mask, direction) in enumerate(case):
                    _add_force_arrows(
                        self.plotter, mask, direction,
                        s.spacing, s.origin,
                        color=color, name=f"forces_{ci}_{li}",
                    )
        elif s.load_mask is not None and s.load_direction is not None:
            _add_force_arrows(
                self.plotter, s.load_mask, s.load_direction,
                s.spacing, s.origin,
                name="forces_0_0",
            )

        if s.support_mask is not None:
            _add_support_markers(
                self.plotter, s.support_mask, s.spacing, s.origin,
            )

        self.plotter.add_axes(
            xlabel="X", ylabel="Y", zlabel="Z",
            line_width=2, labels_off=False,
        )
        self.plotter.add_mesh(
            pv.Box(bounds=[
                ox, ox + nx * sx,
                oy, oy + ny * sy,
                oz, oz + nz * sz,
            ]),
            color="black", style="wireframe", line_width=2,
            name="bounding_box",
        )

    def _add_heatmap(self, density):
        ug = grid_to_heatmap_unstructured(
            density, threshold=self.scene.threshold, cmap=self.scene.cmap,
            spacing=self.scene.spacing, origin=self.scene.origin,
        )
        if ug.n_cells > 0:
            self._heatmap_actor = self.plotter.add_mesh(
                ug, scalars="color", rgb=True, opacity=1.0,
                show_edges=False, name="heatmap",
            )

    def _add_mesh_overlay(self, density):
        mesh = grid_to_mesh(
            density, threshold=self.scene.threshold,
            spacing=self.scene.spacing, origin=self.scene.origin,
            smooth=True,
        )
        if mesh.n_points > 0:
            self._mesh_actor = self.plotter.add_mesh(
                mesh, color="white", opacity=self.scene.mesh_opacity,
                smooth_shading=True, name="mesh_overlay",
            )

    def _poll_state(self):
        iteration = self._state.get("iteration", 0)
        running = self._state.get("running", True)

        if iteration == self._render_version:
            return

        self._render_version = iteration
        density = self._state.get("density")
        compliance = self._state.get("compliance", 0.0)

        if density is None:
            return

        self._rebuild_dynamic(density)
        vol = density.sum() / density.size
        self.plotter.add_title(
            f"Topology Optimization -- iter {iteration}  |  "
            f"compl {compliance:.4e}  |  vol {vol:.3f}",
            font_size=12,
        )
        self.plotter.render()

    def _rebuild_dynamic(self, density):
        if self._heatmap_actor is not None:
            self.plotter.remove_actor(self._heatmap_actor)
        self._add_heatmap(density)

    def show_with_snapshots(self, snapshots, snapshot_interval=5,
                            play_interval=0.0):
        """Interactive viewer with a slider to scrub through snapshots."""
        if not snapshots:
            self.show()
            return

        self._heatmap_actor = None
        self._mesh_actor = None
        self._snapshots = snapshots
        self._snap_interval = snapshot_interval
        self._snap_idx = 0
        self._play_interval = play_interval

        self._add_static_elements()
        self._show_snapshot(0)

        def _on_slider(value):
            idx = int(round(value))
            idx = max(0, min(idx, len(snapshots) - 1))
            self._snap_idx = idx
            self._show_snapshot(idx)

        self.plotter.add_slider_widget(
            _on_slider,
            rng=[0, len(snapshots) - 1],
            value=0,
            title="Snapshot",
            pointa=(0.15, 0.02),
            pointb=(0.85, 0.02),
            style="modern",
        )

        if play_interval > 0:
            def _auto_advance(caller, event):
                self._snap_idx = (self._snap_idx + 1) % len(snapshots)
                self._show_snapshot(self._snap_idx)
                self.plotter.render()

            ms = int(play_interval * 1000)
            self.plotter.iren.add_observer("TimerEvent", _auto_advance)
            self.plotter.iren.create_timer(ms, repeating=True)

        self.plotter.show()

    def _show_snapshot(self, idx):
        density = self._snapshots[idx]
        self._rebuild_dynamic(density)
        vol = density.sum() / density.size
        snap_iter = (idx + 1) * self._snap_interval
        self.plotter.add_title(
            f"Snapshot {idx + 1} / {len(self._snapshots)}  "
            f"(iter {snap_iter})  |  vol {vol:.3f}",
            font_size=12,
        )
        self.plotter.render()


def quick_view(scene_or_grid, threshold=0.5, cmap="viridis",
               mesh_overlay=True, mesh_opacity=0.3,
               screenshot=None):
    """One-liner: open an interactive viewer.  Accepts a Scene or Grid."""
    from grid import Grid

    if isinstance(scene_or_grid, Grid):
        scene = Scene.from_grid(
            scene_or_grid, threshold=threshold, cmap=cmap,
            mesh_overlay=mesh_overlay, mesh_opacity=mesh_opacity,
        )
    elif isinstance(scene_or_grid, Scene):
        scene = scene_or_grid
    else:
        scene = Scene(
            scene_or_grid, threshold=threshold, cmap=cmap,
            mesh_overlay=mesh_overlay, mesh_opacity=mesh_opacity,
        )

    Viewer(scene).show(screenshot=screenshot)
