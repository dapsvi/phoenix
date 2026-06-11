"""Scene: parametric scene definition for topology optimization.

A Scene is a list of CSG primitives (Box, Cylinder).  Each primitive has
a kind (domain role) and an optional bc (boundary condition).

Python API:
    scene = Scene("bridge", 50, 24, 10)
    scene.domain(0, 49, 0, 23, 0, 9)
    scene.support(0, 0, 0, 0, 0, 9)
    scene.load(0, 49, 23, 23, 3, 6, direction=(0, -1, 0))
    grid = scene.to_grid()

JSON:
    {"name": "bridge", "nx": 50, "ny": 24, "nz": 10,
     "objects": [
       {"type": "box", "bounds": [0,49,0,23,0,9], "kind": "solid"},
       {"type": "box", "bounds": [0,0,0,0,0,9],  "kind": "solid", "bc": "support", "constraint": "fix"}
     ]}
    scene = Scene.load("presets/bridge.json")
"""

import os
import json
import numpy as np

from grid import Grid
from ptypes import Load, Support, FixedRegion, Preset


# primitives

_OPS = {"union", "subtract", "intersect"}
_KINDS = {"solid", "void", "fixed_solid", "fixed_void"}
_BCS = {None, "load", "support"}
_CONSTRAINTS = {"fix", "fix_z", "fix_xy", "fix_x", "fix_y"}
_AXES = {"x", "y", "z"}

_TAG_TO_KIND_BC = {
    "solid":       ("solid",        None),
    "void":        ("void",         None),
    "fixed_solid": ("fixed_solid",  None),
    "fixed_void":  ("fixed_void",   None),
    "load":        ("solid",        "load"),
    "support":     ("solid",        "support"),
}


def _resolve_kind_bc(kind=None, bc=None, tag=None):
    """Resolve (kind, bc) from new-style or old-style inputs."""
    if tag is not None:
        if tag not in _TAG_TO_KIND_BC:
            raise ValueError(f"Unknown tag {tag!r}.")
        k, b = _TAG_TO_KIND_BC[tag]
        kind = kind or k
        bc = bc or b
    if kind is None:
        kind = "solid"
    if kind not in _KINDS:
        raise ValueError(f"Unknown kind {kind!r}.  Choose from {sorted(_KINDS)}.")
    if bc is not None and bc not in _BCS:
        raise ValueError(f"Unknown bc {bc!r}.  Choose from {sorted(_BCS - {None})}.")
    return kind, bc


class Box:
    """Axis-aligned box primitive. Bounds are inclusive voxel indices."""

    def __init__(self, x0, x1, y0, y1, z0, z1, *,
                 kind=None, bc=None, tag=None,
                 op="union",
                 direction=None, case=None, constraint=None):
        if op not in _OPS:
            raise ValueError(f"Unknown op {op!r}.  Choose from {sorted(_OPS)}.")
        self.kind, self.bc = _resolve_kind_bc(kind=kind, bc=bc, tag=tag)
        self.x0 = int(x0); self.x1 = int(x1)
        self.y0 = int(y0); self.y1 = int(y1)
        self.z0 = int(z0); self.z1 = int(z1)
        self.op = op
        self.direction = tuple(float(x) for x in direction) if direction else None
        self.case = int(case) if case is not None else None
        self.constraint = constraint
        if self.bc == "load" and self.direction is None:
            raise ValueError("Load Box requires direction.")
        if self.bc == "support" and self.constraint is None:
            self.constraint = "fix"
        if self.constraint is not None and self.constraint not in _CONSTRAINTS:
            raise ValueError(f"Unknown constraint {self.constraint!r}.")

    @property
    def tag(self):
        for t, (k, b) in _TAG_TO_KIND_BC.items():
            if k == self.kind and b == self.bc:
                return t
        return self.kind

    @property
    def bounds(self):
        return (self.x0, self.x1, self.y0, self.y1, self.z0, self.z1)

    def mask(self, shape):
        m = np.zeros(shape, dtype=bool)
        x0 = max(0, self.x0); x1 = min(shape[0] - 1, self.x1)
        y0 = max(0, self.y0); y1 = min(shape[1] - 1, self.y1)
        z0 = max(0, self.z0); z1 = min(shape[2] - 1, self.z1)
        if x0 <= x1 and y0 <= y1 and z0 <= z1:
            m[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True
        return m

    def to_dict(self):
        d = {"type": "box", "bounds": list(self.bounds), "kind": self.kind}
        if self.bc is not None:
            d["bc"] = self.bc
        if self.op != "union":
            d["op"] = self.op
        if self.direction is not None:
            d["direction"] = list(self.direction)
        if self.case is not None:
            d["case"] = self.case
        if self.constraint is not None:
            d["constraint"] = self.constraint
        return d

    @classmethod
    def from_dict(cls, d):
        kind, bc = _resolve_kind_bc(
            kind=d.get("kind"), bc=d.get("bc"), tag=d.get("tag"))
        bx = d["bounds"]
        return cls(bx[0], bx[1], bx[2], bx[3], bx[4], bx[5],
                   kind=kind, bc=bc,
                   op=d.get("op", "union"),
                   direction=tuple(d["direction"]) if "direction" in d else None,
                   case=d.get("case"),
                   constraint=d.get("constraint"))

    def __repr__(self):
        desc = f"Box({self.x0},{self.x1}, {self.y0},{self.y1}, {self.z0},{self.z1}"
        desc += f", kind={self.kind}"
        if self.bc:
            desc += f", bc={self.bc}"
        if self.direction:
            desc += f", dir={self.direction}"
        desc += ")"
        return desc


class Cylinder:
    """Cylinder primitive with axis along X, Y, or Z."""

    def __init__(self, *, center=(0, 0), radius=1.0, axis="z",
                 height=None, z0=None, z1=None,
                 kind=None, bc=None, tag=None,
                 op="union",
                 direction=None, case=None, constraint=None):
        if axis not in _AXES:
            raise ValueError(f"Unknown axis {axis!r}.  Choose from {sorted(_AXES)}.")
        if op not in _OPS:
            raise ValueError(f"Unknown op {op!r}.  Choose from {sorted(_OPS)}.")
        self.kind, self.bc = _resolve_kind_bc(kind=kind, bc=bc, tag=tag)
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(radius)
        self.axis = axis
        if z0 is not None and z1 is not None:
            self.z0 = int(z0); self.z1 = int(z1)
        elif height is not None:
            self.z0 = 0; self.z1 = int(height) - 1
        else:
            raise ValueError("Cylinder requires height (in voxels) or z0/z1.")
        self.op = op
        self.direction = tuple(float(x) for x in direction) if direction else None
        self.case = int(case) if case is not None else None
        self.constraint = constraint
        if self.bc == "load" and self.direction is None:
            raise ValueError("Load Cylinder requires direction.")
        if self.bc == "support" and self.constraint is None:
            self.constraint = "fix"
        if self.constraint is not None and self.constraint not in _CONSTRAINTS:
            raise ValueError(f"Unknown constraint {self.constraint!r}.")

    @property
    def tag(self):
        for t, (k, b) in _TAG_TO_KIND_BC.items():
            if k == self.kind and b == self.bc:
                return t
        return self.kind

    def mask(self, shape):
        nx, ny, nz = shape
        if self.axis == "z":
            xs = np.arange(nx, dtype=np.float64)
            ys = np.arange(ny, dtype=np.float64)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            dist = np.sqrt((X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2)
            xy_mask = dist <= self.radius
            z_mask = np.zeros(nz, dtype=bool)
            z0 = max(0, self.z0); z1 = min(nz - 1, self.z1)
            if z0 <= z1:
                z_mask[z0:z1 + 1] = True
            return xy_mask[:, :, np.newaxis] & z_mask[np.newaxis, np.newaxis, :]
        elif self.axis == "y":
            xs = np.arange(nx, dtype=np.float64)
            zs = np.arange(nz, dtype=np.float64)
            X, Z = np.meshgrid(xs, zs, indexing="ij")
            dist = np.sqrt((X - self.center[0]) ** 2 + (Z - self.center[1]) ** 2)
            xz_mask = dist <= self.radius
            y_mask = np.zeros(ny, dtype=bool)
            y0 = max(0, self.z0); y1 = min(ny - 1, self.z1)
            if y0 <= y1:
                y_mask[y0:y1 + 1] = True
            return xz_mask[:, np.newaxis, :] & y_mask[np.newaxis, :, np.newaxis]
        else:  # axis == "x"
            ys = np.arange(ny, dtype=np.float64)
            zs = np.arange(nz, dtype=np.float64)
            Y, Z = np.meshgrid(ys, zs, indexing="ij")
            dist = np.sqrt((Y - self.center[0]) ** 2 + (Z - self.center[1]) ** 2)
            yz_mask = dist <= self.radius
            x_mask = np.zeros(nx, dtype=bool)
            x0 = max(0, self.z0); x1 = min(nx - 1, self.z1)
            if x0 <= x1:
                x_mask[x0:x1 + 1] = True
            return x_mask[:, np.newaxis, np.newaxis] & yz_mask[np.newaxis, :, :]

    def to_dict(self):
        d = {"type": "cylinder", "center": list(self.center),
             "radius": self.radius, "axis": self.axis,
             "z0": self.z0, "z1": self.z1, "kind": self.kind}
        if self.bc is not None:
            d["bc"] = self.bc
        if self.op != "union":
            d["op"] = self.op
        if self.direction is not None:
            d["direction"] = list(self.direction)
        if self.case is not None:
            d["case"] = self.case
        if self.constraint is not None:
            d["constraint"] = self.constraint
        return d

    @classmethod
    def from_dict(cls, d):
        kind, bc = _resolve_kind_bc(
            kind=d.get("kind"), bc=d.get("bc"), tag=d.get("tag"))
        return cls(center=d.get("center", (0, 0)),
                   radius=d.get("radius", 1.0),
                   axis=d.get("axis", "z"),
                   z0=d.get("z0"), z1=d.get("z1"),
                   kind=kind, bc=bc,
                   op=d.get("op", "union"),
                   direction=tuple(d["direction"]) if "direction" in d else None,
                   case=d.get("case"),
                   constraint=d.get("constraint"))

    def __repr__(self):
        desc = (f"Cylinder(axis={self.axis}, center={self.center}, "
                f"r={self.radius}, z=[{self.z0},{self.z1}], kind={self.kind}")
        if self.bc:
            desc += f", bc={self.bc}"
        desc += ")"
        return desc


# helpers

def _mask_to_boxes(mask):
    """Yield (x0,x1,y0,y1,z0,z1) for each connected component in mask."""
    try:
        from scipy.ndimage import label
        lab, n = label(mask)
        for i in range(1, n + 1):
            xs, ys, zs = np.where(lab == i)
            yield (int(xs.min()), int(xs.max()),
                   int(ys.min()), int(ys.max()),
                   int(zs.min()), int(zs.max()))
    except ImportError:
        xs, ys, zs = np.where(mask)
        if len(xs):
            yield (int(xs.min()), int(xs.max()),
                   int(ys.min()), int(ys.max()),
                   int(zs.min()), int(zs.max()))


def _add_region_as_load(grid, mask, direction, case):
    for x0, x1, y0, y1, z0, z1 in _mask_to_boxes(mask):
        grid.add_load(Load(x0, x1, y0, y1, z0, z1, direction), case=case)


def _add_region_as_support(grid, mask, constraint):
    for x0, x1, y0, y1, z0, z1 in _mask_to_boxes(mask):
        grid.add_support(Support(x0, x1, y0, y1, z0, z1, constraint))


_PRIMITIVES = {"box": Box, "cylinder": Cylinder}


def _primitive_from_dict(d):
    typ = d.get("type", "box")
    cls = _PRIMITIVES.get(typ)
    if cls is None:
        raise ValueError(f"Unknown primitive type {typ!r}. "
                         f"Choose from {sorted(_PRIMITIVES)}.")
    return cls.from_dict(d)


# Scene

class Scene:
    """Parametric topology-optimization scene.

    Parameters
    ----------
    name : str
    nx, ny, nz : int
        Voxel grid dimensions.
    spacing, origin : tuple
        Voxel physical size and world origin.
    symmetry : str or None
        Symmetry constraint ("x", "xy", ...).
    suggested_settings : dict, str, or None
        Inline solver settings dict, or a path/name to a settings file.
    """

    def __init__(self, name, nx, ny, nz, *,
                 spacing=(1.0, 1.0, 1.0),
                 origin=(0.0, 0.0, 0.0),
                 symmetry=None,
                 suggested_settings=None):
        self.name = name
        self.nx = int(nx); self.ny = int(ny); self.nz = int(nz)
        self.spacing = tuple(spacing)
        self.origin = tuple(origin)
        self.symmetry = symmetry
        if isinstance(suggested_settings, (dict, str, type(None))):
            self.suggested_settings = suggested_settings
        else:
            raise TypeError(
                f"suggested_settings must be dict, str, or None; "
                f"got {type(suggested_settings).__name__}")
        self.objects = []

    @property
    def shape(self):
        return (self.nx, self.ny, self.nz)

    def add(self, obj):
        self.objects.append(obj)
        return obj

    def add_box(self, x0, x1, y0, y1, z0, z1, **kwargs):
        box = Box(x0, x1, y0, y1, z0, z1, **kwargs)
        self.objects.append(box)
        return box

    def solid(self, x0, x1, y0, y1, z0, z1):
        return self.add_box(x0, x1, y0, y1, z0, z1, kind="fixed_solid")

    def void(self, x0, x1, y0, y1, z0, z1):
        return self.add_box(x0, x1, y0, y1, z0, z1, kind="fixed_void")

    def load(self, x0, x1, y0, y1, z0, z1, direction, case=None):
        return self.add_box(x0, x1, y0, y1, z0, z1, kind="solid", bc="load",
                            direction=direction, case=case)

    def support(self, x0, x1, y0, y1, z0, z1, constraint="fix"):
        return self.add_box(x0, x1, y0, y1, z0, z1, kind="solid", bc="support",
                            constraint=constraint)

    def domain(self, x0, x1, y0, y1, z0, z1):
        return self.add_box(x0, x1, y0, y1, z0, z1, kind="solid")

    def remove(self, obj):
        self.objects.remove(obj)

    def to_grid(self, initial_density=None):
        """Evaluate CSG tree to a Grid."""
        if initial_density is None:
            initial_density = 0.3
            if isinstance(self.suggested_settings, dict):
                initial_density = self.suggested_settings.get("volfrac", 0.3)

        grid = Grid(self.nx, self.ny, self.nz,
                    initial_density=0.0,
                    spacing=self.spacing, origin=self.origin,
                    symmetry=self.symmetry)

        domain = np.zeros(grid.shape, dtype=bool)
        nop = np.zeros(grid.shape, dtype=bool)

        for obj in self.objects:
            m = obj.mask(grid.shape)
            if not m.any():
                continue

            if obj.kind == "solid":
                if obj.op == "union":
                    domain |= m
                elif obj.op == "subtract":
                    domain &= ~m
                else:  # intersect
                    if domain.any():
                        domain = domain & m
                    else:
                        domain = m
            elif obj.kind == "void":
                nop |= m
            elif obj.kind == "fixed_solid":
                grid.solid_mask |= m
                domain |= m
            elif obj.kind == "fixed_void":
                grid.void_mask |= m

            if obj.bc == "load":
                if obj.direction is None:
                    raise ValueError(f"Load {obj} has no direction")
                _add_region_as_load(grid, m, obj.direction, obj.case or 0)
            elif obj.bc == "support":
                _add_region_as_support(grid, m, obj.constraint or "fix")

        domain &= ~nop
        if domain.any():
            grid.density[domain] = initial_density
        grid.density[grid.solid_mask] = 1.0
        grid.density[grid.void_mask] = 0.0

        return grid

    def to_preset(self):
        return Preset.from_scene(self)

    def to_dict(self):
        d = {
            "name": self.name,
            "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "objects": [obj.to_dict() for obj in self.objects],
        }
        if self.spacing != (1.0, 1.0, 1.0):
            d["spacing"] = list(self.spacing)
        if self.origin != (0.0, 0.0, 0.0):
            d["origin"] = list(self.origin)
        if self.symmetry:
            d["symmetry"] = self.symmetry
        if self.suggested_settings is not None:
            d["suggested_settings"] = self.suggested_settings
        return d

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Scene saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d):
        scene = cls(
            name=d["name"],
            nx=d["nx"], ny=d["ny"], nz=d["nz"],
            spacing=tuple(d.get("spacing", (1.0, 1.0, 1.0))),
            origin=tuple(d.get("origin", (0.0, 0.0, 0.0))),
            symmetry=d.get("symmetry"),
            suggested_settings=d.get("suggested_settings"),
        )
        for obj_d in d.get("objects", []):
            scene.add(_primitive_from_dict(obj_d))
        return scene

    @classmethod
    def from_preset(cls, preset: Preset):
        scene = cls(preset.name, preset.nx, preset.ny, preset.nz,
                    symmetry=preset.symmetry,
                    suggested_settings=preset.settings_name)
        scene.add(Box(0, preset.nx - 1, 0, preset.ny - 1, 0, preset.nz - 1,
                      kind="solid"))
        for case_idx, case in enumerate(preset.load_cases):
            for ld in case:
                scene.add(Box(ld.x0, ld.x1, ld.y0, ld.y1, ld.z0, ld.z1,
                              kind="solid", bc="load",
                              direction=ld.direction, case=case_idx))
        for sp in preset.supports:
            scene.add(Box(sp.x0, sp.x1, sp.y0, sp.y1, sp.z0, sp.z1,
                          kind="solid", bc="support",
                          constraint=sp.constraint))
        for r in preset.solid_regions:
            scene.add(Box(r.x0, r.x1, r.y0, r.y1, r.z0, r.z1,
                          kind="fixed_solid"))
        for r in preset.void_regions:
            scene.add(Box(r.x0, r.x1, r.y0, r.y1, r.z0, r.z1,
                          kind="fixed_void"))
        return scene

    @classmethod
    def from_grid(cls, grid: Grid):
        scene = cls("from_grid", grid.nx, grid.ny, grid.nz,
                    spacing=grid.spacing, origin=grid.origin,
                    symmetry=grid.symmetry)
        designable = (grid.density > 0.0) | (grid.solid_mask | grid.void_mask)
        if designable.any():
            scene.add(Box(0, grid.nx - 1, 0, grid.ny - 1, 0, grid.nz - 1,
                          kind="solid"))
        for case_idx, case in enumerate(grid.load_cases):
            for mask, direction in case:
                xs, ys, zs = np.where(mask)
                if len(xs):
                    scene.add(Box(int(xs.min()), int(xs.max()),
                                  int(ys.min()), int(ys.max()),
                                  int(zs.min()), int(zs.max()),
                                  kind="solid", bc="load",
                                  direction=direction, case=case_idx))
        for code in ("fix", "fix_z", "fix_xy", "fix_x", "fix_y"):
            m = grid.support_mask & (grid.support_constraint_voxel == code)
            if m.any():
                xs, ys, zs = np.where(m)
                scene.add(Box(int(xs.min()), int(xs.max()),
                              int(ys.min()), int(ys.max()),
                              int(zs.min()), int(zs.max()),
                              kind="solid", bc="support", constraint=code))
        if grid.solid_mask.any():
            xs, ys, zs = np.where(grid.solid_mask)
            scene.add(Box(int(xs.min()), int(xs.max()),
                          int(ys.min()), int(ys.max()),
                          int(zs.min()), int(zs.max()),
                          kind="fixed_solid"))
        if grid.void_mask.any():
            xs, ys, zs = np.where(grid.void_mask)
            scene.add(Box(int(xs.min()), int(xs.max()),
                          int(ys.min()), int(ys.max()),
                          int(zs.min()), int(zs.max()),
                          kind="fixed_void"))
        return scene

    def __repr__(self):
        kinds = {}
        for obj in self.objects:
            label = obj.kind
            if obj.bc:
                label += f"+{obj.bc}"
            kinds[label] = kinds.get(label, 0) + 1
        parts = [f"Scene({self.name!r}, {self.nx}x{self.ny}x{self.nz}"]
        if self.symmetry:
            parts.append(f", sym={self.symmetry}")
        parts.append(")")
        parts.append("  Objects: " + ", ".join(f"{n}x{k}"
                     for k, n in sorted(kinds.items())))
        if self.suggested_settings is not None:
            if isinstance(self.suggested_settings, dict):
                parts.append(f"  Settings: inline "
                             f"({self.suggested_settings.get('type', '?')})")
            else:
                parts.append(f"  Settings: {self.suggested_settings}")
        return "\n".join(parts)
