import numpy as np
from datetime import datetime


class Load:
    """A load applied to a rectangular region of voxels. Bounds are inclusive."""

    def __init__(self, x0, x1, y0, y1, z0, z1, direction=(0, -1, 0)):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1
        self.direction = tuple(direction)

    def to_mask(self, nx, ny, nz):
        mask = np.zeros((nx, ny, nz), dtype=bool)
        x0 = max(0, self.x0)
        x1 = min(nx - 1, self.x1)
        y0 = max(0, self.y0)
        y1 = min(ny - 1, self.y1)
        z0 = max(0, self.z0)
        z1 = min(nz - 1, self.z1)
        mask[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True
        return mask

    def to_dict(self):
        return {
            "x0": self.x0, "x1": self.x1,
            "y0": self.y0, "y1": self.y1,
            "z0": self.z0, "z1": self.z1,
            "direction": list(self.direction),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["x0"], d["x1"], d["y0"], d["y1"],
                   d["z0"], d["z1"], tuple(d["direction"]))


class Support:
    """A support (fixed boundary condition) on a rectangular region.

    constraint: "fix" (all DOFs), "fix_x", "fix_y", "fix_z", "fix_xy".
    Bounds are inclusive.
    """

    def __init__(self, x0, x1, y0, y1, z0, z1, constraint="fix"):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1
        if constraint not in ("fix", "fix_z", "fix_xy", "fix_x", "fix_y"):
            raise ValueError(f"Unknown constraint: {constraint}")
        self.constraint = constraint

    def to_mask(self, nx, ny, nz):
        mask = np.zeros((nx, ny, nz), dtype=bool)
        x0 = max(0, self.x0)
        x1 = min(nx - 1, self.x1)
        y0 = max(0, self.y0)
        y1 = min(ny - 1, self.y1)
        z0 = max(0, self.z0)
        z1 = min(nz - 1, self.z1)
        mask[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True
        return mask, self.constraint

    def to_dict(self):
        return {
            "x0": self.x0, "x1": self.x1,
            "y0": self.y0, "y1": self.y1,
            "z0": self.z0, "z1": self.z1,
            "constraint": self.constraint,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["x0"], d["x1"], d["y0"], d["y1"],
                   d["z0"], d["z1"], d["constraint"])


class FixedRegion:
    """A region locked at rho=1 (solid) or rho=0 (void), excluded from optimization."""

    def __init__(self, x0, x1, y0, y1, z0, z1, value=1.0):
        self.x0 = x0; self.x1 = x1
        self.y0 = y0; self.y1 = y1
        self.z0 = z0; self.z1 = z1
        self.value = float(value)

    def to_mask(self, nx, ny, nz):
        mask = np.zeros((nx, ny, nz), dtype=bool)
        x0 = max(0, self.x0); x1 = min(nx - 1, self.x1)
        y0 = max(0, self.y0); y1 = min(ny - 1, self.y1)
        z0 = max(0, self.z0); z1 = min(nz - 1, self.z1)
        mask[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True
        return mask

    def to_dict(self):
        return {"x0": self.x0, "x1": self.x1,
                "y0": self.y0, "y1": self.y1,
                "z0": self.z0, "z1": self.z1,
                "value": self.value}

    @classmethod
    def from_dict(cls, d):
        return cls(d["x0"], d["x1"], d["y0"], d["y1"],
                   d["z0"], d["z1"], d.get("value", 1.0))


from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Linear-elastic material with mass density.

    E in MPa (N/mm2), density in t/mm3, yield_stress in MPa or None.
    """

    name: str
    E: float
    nu: float = 0.3
    density: float = 2.7e-9
    yield_stress: float | None = None

    @property
    def density_gcm3(self) -> float:
        return self.density * 1e9


MATERIALS: dict[str, Material] = {
    "aluminum": Material("Aluminum 6061-T6", E=68900, nu=0.33, density=2.70e-9, yield_stress=276),
    "titanium": Material("Titanium Ti-6Al-4V", E=113800, nu=0.34, density=4.43e-9, yield_stress=880),
    "steel": Material("Steel AISI 4340", E=200000, nu=0.29, density=7.85e-9, yield_stress=470),
    "stainless": Material("Stainless 316L", E=193000, nu=0.30, density=8.00e-9, yield_stress=290),
    "pla": Material("PLA", E=3500, nu=0.35, density=1.24e-9, yield_stress=50),
    "abs": Material("ABS", E=2300, nu=0.35, density=1.04e-9, yield_stress=40),
    "petg": Material("PETG", E=2100, nu=0.35, density=1.27e-9, yield_stress=50),
    "nylon": Material("Nylon PA12", E=1700, nu=0.39, density=1.01e-9, yield_stress=48),
    "cf_pla": Material("Carbon Fiber PLA", E=6500, nu=0.30, density=1.30e-9, yield_stress=60),
    "pc": Material("Polycarbonate", E=2300, nu=0.37, density=1.20e-9, yield_stress=65),
}


def resolve_material(spec):
    """Resolve a material spec to a Material.

    spec can be a key into MATERIALS, a dict with E and optional nu,
    density, yield_stress, or an existing Material.
    """
    if isinstance(spec, Material):
        return spec
    if isinstance(spec, str):
        key = spec.lower().replace(" ", "_")
        if key in MATERIALS:
            return MATERIALS[key]
        for mat in MATERIALS.values():
            if mat.name.lower() == spec.lower():
                return mat
        raise ValueError(f"Unknown material: {spec!r}.  Available: {list(MATERIALS)}")
    if isinstance(spec, dict):
        return Material(name=spec.get("name", "custom"),
                        E=spec["E"],
                        nu=spec.get("nu", 0.3),
                        density=spec.get("density", 2.7e-9),
                        yield_stress=spec.get("yield"))
    raise TypeError(f"Cannot resolve material from {type(spec).__name__}: {spec!r}")


class Preset:
    """A named problem definition: geometry, loads, supports.

    The solver configuration is stored separately in
    settings/{settings_name}.json; the preset holds only a reference.

    Usage
    -----
    preset = Preset("cantilever", 40, 20, 12,
                    loads=[Load(39, 39, 0, 0, 6, 6, (0,-1,0))],
                    supports=[Support(0, 0, 0, 19, 0, 11)],
                    settings_name="structural")
    grid = preset.to_grid()
    solver = create_solver(grid, preset.settings_name)
    """

    def __init__(self, name, nx, ny, nz, loads=None, supports=None,
                 settings_name=None,
                 solid_regions=None, void_regions=None,
                 symmetry=None, load_cases=None):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.nz = nz
        if load_cases is not None:
            self.load_cases = [list(c) for c in load_cases]
        else:
            self.load_cases = [list(loads or [])]
        self.supports = list(supports or [])
        self.settings_name = settings_name
        self.solid_regions = list(solid_regions or [])
        self.void_regions = list(void_regions or [])
        self.symmetry = symmetry

    @property
    def loads(self):
        return [ld for case in self.load_cases for ld in case]

    @loads.setter
    def loads(self, value):
        self.load_cases = [list(value)]

    @property
    def solver_settings(self):
        if self.settings_name is None:
            return None
        import json
        with open(f"settings/{self.settings_name}.json") as f:
            return json.load(f)

    def to_grid(self, initial_density=0.3):
        from grid import Grid
        grid = Grid(self.nx, self.ny, self.nz,
                    initial_density=initial_density,
                    symmetry=self.symmetry)

        for case_idx, case in enumerate(self.load_cases):
            for load in case:
                grid.add_load(load, case=case_idx)

        for support in self.supports:
            grid.add_support(support)

        for r in self.solid_regions:
            grid.solid_mask |= r.to_mask(self.nx, self.ny, self.nz)
        for r in self.void_regions:
            grid.void_mask |= r.to_mask(self.nx, self.ny, self.nz)
        grid.density[grid.solid_mask] = 1.0
        grid.density[grid.void_mask] = 0.0

        return grid

    def to_dict(self):
        d = {
            "name": self.name,
            "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "supports": [s.to_dict() for s in self.supports],
        }
        if len(self.load_cases) <= 1:
            d["loads"] = [l.to_dict() for l in (self.load_cases[0] if self.load_cases else [])]
        else:
            d["load_cases"] = [[l.to_dict() for l in case] for case in self.load_cases]
        if self.settings_name is not None:
            d["settings"] = self.settings_name
        if self.solid_regions:
            d["solid_regions"] = [r.to_dict() for r in self.solid_regions]
        if self.void_regions:
            d["void_regions"] = [r.to_dict() for r in self.void_regions]
        if self.symmetry:
            d["symmetry"] = self.symmetry
        return d

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        import json
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d, resolve_scene=True):
        if "objects" in d and resolve_scene:
            from scene import Scene
            return Scene.from_dict(d).to_preset()

        settings_name = d.get("settings", d.get("solver_settings"))

        raw_sym = d.get("symmetry")
        if isinstance(raw_sym, dict):
            raw_sym = "".join(ch for ch, on in sorted(raw_sym.items()) if on) or None
        elif isinstance(raw_sym, str) and raw_sym.lower() == "none":
            raw_sym = None

        if "load_cases" in d:
            load_cases = []
            for c in d["load_cases"]:
                raw = c["loads"] if isinstance(c, dict) else c
                load_cases.append([Load.from_dict(ld) for ld in raw])
        else:
            load_cases = [[Load.from_dict(ld) for ld in d.get("loads", [])]]

        return cls(
            name=d["name"],
            nx=d["nx"], ny=d["ny"], nz=d["nz"],
            load_cases=load_cases,
            supports=[Support.from_dict(sd) for sd in d["supports"]],
            settings_name=settings_name,
            solid_regions=[FixedRegion.from_dict(r) for r in d.get("solid_regions", [])],
            void_regions=[FixedRegion.from_dict(r) for r in d.get("void_regions", [])],
            symmetry=raw_sym,
        )

    @classmethod
    def from_scene(cls, scene):
        grid = scene.to_grid()
        preset = cls(
            name=scene.name,
            nx=scene.nx, ny=scene.ny, nz=scene.nz,
            load_cases=None,
            supports=[],
            settings_name=(scene.suggested_settings
                           if isinstance(scene.suggested_settings, str)
                           else None),
            symmetry=scene.symmetry,
        )
        raw_cases = {}
        for case_idx, case in enumerate(grid.load_cases):
            for mask, direction in case:
                xs, ys, zs = np.where(mask)
                if len(xs):
                    raw_cases.setdefault(case_idx, [])
                    raw_cases[case_idx].append(
                        Load(int(xs.min()), int(xs.max()),
                             int(ys.min()), int(ys.max()),
                             int(zs.min()), int(zs.max()),
                             direction))
        if raw_cases:
            max_case = max(raw_cases)
            preset.load_cases = [raw_cases.get(i, []) for i in range(max_case + 1)]
        else:
            preset.load_cases = [[]]

        for code in ("fix", "fix_z", "fix_xy", "fix_x", "fix_y"):
            m = grid.support_mask & (grid.support_constraint_voxel == code)
            if m.any():
                xs, ys, zs = np.where(m)
                preset.supports.append(
                    Support(int(xs.min()), int(xs.max()),
                            int(ys.min()), int(ys.max()),
                            int(zs.min()), int(zs.max()),
                            constraint=code))
        if grid.solid_mask.any():
            xs, ys, zs = np.where(grid.solid_mask)
            preset.solid_regions.append(
                FixedRegion(int(xs.min()), int(xs.max()),
                            int(ys.min()), int(ys.max()),
                            int(zs.min()), int(zs.max()), value=1.0))
        if grid.void_mask.any():
            xs, ys, zs = np.where(grid.void_mask)
            preset.void_regions.append(
                FixedRegion(int(xs.min()), int(xs.max()),
                            int(ys.min()), int(ys.max()),
                            int(zs.min()), int(zs.max()), value=0.0))
        return preset

    def __repr__(self):
        return (f"Preset({self.name!r}, {self.nx}x{self.ny}x{self.nz}, "
                f"loads={len(self.loads)}, "
                f"supports={len(self.supports)}, "
                f"settings={self.settings_name!r})")


class Result:
    """The result of a topology optimization run.

    Stores the final density field, the preset used, per-iteration
    history, and optional intermediate density snapshots.
    """

    def __init__(self, density, preset, history=None,
                 density_snapshots=None):
        self.density = density
        self.preset = preset
        self.history = history or {}
        self.density_snapshots = density_snapshots or []

    @property
    def nx(self):
        return self.preset.nx

    @property
    def ny(self):
        return self.preset.ny

    @property
    def nz(self):
        return self.preset.nz

    @property
    def final_compliance(self):
        for key in ["compliance", "objective", "frequency"]:
            c = self.history.get(key, [])
            if c:
                return c[-1]
        return None

    @property
    def final_volume_fraction(self):
        return float(self.density.sum() / self.density.size)

    def save(self, path):
        import json

        save_dict = {
            "density": self.density.astype(np.float32),
            "preset_json": np.bytes_(json.dumps(self.preset.to_dict())),
            "timestamp": datetime.now().isoformat(),
        }

        for key, arr in self.history.items():
            if isinstance(arr, list) and len(arr) > 0:
                save_dict[f"history_{key}"] = np.array(arr, dtype=np.float32)

        for i, snap in enumerate(self.density_snapshots):
            save_dict[f"snapshot_iter_{i}"] = snap.astype(np.float32)

        np.savez_compressed(path, **save_dict)

    @classmethod
    def load(cls, path):
        import json

        data = np.load(path, allow_pickle=True)
        density = data["density"]

        if "preset_json" in data:
            pj = data["preset_json"]
            if hasattr(pj, 'item'):
                pj = pj.item()
            if isinstance(pj, bytes):
                pj = pj.decode("utf-8")
            preset = Preset.from_dict(json.loads(pj))
        else:
            preset = Preset(
                name=str(data.get("preset_name", "loaded")),
                nx=int(data["nx"]), ny=int(data["ny"]),
                nz=int(data["nz"]),
                loads=[], supports=[],
            )

        history = {}
        snapshots = []
        for arr_key in data.keys():
            if arr_key.startswith("history_"):
                key_name = arr_key[len("history_"):]
                history[key_name] = data[arr_key].tolist()
            elif arr_key.startswith("snapshot_iter_"):
                snapshots.append(data[arr_key])

        return cls(density, preset, history, density_snapshots=snapshots)

    def to_grid(self):
        from grid import Grid
        grid = Grid(self.nx, self.ny, self.nz, initial_density=0.0)
        grid.density = self.density.copy()
        return grid

    def __repr__(self):
        c = self.final_compliance
        compl_str = f"compliance={c:.4e}" if c else "compliance=N/A"
        return (f"Result({self.preset.name!r}, "
                f"{self.nx}x{self.ny}x{self.nz}, "
                f"vol={self.final_volume_fraction:.3f}, "
                f"{compl_str})")
