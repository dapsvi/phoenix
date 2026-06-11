"""
STL export for topology optimization results
"""

import numpy as np


def _to_image_data(density, spacing=(1.0, 1.0, 1.0),
                   origin=(0.0, 0.0, 0.0)):
    """Wrap a density array as a PyVista ImageData (shared with visualizer)"""
    import pyvista as pv

    nx, ny, nz = density.shape
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=spacing,
        origin=origin,
    )
    point_density = np.zeros((nx + 1, ny + 1, nz + 1), dtype=density.dtype)
    point_density[:nx, :ny, :nz] = density
    grid.point_data["density"] = point_density.ravel(order="F")
    return grid


def _marching_cubes_pyvista(density, threshold, spacing, origin):
    """Extract an isosurface from cell-centered density via marching cubes"""
    import pyvista as pv

    sx, sy, sz = spacing
    ox, oy, oz = origin

    padded = np.pad(density, 1, mode="constant", constant_values=0.0)
    pnx, pny, pnz = padded.shape

    pv_grid = pv.ImageData(
        dimensions=(pnx + 1, pny + 1, pnz + 1),
        spacing=spacing,
        origin=(ox - sx, oy - sy, oz - sz),
    )
    point_data = np.zeros((pnx + 1, pny + 1, pnz + 1), dtype=density.dtype)
    point_data[:pnx, :pny, :pnz] = padded
    pv_grid.point_data["density"] = point_data.ravel(order="F")
    return pv_grid.contour(
        isosurfaces=[threshold],
        scalars="density",
        method="marching_cubes",
    )


def _marching_cubes_skimage(density, threshold, spacing, origin):
    """Extract an isosurface via scikit-image's marching_cubes"""
    import pyvista as pv
    from skimage.measure import marching_cubes

    density_t = density.transpose(2, 1, 0)
    padded = np.pad(density_t, 1, mode="constant", constant_values=0.0)
    verts, faces, _, _ = marching_cubes(
        padded, level=threshold, spacing=spacing, method="lewiner",
    )
    verts -= np.array(spacing)
    verts += np.array(origin)
    if len(verts) == 0:
        return pv.PolyData()
    return pv.PolyData(verts, np.insert(faces, 0, 3, axis=1).ravel())


def _voxel_mesh(density, threshold, spacing, origin):
    """Build a mesh where every voxel above threshold is a cube"""
    import pyvista as pv

    nx, ny, nz = density.shape
    sx, sy, sz = spacing
    ox, oy, oz = origin

    xs, ys, zs = np.where(density > threshold)
    n_visible = len(xs)
    if n_visible == 0:
        return pv.PolyData()

    corners = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ], dtype=np.float64)

    tri_faces = np.array([
        [0, 3, 1], [1, 3, 2], [4, 5, 7], [5, 6, 7],
        [0, 1, 4], [1, 5, 4], [3, 7, 2], [2, 7, 6],
        [0, 4, 3], [3, 4, 7], [1, 2, 5], [2, 6, 5],
    ], dtype=np.int64)

    points = np.zeros((n_visible * 8, 3), dtype=np.float64)
    all_faces = []
    for k in range(n_visible):
        cx = ox + (xs[k] + 0.5) * sx
        cy = oy + (ys[k] + 0.5) * sy
        cz = oz + (zs[k] + 0.5) * sz
        base = k * 8
        points[base:base + 8] = corners * [sx, sy, sz] + [cx, cy, cz]
        all_faces.append(tri_faces + base)

    faces = np.hstack([np.full((len(all_faces) * 12, 1), 3, dtype=np.int64),
                        np.vstack(all_faces)]).ravel()
    mesh = pv.PolyData(points, faces)
    if mesh.n_points > 0:
        mesh = mesh.clean(tolerance=1e-6)
    return mesh


_SMOOTH_TYPES = {"laplacian", "taubin", "none"}


def _smooth_mesh(mesh, smooth_type, iterations):
    """Apply smoothing to a PyVista PolyData mesh"""
    if smooth_type == "none" or iterations <= 0 or mesh.n_points == 0:
        return mesh
    if smooth_type == "laplacian":
        return mesh.smooth(n_iter=iterations, relaxation_factor=0.2)
    elif smooth_type == "taubin":
        return mesh.smooth_taubin(
            n_iter=iterations, pass_band=0.1,
            feature_angle=60.0, normalize_coordinates=True,
        )
    else:
        raise ValueError(
            f"Unknown smooth_type: {smooth_type!r}. "
            f"Choose from {sorted(_SMOOTH_TYPES)}."
        )


def _apply_decimation(mesh, decimate):
    """Reduce triangle count and returns the possibly decimated mesh"""
    if decimate is None:
        return mesh
    n_orig = mesh.n_faces
    if isinstance(decimate, float) and 0 < decimate < 1:
        target = max(4, int(n_orig * decimate))
    elif isinstance(decimate, int) and decimate >= 4:
        target = decimate
    else:
        raise ValueError(
            f"decimate must be a float in (0,1) or int >= 4, "
            f"got {decimate!r}"
        )
    mesh_out = mesh.decimate(target_reduction=1.0 - target / n_orig)
    print(f"  Decimated: {n_orig} -> {mesh_out.n_faces} faces "
          f"({100 * mesh_out.n_faces / max(n_orig, 1):.0f}%)")
    return mesh_out


def _finalize_mesh(mesh, filename, binary):
    """Clean normals and write to STL"""
    mesh = mesh.clean()
    mesh = mesh.compute_normals(consistent_normals=True,
                                auto_orient_normals=True)
    mesh.save(filename, binary=binary)
    print(f"  Exported {mesh.n_faces} triangles to {filename}")
    return mesh


def to_stl(density, filename, threshold=0.5,
           method="marching_cubes",
           backend="pyvista",
           smooth_type="taubin",
           smooth_iterations=10,
           spacing=(1.0, 1.0, 1.0),
           origin=(0.0, 0.0, 0.0),
           decimate=None,
           binary=True):
    """Export a density grid to an STL file"""
    import pyvista as pv

    if method == "marching_cubes":
        if backend == "pyvista":
            mesh = _marching_cubes_pyvista(density, threshold, spacing, origin)
        elif backend == "skimage":
            mesh = _marching_cubes_skimage(density, threshold, spacing, origin)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")
    elif method == "voxel":
        mesh = _voxel_mesh(density, threshold, spacing, origin)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if mesh.n_points == 0:
        print(f"  WARNING: empty mesh at threshold={threshold}")
        return mesh

    if smooth_type not in _SMOOTH_TYPES:
        raise ValueError(f"Unknown smooth_type: {smooth_type!r}")

    mesh = _smooth_mesh(mesh, smooth_type, smooth_iterations)
    mesh = _apply_decimation(mesh, decimate)
    return _finalize_mesh(mesh, filename, binary)


def grid_to_stl(grid, filename, threshold=0.5, **kwargs):
    return to_stl(grid.density, filename, threshold=threshold,
                  spacing=grid.spacing, origin=grid.origin, **kwargs)


def result_to_stl(result, filename, threshold=0.5, **kwargs):
    return to_stl(result.density, filename, threshold=threshold, **kwargs)


def scene_to_stl(scene, filename, threshold=0.5, **kwargs):
    grid = scene.to_grid()
    return to_stl(grid.density, filename, threshold=threshold,
                  spacing=grid.spacing, origin=grid.origin, **kwargs)


def mesh_to_stl(mesh, filename, decimate=None, binary=True):
    mesh = _apply_decimation(mesh, decimate)
    return _finalize_mesh(mesh, filename, binary)
