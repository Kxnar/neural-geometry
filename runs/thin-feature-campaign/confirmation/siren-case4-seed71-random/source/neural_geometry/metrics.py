"""Resolution-dependent mesh diagnostics and sampled, unsquared Chamfer distance."""
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def mesh_from_volume(volume):
    if not volume.min() < 0 < volume.max():
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    vertices, faces, _, _ = marching_cubes(
        volume, level=0, spacing=(2 / (volume.shape[0] - 1),) * 3)
    return vertices - 1, faces


def mesh_components(faces):
    """Count components joined at vertices, retaining even tiny components."""
    if len(faces) == 0:
        return 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    n = int(faces.max()) + 1
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    _, labels = connected_components(graph, directed=False)
    return int(len(np.unique(labels[np.unique(faces)])))


def mesh_surface_samples(vertices, faces, count, rng):
    """Uniform area sampling, including all mesh components."""
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    if areas.sum() <= 0:
        raise ValueError("Mesh has no positive-area triangles")
    selected = triangles[rng.choice(len(faces), size=count, p=areas / areas.sum())]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    return ((1 - u[:, None]) * selected[:, 0] + (u * (1 - v))[:, None] * selected[:, 1]
            + (u * v)[:, None] * selected[:, 2])


def symmetric_chamfer(a, b):
    """Half the sum of mean nearest-neighbour Euclidean distances; no squaring."""
    if len(a) == 0 or len(b) == 0:
        return None
    return float((cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2)


def analytic_surface_samples(count, shape, feature, seed):
    """Area-uniform analytic reference, independent of training-point sampling."""
    rng = np.random.default_rng(seed)
    if shape == "torus":
        # Surface element is r*(R+r*cos(b)); reject to obtain area-uniform b.
        accepted = []
        remaining = count
        while remaining:
            b = rng.uniform(0, 2 * np.pi, size=max(64, remaining * 2))
            keep = rng.random(len(b)) < (0.55 + feature * np.cos(b)) / (0.55 + feature)
            values = b[keep][:remaining]
            accepted.append(values)
            remaining -= len(values)
        b = np.concatenate(accepted)
        a = rng.uniform(0, 2 * np.pi, count)
        radial = 0.55 + feature * np.cos(b)
        return np.column_stack((radial * np.cos(a), radial * np.sin(a), feature * np.sin(b)))
    if shape in ("sphere", "double_sphere"):
        p = rng.normal(size=(count, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        p *= 0.55 if shape == "sphere" else 0.3
        if shape == "double_sphere":
            p[:, 0] += rng.choice([-1, 1], count) * (0.3 + feature / 2)
        return p
    if shape == "thin_box":
        half = np.array([0.6, 0.6, feature])
        weights = half.prod() / half
        axis = rng.choice(3, count, p=weights / weights.sum())
        p = rng.uniform(-1, 1, (count, 3)) * half
        p[np.arange(count), axis] = rng.choice([-1, 1], count) * half[axis]
        return p
    raise ValueError(shape)
