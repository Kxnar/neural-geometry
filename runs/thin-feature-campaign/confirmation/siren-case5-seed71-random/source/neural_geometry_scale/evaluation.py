"""Independent field probes and resolution-aware surface diagnostics."""
import numpy as np
import torch

from neural_geometry.metrics import mesh_from_volume, mesh_components, mesh_surface_samples, symmetric_chamfer


@torch.no_grad()
def predict(model, points):
    return torch.cat([model(p) for p in points.split(65536)])


def topology(vertices, faces):
    if not len(faces):
        return {"components": 0, "euler": 0, "closed_edges": False}
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return {"components": mesh_components(faces),
            "euler": int(len(np.unique(faces)) - len(unique) + len(faces)),
            "closed_edges": bool(np.all(counts == 2))}


@torch.no_grad()
def field_metrics(model, scene, device, seed, count):
    g = torch.Generator().manual_seed(seed)
    points = (torch.rand(count, 3, generator=g) * 2 - 1).to(device)
    result = {"uniform_sdf_mae": float((predict(model, points) - scene.sdf(points)).abs().mean())}
    all_correct = True
    for name, probes in scene.probes(device).items():
        truth, prediction = scene.sdf(probes), predict(model, probes)
        errors = (prediction - truth).abs()
        correct = (prediction * truth > 0).float().mean().item()
        ratio = errors / truth.abs().clamp_min(1e-12)
        result[name] = {"sign_accuracy": correct, "mae": errors.mean().item(),
                        "error_margin_ratio_p95": torch.quantile(ratio, .95).item(),
                        "error_margin_ratio_max": ratio.max().item(),
                        "samples": len(probes)}
        all_correct &= correct == 1.0
    result["all_probe_signs_correct"] = all_correct
    return result


@torch.no_grad()
def geometry_metrics(model, scene, device, resolution, count, seed):
    axis = torch.linspace(-1, 1, resolution, device=device)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([x.flatten(), y.flatten(), torch.zeros_like(x).flatten()], -1)
    predicted, reference = [], []
    for z in axis:
        points[:, 2] = z
        predicted.append(predict(model, points).reshape(resolution, resolution).cpu().numpy())
        reference.append(scene.sdf(points).reshape(resolution, resolution).cpu().numpy())
    volume, truth = np.stack(predicted, 2), np.stack(reference, 2)
    pv, pf = mesh_from_volume(volume)
    tv, tf = mesh_from_volume(truth)
    target = scene.surface(count, torch.Generator().manual_seed(seed))[0].numpy()
    independent = scene.surface(count, torch.Generator().manual_seed(seed + 1))[0].numpy()
    sampled = mesh_surface_samples(pv, pf, count, np.random.default_rng(seed + 2)) if len(pf) else np.empty((0, 3))
    pt, tt = topology(pv, pf), topology(tv, tf)
    expected = {"components": 2, "euler": 4, "closed_edges": True}
    inside, actual = volume < 0, truth < 0
    union = np.logical_or(inside, actual).sum()
    return {"resolution": resolution, "mesh_present": bool(len(pf)),
            "symmetric_chamfer": symmetric_chamfer(sampled, target),
            "reference_sampling_floor": symmetric_chamfer(independent, target),
            "mesh": pt, "reference_mesh": tt, "reference_resolved": tt == expected,
            "structure_match": pt == expected and tt == expected,
            "grid_iou": float(np.logical_and(inside, actual).sum() / union) if union else None,
            "grid_reference_inside": int(actual.sum())}
