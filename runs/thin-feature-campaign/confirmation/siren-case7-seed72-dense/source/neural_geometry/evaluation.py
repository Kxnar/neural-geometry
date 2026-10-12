"""Evaluate saved fields without retraining; controls use the same checkpoint."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from .geometry import sdf
from .models import Field
from .metrics import (mesh_from_volume, mesh_components, mesh_surface_samples,
                      symmetric_chamfer, analytic_surface_samples)
from .provenance import source_hashes


@torch.no_grad()
def geometry_metrics(model, args, device, write_artifacts=True):
    from .run import predict
    axis = torch.linspace(-1, 1, args.resolution, device=device)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((x.flatten(), y.flatten(), torch.zeros_like(x).flatten()), -1)
    predictions, truths = [], []
    for z in axis:
        points[:, 2] = z
        predictions.append(predict(model, points).reshape(x.shape).cpu().numpy())
        truths.append(sdf(points, args.shape, args.feature).reshape(x.shape).cpu().numpy())
    volume, truth = np.stack(predictions, axis=2), np.stack(truths, axis=2)
    vertices, faces = mesh_from_volume(volume)
    true_vertices, true_faces = mesh_from_volume(truth)
    target = analytic_surface_samples(args.surface_samples, args.shape, args.feature, args.eval_seed)
    prediction = (mesh_surface_samples(vertices, faces, args.surface_samples,
                                      np.random.default_rng(args.eval_seed + 1))
                  if len(faces) else np.empty((0, 3)))
    reference = analytic_surface_samples(args.surface_samples, args.shape, args.feature, args.eval_seed + 2)
    inside, true_inside = volume < 0, truth < 0
    union = np.logical_or(inside, true_inside).sum()
    components, true_components = mesh_components(faces), mesh_components(true_faces)
    expected = 2 if args.shape == "double_sphere" else 1
    metrics = {
        "extraction_resolution": args.resolution,
        "mesh_status": "exported" if len(faces) else "no_zero_crossing",
        "mesh_vertices": len(vertices), "mesh_faces": len(faces),
        "mesh_components": components, "reference_mesh_components": true_components,
        "analytic_expected_components": expected,
        "reference_component_count_match": true_components == expected,
        "component_count_match": bool(len(faces) and components == expected and true_components == expected),
        "symmetric_chamfer": symmetric_chamfer(prediction, target),
        "chamfer_reference_sampling_floor": symmetric_chamfer(reference, target),
        "surface_samples": args.surface_samples,
        "chamfer_definition": "0.5*(mean_pred_to_true+mean_true_to_pred); unsquared Euclidean; sampled surfaces",
        "grid_occupancy_iou": float(np.logical_and(inside, true_inside).sum() / union) if union else None,
        "grid_true_inside_count": int(true_inside.sum()),
        "grid_pred_inside_count": int(inside.sum()),
    }
    if write_artifacts:
        np.savez_compressed(args.out / "field.npz", values=volume, bounds=np.array([-1., 1.]))
        if len(faces):
            with (args.out / "mesh.obj").open("w", encoding="utf-8") as f:
                for v in vertices:
                    f.write("v " + " ".join(f"{value:.8f}" for value in v) + "\n")
                for face in faces + 1:
                    f.write("f " + " ".join(str(int(value)) for value in face) + "\n")
    return metrics


def evaluate_checkpoint(checkpoint, out, resolution=192, surface_samples=32768, device="cuda"):
    from .run import evaluate, git_revision
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    # Older project checkpoints stored torch.__version__ as this string subclass.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    args = SimpleNamespace(**{**config, "out": out, "resolution": resolution,
                              "surface_samples": surface_samples})
    dev = torch.device(device)
    model = Field(config["model"], config["width"], config["depth"],
                  config["frequencies"], config["frequency_scale"]).to(dev)
    model.load_state_dict(saved["model"])
    model.eval()
    out.mkdir(parents=True)
    started = time.perf_counter()
    metrics = evaluate(model, args, dev)
    metrics.update(geometry_metrics(model, args, dev))
    metrics["evaluation_seconds"] = time.perf_counter() - started
    metadata = {"checkpoint": str(Path(checkpoint).resolve()), "training_config": config,
                "resolution": resolution, "surface_samples": surface_samples,
                "evaluation_source": git_revision(), "source_hashes": source_hashes(), "device": device}
    (out / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=192)
    p.add_argument("--surface-samples", type=int, default=32768)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = p.parse_args()
    if args.resolution < 8 or args.surface_samples < 1:
        p.error("resolution >=8 and positive surface sample count required")
    torch.set_num_threads(4)
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.out, args.resolution,
                                       args.surface_samples, args.device), indent=2))


if __name__ == "__main__":
    main()
