"""Train one field, evaluate independent samples, export slices and a mesh."""
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from .geometry import sdf, sample, surface_points, feature_probes, SHAPES
from .models import Field
from .provenance import source_hashes


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["siren", "fourier"], default="siren")
    p.add_argument("--shape", choices=SHAPES, default="torus")
    p.add_argument("--feature", type=float, default=0.08)
    p.add_argument("--sampling", choices=["uniform", "mixed", "adaptive"], default="mixed")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--frequencies", type=int, default=16)
    p.add_argument("--frequency-scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=10000)
    p.add_argument("--eval-points", type=int, default=32768)
    p.add_argument("--resolution", type=int, default=96)
    p.add_argument("--surface-samples", type=int, default=32768)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", type=Path, required=True)
    return p


@torch.no_grad()
def predict(model, points):
    return torch.cat([model(chunk) for chunk in points.split(65536)])


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def git_revision():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


@torch.no_grad()
def evaluate(model, args, device):
    g = torch.Generator(device=device).manual_seed(args.eval_seed)
    x, y = sample(args.eval_points, args.shape, args.feature, "uniform", g, device)
    pred = predict(model, x)
    truth_inside, pred_inside = y < 0, pred < 0
    union = (truth_inside | pred_inside).sum().item()
    intersection = (truth_inside & pred_inside).sum().item()
    surface = surface_points(args.eval_points, args.shape, args.feature, g, device)
    near = (surface + torch.randn(surface.shape, generator=g, device=device) * (args.feature / 2)).clamp(-1, 1)
    near_truth = sdf(near, args.shape, args.feature)
    interior, exterior = feature_probes(args.shape, args.feature, device)
    return {
        "uniform_sdf_mae": (pred - y).abs().mean().item(),
        "uniform_occupancy_iou": intersection / union if union else None,
        "uniform_true_inside_count": truth_inside.sum().item(),
        "surface_abs_predicted_sdf": predict(model, surface).abs().mean().item(),
        "near_surface_sdf_mae": (predict(model, near) - near_truth).abs().mean().item(),
        "interior_probe_retention": (predict(model, interior) < 0).float().mean().item(),
        "exterior_probe_preservation": (predict(model, exterior) > 0).float().mean().item(),
    }


@torch.no_grad()
def export_visuals(model, args, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis = torch.linspace(-1, 1, args.resolution, device=device)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    pts = torch.stack((x.flatten(), y.flatten(), torch.zeros_like(x).flatten()), -1)
    target = sdf(pts, args.shape, args.feature).reshape(x.shape).cpu().numpy()
    result = predict(model, pts).reshape(x.shape).cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, field, title in zip(axes, [target, result, abs(target-result)], ["Ground truth", "Prediction", "Absolute field error"]):
        im = ax.imshow(field.T, origin="lower", extent=(-1,1,-1,1),
                       cmap="magma" if title.startswith("Absolute") else "coolwarm",
                       vmin=0 if title.startswith("Absolute") else -float(abs(target).max()),
                       vmax=None if title.startswith("Absolute") else float(abs(target).max()))
        if not title.startswith("Absolute") and field.min() < 0 < field.max():
            ax.contour(axis.cpu().numpy(), axis.cpu().numpy(), field.T, levels=[0], colors="black", linewidths=1)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"{args.model} | {args.shape} | feature={args.feature} | z=0 slice")
    fig.savefig(args.out / "slice.png", dpi=150)
    plt.close(fig)
    return {}


def run_experiment(args):
    if min(args.steps, args.batch_size, args.eval_points, args.surface_samples) < 1 or args.resolution < 8:
        raise ValueError("Positive steps/batch/eval size and resolution >=8 required")
    if args.seed == args.eval_seed:
        raise ValueError("Use different training and evaluation seeds")
    # Never overwrite an existing experiment.
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator(device=device).manual_seed(args.seed)
    model = Field(args.model, args.width, args.depth, args.frequencies, args.frequency_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metadata = {**vars(args), "out": str(args.out), "device": str(device),
                "torch": str(torch.__version__), "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                "parameters": sum(p.numel() for p in model.parameters()),
                "model_tensor_bytes": sum(t.numel()*t.element_size() for t in model.state_dict().values()),
                "source": git_revision(), "source_hashes": source_hashes(),
                "adaptive_config": {"fraction": 0.25, "capacity": 4096, "refresh_steps": 50}
                    if args.sampling == "adaptive" else None,
                "scope": "Supervised analytic-SDF baseline; not full SIREN paper reproduction."}
    (args.out / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    start = time.perf_counter()
    initial_loss = None
    replay = None
    if args.sampling == "adaptive":
        from .adaptive import ErrorReplay
        replay = ErrorReplay(args.shape, args.feature, g, device)
    with (args.out / "training.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mse"])
        model.train()
        for step in range(1, args.steps + 1):
            points, target = (replay.batch(model, args.batch_size, step) if replay is not None else
                             sample(args.batch_size, args.shape, args.feature, args.sampling, g, device))
            optimizer.zero_grad(set_to_none=True)
            loss = (model(points) - target).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            if initial_loss is None:
                initial_loss = loss.item()
            if step == 1 or step % 100 == 0 or step == args.steps:
                value = loss.item()
                writer.writerow([step, value])
                f.flush()
                print(f"step={step} mse={value:.7f}", flush=True)
    sync(device)
    train_seconds = time.perf_counter() - start
    model.eval()
    evaluation_start = time.perf_counter()
    metrics = evaluate(model, args, device)
    metrics.update({"train_seconds": train_seconds, "initial_batch_mse": initial_loss,
                    "final_batch_mse": loss.item(),
                    "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else None})
    torch.save({"model": model.state_dict(), "config": metadata}, args.out / "checkpoint.pt")
    if not args.no_plots:
        export_visuals(model, args, device)
    from .evaluation import geometry_metrics
    metrics.update(geometry_metrics(model, args, device))
    metrics.update({"label_queries": args.steps * args.batch_size,
                    "gradient_examples": args.steps * args.batch_size,
                    "adaptive_scoring_points": replay.scoring_points if replay else 0,
                    "replayed_gradient_examples": replay.replayed_points if replay else 0,
                    "evaluation_seconds": time.perf_counter() - evaluation_start})
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    run_experiment(parser().parse_args())


if __name__ == "__main__":
    main()
