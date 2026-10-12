"""Frozen explicit manifests, safe completed-run reuse, seed-level aggregation."""
import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

from .run import parser, run_experiment
from .provenance import source_hashes
from .evaluation import evaluate_checkpoint


def run_id(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    return f"{config['model']}-{config['shape']}-f{config['feature']}-{config['sampling']}-s{config['seed']}-{digest}"


def args_for(config, out):
    arguments = []
    for key, value in config.items():
        if key == "no_plots":
            if value:
                arguments.append("--no-plots")
        else:
            arguments.extend(["--" + key.replace("_", "-"), str(value)])
    return parser().parse_args(arguments + ["--out", str(out)])


def reusable(out, config, hashes):
    """Never treat partial, mismatched, or differently sourced runs as complete."""
    if not out.exists():
        return False
    if not (out / "metrics.json").exists() or not (out / "checkpoint.pt").exists():
        raise RuntimeError(f"Partial run preserved at {out}; use a new sweep directory")
    saved = json.loads((out / "config.json").read_text())
    if any(saved.get(k) != v for k, v in config.items()) or saved.get("source_hashes") != hashes:
        raise RuntimeError(f"Configuration/source mismatch at {out}")
    return True


def aggregate(root, configs):
    records = []
    for config in configs:
        out = root / run_id(config)
        if not (out / "metrics.json").exists():
            continue
        metrics = json.loads((out / "metrics.json").read_text())
        fine = out / "evaluation-r192" / "metrics.json"
        if fine.exists():
            metrics.update({"r192_" + k: v for k, v in json.loads(fine.read_text()).items()})
        records.append({**config, "run": out.name, **metrics})
    if not records:
        return
    fields = sorted(set().union(*(r.keys() for r in records)))
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    keys = ["model", "shape", "feature", "sampling", "width", "steps", "batch_size"]
    groups = {}
    for record in records:
        key = tuple(record[k] for k in keys)
        groups.setdefault(key, []).append(record)
    summaries = []
    metrics_to_group = ["symmetric_chamfer", "grid_occupancy_iou", "uniform_sdf_mae",
                        "interior_probe_retention", "exterior_probe_preservation", "mesh_components",
                        "component_count_match", "train_seconds", "label_queries", "gradient_examples",
                        "adaptive_scoring_points", "r192_symmetric_chamfer", "r192_mesh_components",
                        "r192_component_count_match", "r192_grid_occupancy_iou"]
    for key, rows in groups.items():
        group = {**dict(zip(keys, key)), "seeds": [r["seed"] for r in rows], "runs": len(rows),
                 "missing_surfaces": sum(r["mesh_status"] != "exported" for r in rows)}
        for metric in metrics_to_group:
            values = [float(r[metric]) for r in rows if r.get(metric) is not None]
            group[metric] = {"n": len(values), "mean": statistics.mean(values) if values else None,
                             "std": statistics.stdev(values) if len(values) > 1 else None,
                             "min": min(values) if values else None, "max": max(values) if values else None}
        summaries.append(group)
    (root / "aggregate.json").write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fine-resolution", action="store_true")
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    hashes = source_hashes()
    if manifest["source_hashes"] != hashes:
        raise RuntimeError("Manifest source hashes differ; freeze a new manifest before running")
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / "manifest.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != manifest:
            raise RuntimeError("Sweep manifest mismatch; completed work preserved")
    else:
        frozen.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    configs = manifest["runs"]
    if len(set(run_id(c) for c in configs)) != len(configs):
        raise ValueError("Duplicate configurations")
    for index, config in enumerate(configs, 1):
        out = args.out / run_id(config)
        started = time.perf_counter()
        if reusable(out, config, hashes):
            print(f"[{index}/{len(configs)}] reuse {out.name}", flush=True)
        else:
            log = args.out / (out.name + ".log")
            with log.open("w", encoding="utf-8") as f, redirect_stdout(f):
                metrics = run_experiment(args_for(config, out))
            print(f"[{index}/{len(configs)}] {out.name} CD={metrics['symmetric_chamfer']} "
                  f"components={metrics['mesh_components']} train={metrics['train_seconds']:.2f}s "
                  f"elapsed={time.perf_counter()-started:.2f}s", flush=True)
        if args.fine_resolution:
            fine = out / "evaluation-r192"
            if not fine.exists():
                evaluate_checkpoint(out / "checkpoint.pt", fine, 192, config["surface_samples"], config["device"])
            elif not (fine / "metrics.json").exists():
                raise RuntimeError(f"Partial evaluation preserved at {fine}")
            else:
                saved = json.loads((fine / "config.json").read_text())
                if saved["source_hashes"] != hashes or saved["resolution"] != 192:
                    raise RuntimeError(f"Evaluation source/configuration mismatch at {fine}")
        aggregate(args.out, configs)


if __name__ == "__main__":
    main()
