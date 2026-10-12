"""Freeze, run, resume, audit and report the oracle bandwidth experiment.

python -m neural_geometry_scale.study --help
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

from neural_geometry.models import Field
from .geometry import Scene, SCENES, POLICIES, batch
from .evaluation import field_metrics, geometry_metrics


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sources():
    paths = list((ROOT / "neural_geometry").glob("*.py"))
    paths += list((ROOT / "neural_geometry_scale").glob("*.py"))
    paths += [ROOT / "pyproject.toml", ROOT / "requirements-windows-cu126.txt"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def validate(manifest):
    if manifest.get("schema") != 1 or not manifest.get("runs"):
        raise ValueError("Expected nonempty schema-1 manifest")
    ids = set()
    for c in manifest["runs"]:
        Scene(c["scene"], c["feature"], c["angle"])
        if c["policy"] not in POLICIES or c["model"] not in ("siren", "fourier"):
            raise ValueError("Unknown sampler/model")
        if min(c[k] for k in ("steps", "width", "depth", "eval_points", "surface_samples")) < 1 or c["batch_size"] < 2:
            raise ValueError("Positive sizes and batch size >=2 required")
        if not math.isfinite(c["lr"]) or c["lr"] <= 0:
            raise ValueError("Positive finite learning rate required")
        if c["device"] not in ("cpu", "cuda"):
            raise ValueError("Explicit cpu/cuda device required")
        if c["eval_seed"] in (c["seed"], c["seed"] + 1000) or c["seed"] < 0:
            raise ValueError("Independent nonnegative RNG seeds required")
        if not c["resolutions"] or len(set(c["resolutions"])) != len(c["resolutions"]) or min(c["resolutions"]) < 8:
            raise ValueError("Unique resolutions >=8 required")
        key = identity(c)
        if key in ids:
            raise ValueError("Duplicate configuration")
        ids.add(key)


def freeze(args):
    configs = []
    for scene, feature, width, policy, seed in itertools.product(
            args.scenes, args.features, args.widths, args.policies, args.seeds):
        configs.append(dict(scene=scene, feature=feature, width=width, policy=policy, seed=seed,
                            model=args.model, depth=3, lr=1e-4, steps=args.steps,
                            batch_size=args.batch_size, angle=args.angle, device=args.device,
                            eval_seed=50000, eval_points=args.eval_points,
                            surface_samples=args.surface_samples, resolutions=args.resolutions))
    manifest = {"schema": 1, "split": args.split, "source_hashes": sources(), "runs": configs,
                "hypothesis": "Oracle local bandwidth improves structural diagnostics at fixed label/storage budgets.",
                "primary_comparison": "local versus each fixed global control; multiscale is an additional control",
                "scope": "Development unless explicitly frozen as confirmation before tuning; no novelty claim.",
                "limits": "Oracle component thickness/face gap, unweighted MSE, analytic disjoint boxes; sampled diagnostics are not topology certificates."}
    validate(manifest)
    with args.manifest.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Frozen {len(configs)} configurations in {args.manifest}")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_for(c, device):
    torch.manual_seed(c["seed"])
    return Field(c["model"], c["width"], c["depth"]).to(device)


def train(c, out, hashes):
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(c["device"])
    scene = Scene(c["scene"], c["feature"], c["angle"])
    model = model_for(c, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c["lr"])
    generator = torch.Generator(device=device).manual_seed(c["seed"] + 1000)
    meta = {"config": c, "source_hashes": hashes, "python": platform.python_version(),
            "torch": str(torch.__version__), "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "parameters": sum(p.numel() for p in model.parameters()),
            "model_tensor_bytes": sum(p.numel() * p.element_size() for p in model.state_dict().values())}
    atomic_json(out / "config.json", meta)
    component_counts = torch.zeros(2, dtype=torch.long, device=device)
    scale_sum = torch.zeros((), device=device)
    clipped = torch.zeros((), dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    started = time.perf_counter()
    with (out / "training.csv").open("w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        log.writerow(["step", "mse"])
        for step in range(1, c["steps"] + 1):
            points, target, stats = batch(scene, c["batch_size"], c["policy"], generator, device)
            component_counts += torch.bincount(stats["components"], minlength=2)
            scale_sum += stats["scales"].sum()
            clipped += stats["clipped"].sum()
            optimizer.zero_grad(set_to_none=True)
            loss = (model(points) - target).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; partial attempt preserved")
            loss.backward()
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == c["steps"]:
                log.writerow([step, loss.item()])
                f.flush()
    sync(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    temp = out / "checkpoint.pt.tmp"
    torch.save({"model": model.state_dict(), "config": c}, temp)
    temp.replace(out / "checkpoint.pt")
    metrics = {"train_seconds": elapsed, "training_peak_cuda_allocated_bytes": peak,
               "label_queries": c["steps"] * c["batch_size"],
               "gradient_examples": c["steps"] * c["batch_size"],
               "surface_component_samples": component_counts.tolist(),
               "mean_surface_noise_sigma": scale_sum.item() / component_counts.sum().item() / 2,
               "clipped_surface_samples": clipped.item(), "final_batch_mse": loss.item(),
               "parameters": meta["parameters"], "model_tensor_bytes": meta["model_tensor_bytes"]}
    atomic_json(out / "training.json", metrics)
    atomic_json(out / "training-complete.json", {name: digest(out / name) for name in
                ("config.json", "training.json", "training.csv", "checkpoint.pt")})


def check_files(out, records):
    for name, expected in records.items():
        # Records are artifacts, never instructions to read arbitrary paths.
        if Path(name).name != name or not (out / name).is_file() or digest(out / name) != expected:
            raise RuntimeError(f"Artifact hash mismatch: {out / name}")


def evaluate(c, attempt, hashes):
    check_files(attempt, read(attempt / "training-complete.json"))
    if read(attempt / "config.json")["source_hashes"] != hashes or read(attempt / "config.json")["config"] != c:
        raise RuntimeError("Training source/config mismatch")
    checkpoint_hash = digest(attempt / "checkpoint.pt")
    saved = torch.load(attempt / "checkpoint.pt", map_location="cpu", weights_only=True)
    if saved["config"] != c:
        raise RuntimeError("Checkpoint configuration mismatch")
    device = torch.device(c["device"])
    model = model_for(c, device)
    model.load_state_dict(saved["model"])
    model.eval()
    scene = Scene(c["scene"], c["feature"], c["angle"])
    field = field_metrics(model, scene, device, c["eval_seed"], c["eval_points"])
    # Field evaluation is deterministic and cheap; replace only this derived file.
    atomic_json(attempt / "field.json", field)
    evaluations = {}
    for resolution in c["resolutions"]:
        target = attempt / f"evaluation-r{resolution}.json"
        metadata = {"checkpoint_sha256": checkpoint_hash, "source_hashes": hashes,
                    "resolution": resolution, "surface_samples": c["surface_samples"], "eval_seed": c["eval_seed"]}
        if target.exists():
            result = read(target)
            if result["metadata"] != metadata:
                raise RuntimeError(f"Evaluation provenance mismatch: {target}")
        else:
            started = time.perf_counter()
            result = {"metadata": metadata, "metrics": geometry_metrics(
                model, scene, device, resolution, c["surface_samples"], c["eval_seed"])}
            result["metrics"]["evaluation_seconds"] = time.perf_counter() - started
            atomic_json(target, result)
        evaluations[str(resolution)] = result["metrics"]
    training = read(attempt / "training.json")
    record = {"config": c, "training": training, "field": field, "evaluations": evaluations}
    atomic_json(attempt / "metrics.json", record)
    files = [p for p in attempt.iterdir() if p.is_file() and p.name != "complete.json" and not p.name.endswith(".tmp")]
    atomic_json(attempt / "complete.json", {p.name: digest(p) for p in files})
    return record


def completed(root, c):
    folder = root / identity(c)
    for attempt in sorted(folder.glob("attempt-*")) if folder.exists() else []:
        if (attempt / "complete.json").exists():
            check_files(attempt, read(attempt / "complete.json"))
            record = read(attempt / "metrics.json")
            if record["config"] != c:
                raise RuntimeError("Completed configuration mismatch")
            return attempt, record
    return None, None


def run(manifest_path, root):
    manifest = read(manifest_path)
    validate(manifest)
    hashes = sources()
    if manifest["source_hashes"] != hashes:
        raise RuntimeError("Source changed after freeze; create a fresh manifest/output")
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "manifest.json"
    if frozen.exists() and read(frozen) != manifest:
        raise RuntimeError("Existing output belongs to a different manifest")
    if not frozen.exists():
        atomic_json(frozen, manifest)
    snapshot = root / "source"
    for name, expected in hashes.items():
        p = snapshot / name
        if p.exists():
            if digest(p) != expected:
                raise RuntimeError("Source snapshot mismatch")
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((ROOT / name).read_bytes())
    torch.set_num_threads(4)
    for i, c in enumerate(manifest["runs"], 1):
        attempt, record = completed(root, c)
        if record is None:
            folder = root / identity(c)
            folder.mkdir(exist_ok=True)
            attempts = sorted(folder.glob("attempt-*"))
            resumable = [p for p in attempts if (p / "training-complete.json").exists()]
            if resumable:
                attempt = resumable[-1]
            else:
                index = max([int(p.name.split("-")[1]) for p in attempts], default=0) + 1
                attempt = folder / f"attempt-{index:04d}"
                train(c, attempt, hashes)
            record = evaluate(c, attempt, hashes)
        print(f"[{i}/{len(manifest['runs'])}] {c['scene']} f={c['feature']} w={c['width']} "
              f"{c['policy']} seed={c['seed']} CD={record['evaluations'][str(max(c['resolutions']))]['symmetric_chamfer']}", flush=True)
        report(root, plot=False)
    report(root)


def report(root, plot=True):
    manifest = read(root / "manifest.json")
    validate(manifest)
    for name, expected in manifest["source_hashes"].items():
        target = (root / "source" / name).resolve()
        if not target.is_relative_to((root / "source").resolve()) or digest(target) != expected:
            raise RuntimeError("Frozen source snapshot mismatch")
    rows, evidence, groups = [], {}, {}
    paired_counts = {}
    for c in manifest["runs"]:
        attempt, record = completed(root, c)
        if record is None:
            continue
        meta = read(attempt / "config.json")
        if meta["source_hashes"] != manifest["source_hashes"] or meta["config"] != c:
            raise RuntimeError("Manifest/run mismatch")
        if record["training"]["label_queries"] != c["steps"] * c["batch_size"]:
            raise RuntimeError("Label budget mismatch")
        if record["training"]["gradient_examples"] != c["steps"] * c["batch_size"]:
            raise RuntimeError("Gradient budget mismatch")
        pair = identity({k: v for k, v in c.items() if k != "policy"})
        counts = record["training"]["surface_component_samples"]
        if sum(counts) != c["steps"] * (c["batch_size"] // 2):
            raise RuntimeError("Surface sample accounting mismatch")
        if pair in paired_counts and paired_counts[pair] != counts:
            raise RuntimeError("Paired surface component draws differ")
        paired_counts[pair] = counts
        for res in c["resolutions"]:
            evaluation = read(attempt / f"evaluation-r{res}.json")
            if (evaluation["metadata"]["checkpoint_sha256"] != digest(attempt / "checkpoint.pt")
                    or evaluation["metadata"]["source_hashes"] != manifest["source_hashes"]
                    or evaluation["metrics"] != record["evaluations"][str(res)]):
                raise RuntimeError("Evaluation/checkpoint mismatch")
        rows.append(record)
        evidence[attempt.relative_to(root).as_posix()] = read(attempt / "complete.json")
        group = (c["scene"], c["feature"], c["width"], c["policy"], c["model"], c["steps"], c["angle"])
        groups.setdefault(group, []).append(record)
    summary = []
    lines = ["# Oracle local-scale sampling study", "", f"Split: **{manifest['split']}**. Completed **{len(rows)}/{len(manifest['runs'])}** runs.", "",
             "Unweighted SDF MSE; 50% uniform domain points and 50% area-uniform surface points plus isotropic noise. "
             "All policies share initial weights, component/face draws, uniform points and Gaussian noise within each geometry/seed. "
             "Only bandwidth changes. Local sampling has oracle component thickness and inward-face gap information.", "",
             "Chamfer is sampled, symmetric, unsquared and divided by two; missing surfaces have null distance and remain failures. "
             "Mean distances exclude missing surfaces, with valid counts shown. SD is between seeds. "
             "Structure is a diagnostic requiring two components, Euler characteristic 4, closed edges and correct probe signs, "
             "conditional on a resolved analytic reference. It is not a topology certificate.", "",
             "| Scene | Feature | Width | Policy | Grid | Chamfer mean ± SD | Valid | Structure / resolved | Missing | Train s |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for key, records in sorted(groups.items()):
        for res in sorted({r for row in records for r in row["config"]["resolutions"]}):
            vals = [r["evaluations"][str(res)] for r in records]
            distances = [v["symmetric_chamfer"] for v in vals if v["symmetric_chamfer"] is not None]
            resolved = sum(v["reference_resolved"] for v in vals)
            success = sum(v["structure_match"] and r["field"]["all_probe_signs_correct"] for v, r in zip(vals, records))
            item = dict(scene=key[0], feature=key[1], width=key[2], policy=key[3], model=key[4], steps=key[5], angle=key[6], resolution=int(res),
                        n=len(records), valid=len(distances), reference_resolved=resolved, structural_successes=success,
                        missing=sum(not v["mesh_present"] for v in vals),
                        chamfer_mean=statistics.mean(distances) if distances else None,
                        chamfer_sd=statistics.stdev(distances) if len(distances) > 1 else None,
                        train_seconds=statistics.mean(r["training"]["train_seconds"] for r in records))
            summary.append(item)
            cd = f"{item['chamfer_mean']:.6f}" if distances else "missing"
            sd = f"{item['chamfer_sd']:.6f}" if len(distances) > 1 else "n/a"
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {res} | {cd} ± {sd} | "
                         f"{len(distances)}/{len(records)} | {success}/{resolved} | {item['missing']} | {item['train_seconds']:.2f} |")
    lines += ["", "## Paired comparisons at the finest grid", "",
              "Positive reduction favours local bandwidth. Comparisons require both surfaces; missing pairs are disclosed.", "",
              "| Scene | Feature | Width | Comparator | Valid pairs / total | Mean paired reduction | Local distance wins |", "|---|---:|---:|---|---:|---:|---:|"]
    comparisons = []
    for key, local in sorted(groups.items()):
        if key[3] != "local":
            continue
        for policy in POLICIES[:-1]:
            other = groups.get((*key[:3], policy, *key[4:]), [])
            by_seed = {r["config"]["seed"]: r for r in other}
            changes, wins, total = [], 0, 0
            for row in local:
                comparator = by_seed.get(row["config"]["seed"])
                if comparator is None:
                    continue
                total += 1
                res = str(max(row["config"]["resolutions"]))
                a, b = row["evaluations"][res]["symmetric_chamfer"], comparator["evaluations"][res]["symmetric_chamfer"]
                if a is not None and b is not None and b > 0:
                    changes.append(100 * (b - a) / b)
                    wins += a < b
            reduction = statistics.mean(changes) if changes else None
            comparisons.append(dict(scene=key[0], feature=key[1], width=key[2], comparator=policy,
                                    valid_pairs=len(changes), total_pairs=total, mean_paired_reduction_percent=reduction, wins=wins))
            value = f"{reduction:.2f}%" if reduction is not None else "n/a"
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {policy} | {len(changes)}/{total} | {value} | {wins} |")
    lines += ["", "These are analytic per-shape development results unless the manifest explicitly identifies a fresh confirmation split. "
              "Three seeds support descriptive comparisons, not a broad superiority claim. The sampler changes the effective training objective; "
              "importance weighting, curvature/residual controls, estimated scale, real meshes and topology guarantees are not implemented in this study.", ""]
    atomic_json(root / "summary.json", {"groups": summary, "paired_comparisons": comparisons, "runs": rows})
    atomic_json(root / "audit.json", {"expected": len(manifest["runs"]), "completed": len(rows),
                "source_hashes": manifest["source_hashes"], "artifacts": evidence,
                "checks": ["source snapshot identity", "artifact SHA256", "training config", "label/gradient counts",
                           "paired surface component draws", "checkpoint/evaluation pairing"]})
    (root / "results.md").write_text("\n".join(lines), encoding="utf-8")
    if plot and summary:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cases = sorted({(r["scene"], r["feature"], r["width"]) for r in summary})
        finest = max(r["resolution"] for r in summary)
        fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, len(cases) * .7)), constrained_layout=True)
        for ax, metric, title in zip(axes, ("chamfer_mean", "structural_successes"),
                                      (f"Sampled surface distance (grid {finest}; lower better)", "Structural diagnostic success fraction")):
            matrix = np.full((len(cases), len(POLICIES)), np.nan)
            for row in summary:
                if row["resolution"] != finest:
                    continue
                i, j = cases.index((row["scene"], row["feature"], row["width"])), POLICIES.index(row["policy"])
                matrix[i, j] = ((row[metric] / row["reference_resolved"] if row["reference_resolved"] else np.nan)
                                if metric == "structural_successes" else (row[metric] if row[metric] is not None else np.nan))
            im = ax.imshow(matrix, aspect="auto", cmap="viridis_r" if metric == "chamfer_mean" else "viridis", vmin=0,
                           vmax=1 if metric == "structural_successes" else None)
            for i in range(len(cases)):
                for j in range(len(POLICIES)):
                    ax.text(j, i, "n/a" if np.isnan(matrix[i, j]) else f"{matrix[i,j]:.3f}", ha="center", va="center", color="black",
                            bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
            ax.set_xticks(range(len(POLICIES)), POLICIES, rotation=25, ha="right")
            ax.set_yticks(range(len(cases)), [f"{s}\nf={f}, width={w}" for s, f, w in cases])
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=.7)
        fig.suptitle(f"Oracle bandwidth study — {len(rows)} completed runs ({manifest['split']})")
        fig.savefig(root / "comparison.png", dpi=160)
        plt.close(fig)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", choices=["development", "confirmation"], default="development")
    p.add_argument("--scenes", choices=SCENES, nargs="+", default=list(SCENES))
    p.add_argument("--features", type=float, nargs="+", default=[.02, .04])
    p.add_argument("--widths", type=int, nargs="+", default=[64])
    p.add_argument("--policies", choices=POLICIES, nargs="+", default=list(POLICIES))
    p.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33])
    p.add_argument("--resolutions", type=int, nargs="+", default=[96, 192])
    p.add_argument("--model", choices=["siren", "fourier"], default="siren")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--eval-points", type=int, default=32768)
    p.add_argument("--surface-samples", type=int, default=32768)
    p.add_argument("--angle", type=float, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p = sub.add_parser("run")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "run":
        run(args.manifest, args.out)
    else:
        print(f"Verified {report(args.out)} completed runs")


if __name__ == "__main__":
    main()
