"""Continuous-training gap controls; preserve existing frozen study sources.

Run from the repository: python -m scripts.gap_training_controls freeze|run|report.
Each checkpoint contains Adam and sampling RNG state. Interrupted segments are
retained in numbered attempts; evaluation resumes without repeating training.
"""
import argparse
import csv
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

from neural_geometry_scale.geometry import Scene, batch
from neural_geometry_scale.evaluation import field_metrics, geometry_metrics
from neural_geometry_scale.study import (ROOT, atomic_json, check_files, digest,
                                         identity, model_for, read, sources, sync)


def source_hashes():
    return {**sources(), 'scripts/gap_training_controls.py': digest(__file__)}


def freeze(path, width=64, checkpoints=(1000, 3000, 10000)):
    original = read(ROOT / 'docs/local-scale/development.json')
    configs = [dict(c, width=width, steps=max(checkpoints), checkpoints=list(checkpoints))
               for c in original['runs'] if c['scene'] == 'parallel_plates'
               and c['feature'] == .02 and c['policy'] in ('global_narrow', 'local')]
    m = dict(schema=1, split='development', source_hashes=source_hashes(), runs=configs,
             baseline_manifest_sha256=digest(ROOT / 'docs/local-scale/development.json'),
             decision='All 256 original gap probes correct AND mesh structure correct at grid 192 in all three seeds per policy. Mixed outcomes remain mixed.',
             checkpoint_semantics='One continuous trajectory, fixed learning rate; no restarts between checkpoints.',
             next_control='If both policies still fail the gate, repeat with width 128; if both controls fail, compare sign-aware loss against weighted MSE on identical points.',
             dense_probe_scope='Additional diagnostic only: 64x64x9 points across the central 80% of the gap and plate footprint; no topology guarantee.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(m, f, indent=2)
        f.write('\n')


def dense_gap(model, scene, device):
    with torch.no_grad():
        xy = torch.linspace(-.48, .48, 64, device=device)
        z = torch.linspace(-.4 * scene.feature, .4 * scene.feature, 9, device=device)
        p = torch.stack(torch.meshgrid(xy, xy, z, indexing='ij'), -1).reshape(-1, 3)
        p = p @ scene.rotation(p).T
        truth, pred = scene.sdf(p), model(p)
        return dict(samples=len(p), sign_accuracy=float((pred * truth > 0).float().mean()),
                    min_prediction=float(pred.min()), mae=float((pred-truth).abs().mean()))


def train_segment(c, previous, end, folder):
    folder.mkdir(parents=True, exist_ok=False)
    device = torch.device(c['device'])
    scene = Scene(c['scene'], c['feature'], c['angle'])
    model = model_for(c, device)
    opt = torch.optim.Adam(model.parameters(), lr=c['lr'])
    gen = torch.Generator(device=device).manual_seed(c['seed'] + 1000)
    start, elapsed, history = 0, 0., []
    if previous is not None:
        saved = torch.load(previous, map_location=device, weights_only=True)
        if saved['config'] != c:
            raise RuntimeError('Checkpoint configuration mismatch')
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        gen.set_state(saved['sample_rng'].cpu())
        start, elapsed, history = saved['step'], saved['train_seconds'], saved['history']
    sync(device)
    began = time.perf_counter()
    with (folder / 'training.csv').open('w', newline='', encoding='utf-8') as f:
        log = csv.writer(f)
        log.writerow(['step', 'batch_mse', 'gap_accuracy', 'plate_0_accuracy', 'plate_1_accuracy'])
        for step in range(start + 1, end + 1):
            points, target, _ = batch(scene, c['batch_size'], c['policy'], gen, device)
            opt.zero_grad(set_to_none=True)
            loss = (model(points) - target).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; partial attempt retained')
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == end:
                with torch.no_grad():
                    probes = scene.probes(device)
                    accuracy = {k: float((model(p) * scene.sdf(p) > 0).float().mean()) for k, p in probes.items()}
                row = dict(step=step, batch_mse=loss.item(), **accuracy)
                history.append(row)
                log.writerow([step, row['batch_mse'], accuracy['gap'], accuracy['component_0'], accuracy['component_1']])
                f.flush()
    sync(device)
    elapsed += time.perf_counter() - began
    saved = dict(config=c, step=end, model=model.state_dict(), optimizer=opt.state_dict(),
                 sample_rng=gen.get_state(), history=history, train_seconds=elapsed)
    temp = folder / 'checkpoint.pt.tmp'
    torch.save(saved, temp)
    temp.replace(folder / 'checkpoint.pt')
    atomic_json(folder / 'training.json', dict(step=end, train_seconds=elapsed,
                label_queries=end*c['batch_size'], gradient_examples=end*c['batch_size'],
                parameters=sum(p.numel() for p in model.parameters()), history=history))
    atomic_json(folder / 'training-complete.json', {n: digest(folder / n) for n in
                ('checkpoint.pt', 'training.json', 'training.csv')})


def evaluate(c, folder, hashes):
    check_files(folder, read(folder / 'training-complete.json'))
    saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    if saved['config'] != c:
        raise RuntimeError('Checkpoint configuration mismatch')
    device = torch.device(c['device'])
    model = model_for(c, device)
    model.load_state_dict(saved['model'])
    model.eval()
    scene = Scene(c['scene'], c['feature'], c['angle'])
    metadata = dict(checkpoint_sha256=digest(folder / 'checkpoint.pt'), source_hashes=hashes,
                    config=c, step=saved['step'])
    result = dict(metadata=metadata, field=field_metrics(model, scene, device, c['eval_seed'], c['eval_points']),
                  dense_gap=dense_gap(model, scene, device), evaluations={})
    for resolution in c['resolutions']:
        path = folder / f'evaluation-r{resolution}.json'
        if path.exists():
            record = read(path)
            if record['metadata'] != metadata:
                raise RuntimeError('Evaluation provenance mismatch')
        else:
            record = dict(metadata=metadata, metrics=geometry_metrics(model, scene, device, resolution,
                          c['surface_samples'], c['eval_seed']))
            atomic_json(path, record)
        result['evaluations'][str(resolution)] = record['metrics']
    result['training'] = read(folder / 'training.json')
    atomic_json(folder / 'metrics.json', result)
    atomic_json(folder / 'complete.json', {p.name: digest(p) for p in folder.iterdir()
                if p.is_file() and p.name != 'complete.json' and not p.name.endswith('.tmp')})
    return result


def run(path, out):
    m = read(path)
    if m['source_hashes'] != source_hashes():
        raise RuntimeError('Source changed after freeze')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists() and read(out / 'manifest.json') != m:
        raise RuntimeError('Output belongs to another manifest')
    atomic_json(out / 'manifest.json', m)
    for name, expected in m['source_hashes'].items():
        target = out / 'source' / name
        if target.exists():
            if digest(target) != expected:
                raise RuntimeError('Source snapshot mismatch')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
    atomic_json(out / 'environment.json', dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
    torch.set_num_threads(4)
    for c in m['runs']:
        previous = None
        for step in c['checkpoints']:
            stage = out / identity(c) / f'step-{step:05d}'
            attempts = sorted(stage.glob('attempt-*'))
            finished = [a for a in attempts if (a / 'training-complete.json').exists()]
            if finished:
                folder = finished[-1]
                check_files(folder, read(folder / 'training-complete.json'))
            else:
                index = max([int(a.name.split('-')[1]) for a in attempts], default=0) + 1
                folder = stage / f'attempt-{index:04d}'
                train_segment(c, previous, step, folder)
            previous = folder / 'checkpoint.pt'
            if (folder / 'complete.json').exists():
                check_files(folder, read(folder / 'complete.json'))
                r = read(folder / 'metrics.json')
            else:
                r = evaluate(c, folder, m['source_hashes'])
            print(f"w={c['width']} {c['policy']} seed={c['seed']} step={step} "
                  f"gap={r['field']['gap']['sign_accuracy']:.4f} dense={r['dense_gap']['sign_accuracy']:.4f} "
                  f"mesh={r['evaluations'][str(max(c['resolutions']))]['mesh']}", flush=True)
            report(out)


def report(out):
    m = read(out / 'manifest.json')
    for name, expected in m['source_hashes'].items():
        if digest(out / 'source' / name) != expected:
            raise RuntimeError('Source snapshot mismatch')
    rows, evidence = [], {}
    for c in m['runs']:
        for step in c['checkpoints']:
            stage = out / identity(c) / f'step-{step:05d}'
            for folder in sorted(stage.glob('attempt-*')):
                if not (folder / 'complete.json').exists():
                    continue
                hashes = read(folder / 'complete.json')
                check_files(folder, hashes)
                r = read(folder / 'metrics.json')
                if (r['metadata']['config'] != c or r['metadata']['step'] != step
                        or r['metadata']['source_hashes'] != m['source_hashes']
                        or r['metadata']['checkpoint_sha256'] != digest(folder / 'checkpoint.pt')
                        or r['training']['label_queries'] != step*c['batch_size']):
                    raise RuntimeError('Checkpoint/config/budget mismatch')
                rows.append(r)
                evidence[folder.relative_to(out).as_posix()] = hashes
                break
    atomic_json(out / 'summary.json', dict(runs=rows))
    atomic_json(out / 'audit.json', dict(completed_checkpoints=len(rows),
                expected_checkpoints=sum(len(c['checkpoints']) for c in m['runs']), artifacts=evidence,
                source_hashes=m['source_hashes']))
    lines = ['# Longer-training gap controls', '',
             'Same 0.02 gap, seeds 31–33, fixed Adam learning rate, batch size and paired sampling streams as the original development study. Continuous trajectories with optimizer/RNG checkpoints. All seeds retained.', '',
             '| Policy | Updates | Gap accuracy mean (range) | Dense gap mean | Chamfer mean | Structure at 192 |',
             '|---|---:|---:|---:|---:|---:|']
    groups = {}
    for r in rows:
        c = r['metadata']['config']
        groups.setdefault((c['policy'], r['metadata']['step']), []).append(r)
    for (policy, step), group in sorted(groups.items()):
        gap = [r['field']['gap']['sign_accuracy'] for r in group]
        dense = [r['dense_gap']['sign_accuracy'] for r in group]
        fine = [r['evaluations'][str(max(r['metadata']['config']['resolutions']))] for r in group]
        cds = [r['symmetric_chamfer'] for r in fine if r['symmetric_chamfer'] is not None]
        success = sum(e['structure_match'] and r['field']['all_probe_signs_correct'] for e,r in zip(fine,group))
        lines.append(f"| {policy} | {step} | {statistics.mean(gap):.2%} ({min(gap):.2%}–{max(gap):.2%}) | "
                     f"{statistics.mean(dense):.2%} | {statistics.mean(cds) if cds else float('nan'):.6f} | {success}/{len(group)} |")
    lines += ['', 'Structure requires correct original probes, two components, Euler characteristic 4 and closed edges, conditional on resolved analytic reference. Dense probes cover the central gap volume, not the full gap or a topology certificate. Chamfer is sampled unsquared symmetric distance (half-sum); original 96 and 192 extraction grids retained. The 96-grid analytic reference does not resolve the 0.02 gap. Training timing includes periodic probe diagnostics and excludes mesh evaluation/checkpoint I/O.', '']
    (out / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['freeze', 'run', 'report'])
    p.add_argument('--manifest', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--checkpoints', type=int, nargs='+', default=[1000, 3000, 10000])
    args = p.parse_args()
    if args.command == 'freeze':
        freeze(args.manifest, args.width, args.checkpoints)
    elif args.command == 'run':
        run(args.manifest, args.out)
    else:
        report(args.out)


if __name__ == '__main__':
    main()
