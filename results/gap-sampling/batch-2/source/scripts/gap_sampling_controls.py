"""Budgeted balanced cycling through the frozen dense pool; ordinary batch MSE."""
import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from neural_geometry_scale.study import ROOT, atomic_json, check_files, digest, identity, model_for, read, sync
from scripts.gap_density_controls import source_hashes as density_sources, nested_data, overlap, full_gradient, measurements
from scripts.gap_lr_controls import fixed_metrics, lr_at, tensor_hash

STEPS = [23000, 26000, 30000]


def sources():
    return {**density_sources(), **{n: digest(ROOT/n) for n in [
        'scripts/gap_sampling_controls.py', 'tests/test_gap_sampling_controls.py']}}


def selections(n, phase=0):
    """Disjoint balanced blocks, with each pool point used once per cycle."""
    assert n in (4096, 8192, 16384, 32768, 65536)
    count = 65536 // n
    return [torch.arange(((j+phase) % count)*n, (((j+phase) % count)+1)*n) for j in range(count)]


def stream_hash(data, n, phase, start, end):
    h = hashlib.sha256()
    h.update(tensor_hash(data['points'], data['target']).encode())
    # Hash canonical exact per-update indices; coordinates are fixed and hashed above.
    choices = selections(n, phase)
    raw = [v.numpy().tobytes() for v in choices]
    for step in range(start+1, end+1):
        h.update(raw[(step-20001) % len(raw)])
    return h.hexdigest()


def freeze(protocol, dest):
    spec = read(protocol)
    assert spec['batch'] in (1, 2, 3) and spec['hypothesis'] and spec['decision_criteria']
    old = ROOT/'runs/gap-density-sweep'
    m = read(old/'manifest.json')
    for name, sha in m['source_hashes'].items():
        assert digest(ROOT/name) == sha, ('Inherited source drift', name)
    for p in m['pairs']:
        assert digest(p['checkpoint']) == p['sha256']
        saved = torch.load(p['checkpoint'], map_location='cpu', weights_only=True)
        assert saved['step'] == 20000
        assert all(int(s['step']) == 20000 for s in saved['optimizer']['state'].values())
    for a in spec['arms']:
        selections(a['points'], a['phase'])
    m.update(source_hashes=sources(), experiment='sampling-efficiency', protocol=spec,
             protocol_sha256=digest(protocol), checkpoints=STEPS,
             historical_manifest_sha256=digest(old/'manifest.json'),
             historical_summary_sha256=digest(old/'summary.json'),
             selection='At update j=1..10000 select contiguous n-point balanced blocks at index (j-1+phase) modulo (65536/n). All pool points once per cycle. Fixed deterministic order, ordinary equal-point batch MSE; this changes gradient noise and ordering, not just coverage.',
             objective='Ordinary equal-point batch MSE; chunk sums divided by batch target element count, one Adam step per update.',
             compute='10000 updates. Count batch points times updates; all 65536 pool labels retained. Report measured training and evaluation seconds separately. No storage or label savings claimed.',
             created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x', encoding='utf-8') as f:
        json.dump(m, f, indent=2); f.write('\n')


def train(c, previous, end, folder, data, arm):
    folder.mkdir(parents=True, exist_ok=False)
    device = torch.device(c['device'])
    saved = torch.load(previous, map_location=device, weights_only=True)
    assert saved['config'] == c and 20000 <= saved['step'] < end <= 30000
    model = model_for(c, device); model.load_state_dict(saved['model'])
    opt = torch.optim.Adam(model.parameters(), lr=1e-4); opt.load_state_dict(saved['optimizer'])
    initial = fixed_metrics(model, data, device)
    choices = selections(arm['points'], arm['phase'])
    # Preload views: selection overhead is fixed and counted by the enclosing run timer.
    batches = [(data['points'][i].to(device), data['target'][i].to(device)) for i in choices]
    start = saved['step']; history = []
    sync(device); began = time.perf_counter()
    with (folder/'training.jsonl').open('x', encoding='utf-8') as log:
        for step in range(start+1, end+1):
            for group in opt.param_groups: group['lr'] = lr_at(step-20000)
            points, target = batches[(step-20001) % len(batches)]
            opt.zero_grad(set_to_none=True)
            loss = full_gradient(model, points, target, 16384)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; partial attempt retained')
            opt.step()
            if step % 100 == 0 or step == end:
                row = dict(step=step, lr=lr_at(step-20000), batch_mse=loss.item(), elapsed_seconds=time.perf_counter()-began)
                history.append(row); log.write(json.dumps(row)+'\n'); log.flush()
    sync(device); seconds = time.perf_counter()-began
    result = dict(config=c, step=end, model=model.state_dict(), optimizer=opt.state_dict(),
                  sample_rng=saved['sample_rng'], schedule='cosine', origin=20000,
                  history=history, train_seconds=seconds, arm=arm)
    torch.save(result, folder/'checkpoint.pt.tmp'); (folder/'checkpoint.pt.tmp').replace(folder/'checkpoint.pt')
    atomic_json(folder/'training.json', dict(start=start, end=end, schedule='cosine', origin=20000,
        seconds=seconds, arm=arm, density=65536, batch_points=arm['points'], chunk_size=16384,
        point_presentations=(end-start)*arm['points'], parent_sha256=digest(previous),
        stream_sha256=stream_hash(data, arm['points'], arm['phase'], start, end),
        pool_sha256=tensor_hash(data['points'], data['target']),
        sample_rng_sha256=tensor_hash(saved['sample_rng']), initial_sample_rng_sha256=tensor_hash(saved['sample_rng']),
        initial_fixed=initial, final_fixed=fixed_metrics(model, data, device), history=history))
    atomic_json(folder/'training-complete.json', {n:digest(folder/n) for n in ['checkpoint.pt','training.json','training.jsonl']})


def run(manifest, out):
    m = read(manifest)
    assert m['source_hashes'] == sources()
    assert digest(ROOT/'docs/gap-controls/fresh-cases.json') == m['fresh_cases_sha256']
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'manifest.json', m)
    for name, sha in m['source_hashes'].items():
        p = out/'source'/name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT/name).read_bytes()); assert digest(p) == sha
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    atomic_json(out/'environment.json', dict(python=platform.python_version(), torch=str(torch.__version__),
        numpy=np.__version__, gpu=torch.cuda.get_device_name(), deterministic_algorithms=True, tf32=False))
    records = []
    for pair_index, pair in enumerate(m['pairs']):
        c=pair['config']; assert digest(pair['checkpoint']) == pair['sha256']
        root=out/identity(c); root.mkdir()
        data=nested_data(c); independent=nested_data(c,65536,m['independent_seed'])
        assert overlap(data['points'],independent['points']) == 0
        old=torch.load(ROOT/'runs/gap-density-sweep'/identity(c)/'nested-data.pt',weights_only=True)
        assert all(torch.equal(data[k],old[k]) for k in data)
        for name,d in [('pool.pt',data),('independent.pt',independent)]: torch.save(d,root/name)
        atomic_json(root/'data-audit.json',dict(files={n:digest(root/n) for n in ['pool.pt','independent.pt']},
            overlap=0, exact_historical_pool=True, group_counts=torch.bincount(data['groups']).tolist()))
        # Alternate control/candidate execution order across pairs to reduce order bias.
        arms=m['protocol']['arms'] if pair_index%2==0 else list(reversed(m['protocol']['arms']))
        for arm in arms:
            previous=Path(pair['checkpoint'])
            for end in STEPS:
                folder=root/arm['name']/f'step-{end}'/'attempt-0001'
                train(c,previous,end,folder,data,arm)
                r=measurements(c,folder,m,data,independent)
                previous=folder/'checkpoint.pt'
                records.append(dict(folder=str(folder),**r)); atomic_json(out/'summary.json',dict(runs=records))
                print(json.dumps(dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],step=end,
                    train_seconds=r['training']['seconds'],eval_seconds=r['evaluation_seconds'],
                    primary=r['primary_gate'],gap_signs=r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'])),flush=True)
    assert len(records)==len(m['pairs'])*len(m['protocol']['arms'])*3
    assert digest(ROOT/'docs/gap-controls/fresh-cases.json')==m['fresh_cases_sha256']
    atomic_json(out/'run-complete.json',dict(checkpoints=len(records),summary_sha256=digest(out/'summary.json'),fresh_reservation_unchanged=True))


def budget_run(manifest,out,ledger):
    """Conservative whole-process wall budget, including prep, hashes, I/O, failures."""
    ledger.parent.mkdir(parents=True,exist_ok=True)
    state=read(ledger) if ledger.exists() else dict(limit_seconds=7200,maximum_batches=3,attempts=[])
    assert all(a.get('finished') for a in state['attempts']), 'Unclosed attempt: reconcile elapsed time before more work'
    batch=read(manifest)['protocol']['batch']
    assert batch not in [a['batch'] for a in state['attempts']], 'Each frozen batch runs once; failed work retained'
    assert len(state['attempts'])<3
    remaining=7200-sum(a['charged_seconds'] for a in state['attempts'])
    assert remaining>60
    row=dict(batch=batch,manifest=str(manifest),manifest_sha256=digest(manifest),out=str(out),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),finished=False)
    state['attempts'].append(row); atomic_json(ledger,state)
    began=time.perf_counter(); code=-1
    try:
        result=subprocess.run([sys.executable,'-m','scripts.gap_sampling_controls','run','--manifest',str(manifest),'--out',str(out)], timeout=remaining-5)
        code=result.returncode
    except subprocess.TimeoutExpired:
        row['timed_out']=True
    finally:
        row.update(charged_seconds=time.perf_counter()-began,finished=True,exit_code=code,
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic_json(ledger,state)
    if code: raise SystemExit(code)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','run','budget-run'])
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--out',type=Path)
    p.add_argument('--protocol',type=Path); p.add_argument('--ledger',type=Path)
    a=p.parse_args()
    if a.command=='freeze': freeze(a.protocol,a.manifest)
    elif a.command=='run': run(a.manifest,a.out)
    else: budget_run(a.manifest,a.out,a.ledger)
