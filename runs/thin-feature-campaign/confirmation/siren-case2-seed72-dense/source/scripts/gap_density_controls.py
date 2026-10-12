"""Nested fixed-data density intervention; ordinary exact full-dataset MSE."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from neural_geometry_scale.geometry import Scene, batch
from neural_geometry_scale.evaluation import predict
from neural_geometry_scale.study import ROOT, atomic_json, check_files, digest, identity, model_for, read, sync
from scripts.gap_lr_controls import (source_hashes as lr_sources, tensor_hash, make_fixed,
                                    fixed_metrics, lr_at, evaluate)

SIZES = [4096, 16384, 65536]
STEPS = [23000, 26000, 30000]
GROUPS = ['original', 'gap', 'interior']


def source_hashes():
    return {**lr_sources(), 'scripts/gap_density_controls.py': digest(__file__)}


def nested_data(c, size=65536, seed=None):
    """Balanced 4096-point blocks preserve the old data byte-for-byte as prefix."""
    assert size > 0 and size % 4096 == 0
    scene = Scene(c['scene'], c['feature'], c['angle'])
    assert c['scene'] == 'parallel_plates' and c['feature'] == .02 and c['angle'] == 0
    gen = torch.Generator().manual_seed(c['seed'] + 91000 if seed is None else seed)
    blocks = []
    for _ in range(size // 4096):
        original = batch(scene, 2048, c['policy'], gen)[0]
        gap = (torch.rand(1024, 3, generator=gen)*2-1)*torch.tensor([.6,.6,.01])
        inside = (torch.rand(1024, 3, generator=gen)*2-1)*torch.tensor([.6,.6,1.])
        inside[:,2] = inside[:,2].sign()*(.01+.01*torch.rand(1024, generator=gen))
        blocks.append(torch.cat([original,gap,inside]))
    points = torch.cat(blocks)
    groups = torch.cat([torch.zeros(2048,dtype=torch.long),torch.ones(1024,dtype=torch.long),torch.full((1024,),2,dtype=torch.long)]).repeat(size//4096)
    return dict(points=points, target=scene.sdf(points), groups=groups)


def prefix(data, n):
    return {k: v[:n].clone() for k,v in data.items()}


def overlap(a,b):
    def packed(t):
        x=t.contiguous().numpy()
        return x.view(np.dtype((np.void,x.dtype.itemsize*x.shape[1]))).ravel()
    return int(np.intersect1d(packed(a),packed(b)).size)


@torch.no_grad()
def stratified(model, data, device):
    pred = predict(model,data['points'].to(device)).cpu().flatten().double()
    truth = data['target'].flatten().double()
    distance = truth.abs()
    error = pred-truth
    masks = {'all': torch.ones(len(truth),dtype=torch.bool)}
    masks.update({name:data['groups']==i for i,name in enumerate(GROUPS)})
    outer=(data['points'][:,:2].abs()>.48).any(-1)
    for i,name in [(1,'gap'),(2,'interior')]:
        masks[name+'_outer']=(data['groups']==i)&outer
        masks[name+'_central']=(data['groups']==i)&~outer
    result={}
    for name,mask in masks.items():
        bins={'all':mask}
        for lo,hi in [(0,.001),(.001,.0025),(.0025,.005),(.005,.01),(.01,.05),(.05,float('inf'))]:
            bins[f'{lo:g}_{hi:g}']=mask&(distance>lo)&(distance<=hi)
        result[name]={}
        for label,selected in bins.items():
            n=int(selected.sum())
            valid=selected&(distance>1e-7)
            result[name][label]=dict(samples=n,mae=float(error[selected].abs().mean()) if n else None,
                rmse=float(error[selected].square().mean().sqrt()) if n else None,
                sign_samples=int(valid.sum()),sign_errors=int((pred[valid]*truth[valid]<=0).sum()))
    return result


def full_gradient(model, points, target, chunk):
    """One full mean gradient, no optimiser step or gradient reset between chunks."""
    n=len(points)
    total=points.new_zeros(())
    for start in range(0,n,chunk):
        loss=(model(points[start:start+chunk])-target[start:start+chunk]).square().sum()/target.numel()
        loss.backward()
        total.add_(loss.detach())
    return total


def freeze(path):
    old=ROOT/'runs/gap-lr-continuations'
    previous=read(old/'manifest.json')
    records=read(old/'summary.json')['runs']
    pairs=[]
    for p in previous['pairs']:
        c=p['config']
        r=next(r for r in records if r['metadata']['config']==c and r['metadata']['step']==20000 and r['training']['schedule']=='cosine')
        folder=Path(r['folder']); check_files(folder,read(folder/'complete.json'))
        checkpoint=folder/'checkpoint.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['step']==20000 and saved['config']==c
        assert all(int(s['step'])==20000 for s in saved['optimizer']['state'].values())
        data_path=old/identity(c)/'fixed-data/data.pt'
        old_data=torch.load(data_path,weights_only=True)
        generated=make_fixed(c)
        assert all(torch.equal(old_data[k],generated[k]) for k in generated)
        pairs.append(dict(config=c,checkpoint=str(checkpoint),sha256=digest(checkpoint),
                          original_data=str(data_path),original_data_sha256=digest(data_path)))
    assert {(p['config']['policy'],p['config']['seed']) for p in pairs}=={(p,s) for p in ('global_narrow','local') for s in (31,32,33)}
    m=dict(schema=1,split='development',source_hashes=source_hashes(),pairs=pairs,sizes=SIZES,
        origin=20000,additional_updates=10000,checkpoints=STEPS,chunk_size=16384,
        objective='Ordinary equal-point MSE. Sum of chunk squared errors divided by total target elements; one Adam step per full dataset.',
        nesting='Balanced 4096-point blocks from one CPU generator seeded training_seed+91000. Exact old 4096-point dataset is prefix. Prefix counts 50% original, 25% full gap, 25% facing interior.',
        schedule='Cosine restart identical to earlier fixed-data diagnostic: update 1 = 1e-4, update 10000 = 1e-6. Adam moments and step counts retained.',
        independent_seed=61002,independent_points=65536,
        independent_distribution='Same block sampling/composition per sampler, common evaluation seed across training seeds and densities; disjoint coordinates checked.',
        resolutions=[192,384],gap_seed=previous['gap_seed'],gap_samples=previous['gap_samples'],
        primary_gate=previous['primary_gate'],
        fit_gate=previous['conditional_fixed_data']['fit_gate'],
        persistence='Unchanged structural gate at both 26000 and 30000. Report all checkpoint failures/regressions; no claim between observed checkpoints.',
        decision='Persistent restoration across all six pairs at a denser size supports coverage. Fitted finite data with failed geometry motivates between-sample bridge localisation. Unfitted dense data leaves optimisation/capacity unresolved. Mixed outcomes stay mixed.',
        compute='Match 10000 updates, NOT examples or wall time. 40,960,000 / 163,840,000 / 655,360,000 point presentations per arm: 1x / 4x / 16x. Record measured training/evaluation seconds.',
        fresh_cases_sha256=digest(ROOT/'docs/gap-controls/fresh-cases.json'),fresh_cases_evaluated=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(m,f,indent=2); f.write('\n')


def train(c, previous, end, folder, data, chunk):
    folder.mkdir(parents=True,exist_ok=False)
    device=torch.device(c['device'])
    saved=torch.load(previous,map_location=device,weights_only=True)
    assert saved['config']==c and 20000<=saved['step']<end<=30000
    model=model_for(c,device); model.load_state_dict(saved['model'])
    opt=torch.optim.Adam(model.parameters(),lr=1e-4); opt.load_state_dict(saved['optimizer'])
    initial=fixed_metrics(model,data,device)
    points,target=data['points'].to(device),data['target'].to(device)
    start=saved['step']; history=[]
    sync(device); began=time.perf_counter()
    with (folder/'training.jsonl').open('w',encoding='utf-8') as log:
        for step in range(start+1,end+1):
            lr=lr_at(step-20000)
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True)
            loss=full_gradient(model,points,target,chunk)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; partial attempt retained')
            opt.step()
            if step%100==0 or step==end:
                row=dict(step=step,lr=lr,full_mse=loss.item())
                history.append(row); log.write(json.dumps(row)+'\n'); log.flush()
    sync(device); seconds=time.perf_counter()-began
    result=dict(config=c,step=end,model=model.state_dict(),optimizer=opt.state_dict(),sample_rng=saved['sample_rng'],
        schedule='cosine',origin=20000,history=history,train_seconds=seconds,density=len(points),chunk_size=chunk)
    torch.save(result,folder/'checkpoint.pt.tmp'); (folder/'checkpoint.pt.tmp').replace(folder/'checkpoint.pt')
    atomic_json(folder/'training.json',dict(start=start,end=end,schedule='cosine',origin=20000,seconds=seconds,
        density=len(points),chunk_size=chunk,point_presentations=(end-start)*len(points),
        parent_sha256=digest(previous),stream_sha256=tensor_hash(data['points'],data['target']),
        sample_rng_sha256=tensor_hash(saved['sample_rng']),initial_sample_rng_sha256=tensor_hash(saved['sample_rng']),
        initial_fixed=initial,final_fixed=fixed_metrics(model,data,device),history=history))
    atomic_json(folder/'training-complete.json',{n:digest(folder/n) for n in ['checkpoint.pt','training.json','training.jsonl']})


def measurements(c,folder,m,data,independent):
    started=time.perf_counter()
    # Reuse the unchanged field, full-gap, mesh and gate evaluator.
    r=evaluate(c,folder,m)
    saved=torch.load(folder/'checkpoint.pt',map_location=c['device'],weights_only=True)
    model=model_for(c,torch.device(c['device'])); model.load_state_dict(saved['model']); model.eval()
    r['surface_distance_errors']=dict(training=stratified(model,data,c['device']),independent=stratified(model,independent,c['device']))
    r['independent_fit']=fixed_metrics(model,independent,c['device'])
    r['independent_points_sha256']=tensor_hash(independent['points'],independent['target'])
    sync(torch.device(c['device']))
    r['evaluation_seconds']=time.perf_counter()-started
    atomic_json(folder/'metrics.json',r)
    atomic_json(folder/'complete.json',{p.name:digest(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json' and not p.name.endswith('.tmp')})
    return r


def run(path,out):
    m=read(path)
    assert m['source_hashes']==source_hashes(), 'Frozen source changed'
    assert digest(ROOT/'docs/gap-controls/fresh-cases.json')==m['fresh_cases_sha256']
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): assert read(out/'manifest.json')==m
    else: atomic_json(out/'manifest.json',m)
    for name,sha in m['source_hashes'].items():
        dest=out/'source'/name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((ROOT/name).read_bytes())
        assert digest(dest)==sha
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    atomic_json(out/'environment.json',dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
        gpu=torch.cuda.get_device_name(),deterministic_algorithms=True,tf32=False))
    records=[]
    for pair in m['pairs']:
        c=pair['config']; checkpoint=Path(pair['checkpoint'])
        assert digest(checkpoint)==pair['sha256']
        assert digest(pair['original_data'])==pair['original_data_sha256']
        largest=nested_data(c)
        old=torch.load(pair['original_data'],weights_only=True)
        assert all(torch.equal(old[k],largest[k][:4096]) for k in old)
        independent=nested_data(c,m['independent_points'],m['independent_seed'])
        assert overlap(largest['points'],independent['points'])==0
        root=out/identity(c); root.mkdir(parents=True,exist_ok=True)
        for name,data in [('nested-data.pt',largest),('independent-data.pt',independent)]:
            dest=root/name
            if dest.exists():
                loaded=torch.load(dest,weights_only=True)
                assert all(torch.equal(data[k],loaded[k]) for k in data)
            else: torch.save(data,dest)
        atomic_json(root/'data-audit.json',dict(files={n:digest(root/n) for n in ['nested-data.pt','independent-data.pt']},
            exact_old_prefix=True,overlap=0,groups={str(n):torch.bincount(largest['groups'][:n]).tolist() for n in SIZES},
            prefixes={str(n):tensor_hash(largest['points'][:n],largest['target'][:n]) for n in SIZES}))
        # Common start; baseline geometry already measured in the prior experiment.
        baseline_path=root/'baseline-errors.json'
        if not baseline_path.exists():
            saved=torch.load(checkpoint,map_location=c['device'],weights_only=True)
            model=model_for(c,torch.device(c['device'])); model.load_state_dict(saved['model']); model.eval()
            atomic_json(baseline_path,dict(checkpoint_sha256=pair['sha256'],independent=stratified(model,independent,c['device']),
                training={str(n):stratified(model,prefix(largest,n),c['device']) for n in SIZES}))
        for n in SIZES:
            data=prefix(largest,n); previous=checkpoint
            for end in STEPS:
                stage=root/f'n-{n}'/f'step-{end}'
                attempts=sorted(stage.glob('attempt-*'))
                completed=[a for a in attempts if (a/'training-complete.json').exists()]
                if completed:
                    folder=completed[-1]; check_files(folder,read(folder/'training-complete.json'))
                    t=read(folder/'training.json')
                    assert t['parent_sha256']==digest(previous) and t['density']==n and t['end']==end
                    assert t['stream_sha256']==tensor_hash(data['points'],data['target']) and t['chunk_size']==m['chunk_size']
                else:
                    number=max([int(a.name.split('-')[1]) for a in attempts],default=0)+1
                    folder=stage/f'attempt-{number:04d}'
                    train(c,previous,end,folder,data,m['chunk_size'])
                if (folder/'complete.json').exists():
                    check_files(folder,read(folder/'complete.json'))
                    r=read(folder/'metrics.json')
                    if 'surface_distance_errors' not in r: r=measurements(c,folder,m,data,independent)
                else: r=measurements(c,folder,m,data,independent)
                previous=folder/'checkpoint.pt'
                records.append(dict(folder=str(folder),**r))
                atomic_json(out/'summary.json',dict(runs=records))
                print(json.dumps(dict(policy=c['policy'],seed=c['seed'],density=n,step=end,
                    train_seconds=r['training']['seconds'],fit=r['training']['final_fixed']['fit_gate'],
                    gap=r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'],primary=r['primary_gate'])),flush=True)
    assert len(records)==54
    assert digest(ROOT/'docs/gap-controls/fresh-cases.json')==m['fresh_cases_sha256']
    atomic_json(out/'run-complete.json',dict(checkpoints=54,fresh_reservation_unchanged=True,summary_sha256=digest(out/'summary.json')))
    print('COMPLETE: 18 density arms, 54 checkpoints; fresh cases untouched.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','run'])
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.command=='freeze': freeze(a.manifest)
    else: run(a.manifest,a.out)
