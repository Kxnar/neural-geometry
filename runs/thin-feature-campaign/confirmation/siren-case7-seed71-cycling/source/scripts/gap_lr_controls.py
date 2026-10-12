"""Matched LR continuations and conditional fixed-data MSE diagnostic."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from neural_geometry_scale.geometry import Scene, batch
from neural_geometry_scale.evaluation import field_metrics, geometry_metrics, predict
from neural_geometry_scale.study import ROOT, atomic_json, check_files, digest, identity, model_for, read, sync
from scripts.gap_training_controls import source_hashes as base_sources


def source_hashes():
    return {**base_sources(), 'scripts/gap_lr_controls.py': digest(__file__)}


def lr_at(update, total=10000, schedule='cosine'):
    if not 1 <= update <= total:
        raise ValueError('Update outside schedule')
    if schedule == 'constant':
        return 1e-4
    if schedule != 'cosine':
        raise ValueError(schedule)
    return 1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * (update-1)/(total-1)))


def tensor_hash(*values):
    h = hashlib.sha256()
    for value in values:
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def freeze(path):
    old = ROOT/'runs/gap-controls-wide'
    baseline = read(old/'manifest.json')
    pairs = []
    for c in baseline['runs']:
        assert c['width'] == 128 and c['steps'] == 10000 and 'loss' not in c
        folders = [p for p in sorted((old/identity(c)/'step-10000').glob('attempt-*')) if (p/'complete.json').exists()]
        assert len(folders) == 1
        folder = folders[0]
        check_files(folder, read(folder/'complete.json'))
        saved = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
        assert saved['config'] == c and saved['step'] == 10000
        pairs.append(dict(config=c, checkpoint=str(folder/'checkpoint.pt'), sha256=digest(folder/'checkpoint.pt')))
    assert {(p['config']['policy'], p['config']['seed']) for p in pairs} == {(p,s) for p in ('local','global_narrow') for s in (31,32,33)}
    m = dict(schema=1, split='development', source_hashes=source_hashes(), pairs=pairs,
             schedules=['constant','cosine'], checkpoints=[13000,16000,20000], additional_updates=10000,
             lr_definition='Update j=1..10000: constant 1e-4; cosine 1e-6 + (1e-4-1e-6)/2*(1+cos(pi*(j-1)/9999)). Both endpoints used.',
             resolutions=[192,384], gap_seed=61000, gap_samples=131072,
             gap_margin_edges=[0,.001,.0025,.005,.011], numerical_margin_exclusion=1e-7,
             primary_gate='All original gap and plate probes correct; two components, Euler 4, closed edges, resolved analytic reference at BOTH 192 and 384.',
             persistence='Report each checkpoint; sustained observed success requires 16000 AND 20000; earliest suffix of >=2 passing checkpoints is interval-censored, not continuous-time certification.',
             consistent_repair='All six cosine runs pass primary gate and all full-footprint sampled gap signs at both 16000 and 20000. No fresh-case evaluation in this experiment.',
             conditional_fixed_data=dict(trigger='If consistent_repair fails', starts='six cosine 20000 checkpoints; retain Adam',
                 points=4096, composition='2048 original sampler + 1024 uniform full-gap + 1024 facing plate interior within 0.01 of gap surface; seed=seed+91000 CPU',
                 objective='Ordinary equal-point MSE, exact full batch at every update; fixed coordinates and analytic targets saved once. This changes data coverage, not the loss.',
                 updates=10000, schedule='same cosine 1e-4 to 1e-6, endpoints included', checkpoints=[23000,26000,30000],
                 fit_gate='Each of original/gap/interior subsets RMSE <=0.001 and 100% training signs for abs(true SDF)>0.001. Descriptive diagnostic only; failure is not representational impossibility.'),
             fresh_cases_sha256=digest(ROOT/'docs/gap-controls/fresh-cases.json'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(m,f,indent=2)
        f.write('\n')


@torch.no_grad()
def full_gap(model, scene, device, seed=61000, count=131072):
    g = torch.Generator().manual_seed(seed)
    p = (torch.rand(count,3,generator=g)*2-1)*torch.tensor([.6,.6,scene.feature/2])
    truth = scene.sdf(p).flatten()
    valid = truth > 1e-7
    p, truth = p[valid], truth[valid]
    pred = predict(model,p.to(device)).cpu().flatten()
    outer = (p[:,:2].abs()>.48).any(-1)
    regions = dict(whole_gap=torch.ones(len(p),dtype=torch.bool), central_footprint=~outer, outer_footprint=outer)
    result = {}
    for name, region in regions.items():
        masks = {'all_margins':region}
        for low,high in [(0,.001),(.001,.0025),(.0025,.005),(.005,.011)]:
            masks[f'margin_{low:g}_{high:g}'] = region & (truth>low) & (truth<=high)
        result[name] = {}
        for label,mask in masks.items():
            result[name][label] = dict(samples=int(mask.sum()), errors=int((pred[mask]<=0).sum()),
                sign_accuracy=float((pred[mask]>0).float().mean()), sdf_mae=float((pred[mask]-truth[mask]).abs().mean()))
    return dict(seed=seed, excluded=int((~valid).sum()), points_sha256=tensor_hash(p,truth), regions=result)


def make_fixed(c):
    scene = Scene(c['scene'],c['feature'],c['angle'])
    gen = torch.Generator().manual_seed(c['seed']+91000)
    original = batch(scene,2048,c['policy'],gen)[0]
    gap = (torch.rand(1024,3,generator=gen)*2-1)*torch.tensor([.6,.6,.01])
    inside = (torch.rand(1024,3,generator=gen)*2-1)*torch.tensor([.6,.6,1.])
    inside[:,2] = inside[:,2].sign()*(.01 + .01*torch.rand(1024,generator=gen))
    points = torch.cat([original,gap,inside])
    return dict(points=points,target=scene.sdf(points),groups=torch.cat([torch.zeros(2048,dtype=torch.long),torch.ones(1024,dtype=torch.long),torch.full((1024,),2,dtype=torch.long)]))


@torch.no_grad()
def fixed_metrics(model,data,device):
    pred=predict(model,data['points'].to(device)).cpu()
    target=data['target']
    result={}
    for i,name in enumerate(['original','gap','interior']):
        mask=data['groups']==i
        strong=mask & (target.flatten().abs()>.001)
        result[name]=dict(samples=int(mask.sum()),mse=float((pred[mask]-target[mask]).square().mean()),
            rmse=float((pred[mask]-target[mask]).square().mean().sqrt()),
            max_abs_error=float((pred[mask]-target[mask]).abs().max()),
            sign_errors=int((pred[mask]*target[mask]<=0).sum()),
            margin_gt_001_samples=int(strong.sum()),margin_gt_001_errors=int((pred[strong]*target[strong]<=0).sum()))
    result['fit_gate']=all(r['rmse']<=.001 and r['margin_gt_001_errors']==0 for r in result.values())
    return result


def train_segment(c, previous, end, folder, schedule, origin, data=None):
    folder.mkdir(parents=True,exist_ok=False)
    device=torch.device(c['device'])
    saved=torch.load(previous,map_location=device,weights_only=True)
    assert saved['config']==c and origin<=saved['step']<end<=origin+10000
    model=model_for(c,device)
    model.load_state_dict(saved['model'])
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    opt.load_state_dict(saved['optimizer'])
    gen=torch.Generator(device=device)
    gen.set_state(saved['sample_rng'].cpu())
    scene=Scene(c['scene'],c['feature'],c['angle'])
    h=hashlib.sha256()
    start=saved['step']
    history=[]
    sync(device)
    began=time.perf_counter()
    initial_fixed=fixed_metrics(model,data,device) if data is not None else None
    if data is not None:
        fixed_p,fixed_t=data['points'].to(device),data['target'].to(device)
    with (folder/'training.jsonl').open('w',encoding='utf-8') as log:
        for step in range(start+1,end+1):
            lr=lr_at(step-origin,schedule=schedule)
            for group in opt.param_groups:
                group['lr']=lr
            if data is None:
                points,target,_=batch(scene,c['batch_size'],c['policy'],gen,device)
                h.update(points.detach().cpu().numpy().tobytes())
                h.update(target.detach().cpu().numpy().tobytes())
            else:
                points,target=fixed_p,fixed_t
            opt.zero_grad(set_to_none=True)
            loss=(model(points)-target).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; retain partial attempt')
            loss.backward()
            opt.step()
            if step%100==0 or step==end:
                with torch.no_grad():
                    acc={k:float((model(p)*scene.sdf(p)>0).float().mean()) for k,p in scene.probes(device).items()}
                row=dict(step=step,lr=lr,batch_mse=loss.item(),**acc)
                history.append(row)
                log.write(json.dumps(row)+'\n'); log.flush()
    sync(device)
    seconds=time.perf_counter()-began
    result=dict(config=c,step=end,model=model.state_dict(),optimizer=opt.state_dict(),sample_rng=gen.get_state(),
                schedule=schedule,origin=origin,history=history,train_seconds=seconds)
    torch.save(result,folder/'checkpoint.pt.tmp')
    (folder/'checkpoint.pt.tmp').replace(folder/'checkpoint.pt')
    atomic_json(folder/'training.json',dict(start=start,end=end,schedule=schedule,origin=origin,seconds=seconds,
        parent_sha256=digest(previous),stream_sha256=h.hexdigest() if data is None else tensor_hash(data['points'],data['target']),
        sample_rng_sha256=tensor_hash(gen.get_state()),initial_sample_rng_sha256=tensor_hash(saved['sample_rng']),
        initial_fixed=initial_fixed,final_fixed=fixed_metrics(model,data,device) if data is not None else None,history=history))
    atomic_json(folder/'training-complete.json',{n:digest(folder/n) for n in ['checkpoint.pt','training.json','training.jsonl']})


def evaluate(c,folder,m):
    check_files(folder,read(folder/'training-complete.json'))
    saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=model_for(c,torch.device(c['device']))
    model.load_state_dict(saved['model']); model.eval()
    scene=Scene(c['scene'],c['feature'],c['angle'])
    metadata=dict(checkpoint_sha256=digest(folder/'checkpoint.pt'),source_hashes=m['source_hashes'],step=saved['step'],config=c)
    result=dict(metadata=metadata,training=read(folder/'training.json'),
        field=field_metrics(model,scene,c['device'],c['eval_seed'],c['eval_points']),
        full_gap=full_gap(model,scene,c['device'],m['gap_seed'],m['gap_samples']),evaluations={})
    for resolution in m['resolutions']:
        path=folder/f'evaluation-r{resolution}.json'
        if path.exists():
            record=read(path)
            assert record['metadata']==metadata
        else:
            record=dict(metadata=metadata,metrics=geometry_metrics(model,scene,c['device'],resolution,c['surface_samples'],c['eval_seed']))
            atomic_json(path,record)
        result['evaluations'][str(resolution)]=record['metrics']
    result['primary_gate']=result['field']['all_probe_signs_correct'] and all(e['structure_match'] for e in result['evaluations'].values())
    result['full_gap_gate']=result['full_gap']['regions']['whole_gap']['all_margins']['errors']==0
    atomic_json(folder/'metrics.json',result)
    atomic_json(folder/'complete.json',{p.name:digest(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json' and not p.name.endswith('.tmp')})
    return result


def trajectory(c,previous,out,schedule,origin,m,data=None):
    records=[]
    for end in [origin+3000,origin+6000,origin+10000]:
        stage=out/f'step-{end:05d}'
        attempts=sorted(stage.glob('attempt-*'))
        completed=[a for a in attempts if (a/'training-complete.json').exists()]
        if completed:
            folder=completed[-1]
            check_files(folder,read(folder/'training-complete.json'))
            t=read(folder/'training.json')
            assert t['parent_sha256']==digest(previous) and t['end']==end and t['schedule']==schedule and t['origin']==origin
        else:
            index=max([int(a.name.split('-')[1]) for a in attempts],default=0)+1
            folder=stage/f'attempt-{index:04d}'
            train_segment(c,previous,end,folder,schedule,origin,data)
        previous=folder/'checkpoint.pt'
        if (folder/'complete.json').exists():
            check_files(folder,read(folder/'complete.json'))
            r=read(folder/'metrics.json')
        else:
            r=evaluate(c,folder,m)
        records.append(dict(folder=str(folder),**r))
        print(json.dumps(dict(policy=c['policy'],seed=c['seed'],schedule=schedule,fixed=data is not None,step=end,
            gap=r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'],primary=r['primary_gate'],
            meshes={k:v['mesh'] for k,v in r['evaluations'].items()})),flush=True)
    return records


def run(path,out):
    m=read(path)
    assert m['source_hashes']==source_hashes(), 'Source changed after freeze'
    assert m['fresh_cases_sha256']==digest(ROOT/'docs/gap-controls/fresh-cases.json')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        assert read(out/'manifest.json')==m
    else:
        atomic_json(out/'manifest.json',m)
    for name,expected in m['source_hashes'].items():
        dest=out/'source'/name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((ROOT/name).read_bytes())
        assert digest(dest)==expected
    atomic_json(out/'environment.json',dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
        gpu=torch.cuda.get_device_name(),deterministic_algorithms=True,tf32=False))
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    records=[]; audits=[]
    for pair in m['pairs']:
        c=pair['config']; checkpoint=Path(pair['checkpoint'])
        assert digest(checkpoint)==pair['sha256']
        arms={}
        for schedule in m['schedules']:
            arms[schedule]=trajectory(c,checkpoint,out/identity(c)/schedule,schedule,10000,m)
            records.extend(arms[schedule])
            atomic_json(out/'summary.json',dict(runs=records))
        for a,b in zip(arms['constant'],arms['cosine']):
            for key in ['stream_sha256','sample_rng_sha256','initial_sample_rng_sha256']:
                assert a['training'][key]==b['training'][key], f'Pair mismatch: {key}'
        audits.append(dict(policy=c['policy'],seed=c['seed'],initial_checkpoint_sha256=pair['sha256'],
            matched_segments=[dict(start=r['training']['start'],end=r['training']['end'],stream_sha256=r['training']['stream_sha256'],
                sample_rng_sha256=r['training']['sample_rng_sha256']) for r in arms['constant']]))
        atomic_json(out/'pair-audit.json',dict(pairs=audits))
    decay=[r for r in records if r['training']['schedule']=='cosine' and r['metadata']['step']>=16000]
    repaired=len(decay)==12 and all(r['primary_gate'] and r['full_gap_gate'] for r in decay)
    atomic_json(out/'decision.json',dict(consistent_repair=repaired,expected_continuations=12,completed_checkpoints=len(records),
        next_step='optimisation stability and time to sustained geometry' if repaired else 'fixed-data ordinary-MSE diagnostic',fresh_cases_evaluated=False))
    if not repaired:
        fixed_records=[]
        for pair in m['pairs']:
            c=pair['config']
            endpoint=next(r for r in records if r['metadata']['config']==c and r['training']['schedule']=='cosine' and r['metadata']['step']==20000)
            dest=out/identity(c)/'fixed-data'
            dest.mkdir(parents=True,exist_ok=True)
            data=make_fixed(c)
            data_path=dest/'data.pt'
            if data_path.exists():
                loaded=torch.load(data_path,weights_only=True)
                assert all(torch.equal(data[k],loaded[k]) for k in data)
            else:
                torch.save(data,data_path)
            atomic_json(dest/'data-audit.json',dict(sha256=digest(data_path),points_targets_sha256=tensor_hash(data['points'],data['target']),
                seed=c['seed']+91000,points=len(data['points']),objective='ordinary full-batch MSE'))
            fixed_records.extend(trajectory(c,Path(endpoint['folder'])/'checkpoint.pt',dest,'cosine',20000,m,data))
            atomic_json(out/'fixed-summary.json',dict(runs=fixed_records))
    assert m['fresh_cases_sha256']==digest(ROOT/'docs/gap-controls/fresh-cases.json')
    print('COMPLETE: continuations, pair audit, conditional diagnostic; fresh cases untouched.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','run'])
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.command=='freeze':
        freeze(args.manifest)
    else:
        run(args.manifest,args.out)
