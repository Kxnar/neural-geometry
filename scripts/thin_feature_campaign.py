"""Budgeted analytic thin-feature reconstruction campaign.

Never import this module to expose reserved geometry. Only a sealed worker may
generate reserved data. Every trajectory and failed attempt is externally charged.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from neural_geometry.models import Field
from neural_geometry.metrics import mesh_from_volume, mesh_surface_samples, symmetric_chamfer
from neural_geometry_scale.geometry import Scene, batch
from neural_geometry_scale.evaluation import predict, topology, field_metrics
from neural_geometry_scale.study import ROOT, digest, read, sources, sync
from scripts.gap_lr_controls import tensor_hash, lr_at
from scripts.gap_density_controls import full_gradient, overlap
from scripts.gap_reserved_confirmation import pool as legacy_pool

DEST = ROOT/'results/thin-feature-campaign'
RUNS = ROOT/'runs/thin-feature-campaign'
RESERVATION = ROOT/'inputs/gap-confirmation/fresh-cases-reserved.json'
OLD_CONSUMPTION = ROOT/'results/gap-confirmation/reservation-consumption.json'
CONFIGS = [dict(model='siren', width=128, depth=3),
           dict(model='fourier', width=128, depth=3, frequencies=64, scale=10.)]
ARMS = ['dense', 'random', 'cycling']
ENDS = [23000, 26000, 30000]
LIMITS = dict(development=3600., confirmation=14400., qa=1800., global_seconds=19800.)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    else:
        temp = path.with_suffix(path.suffix+'.tmp')
        with temp.open('w', encoding='utf-8') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)


def scientific_hashes():
    paths = ['scripts/thin_feature_campaign.py', 'scripts/report_thin_feature_campaign.py',
             'tests/test_thin_feature_campaign.py', 'scripts/gap_lr_controls.py',
             'scripts/gap_density_controls.py', 'scripts/gap_reserved_confirmation.py',
             'scripts/gap_confirmation.py', 'scripts/gap_training_controls.py']
    return {**sources(), **{p:digest(ROOT/p) for p in paths}}


def environment():
    return dict(python=platform.python_version(), executable=sys.executable,
                executable_sha256=digest(sys.executable), torch=str(torch.__version__),
                numpy=np.__version__, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                deterministic=True, tf32=False, cublas=':4096:8')


def deterministic():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False


def factory(config, seed, device):
    torch.manual_seed(seed)
    kwargs=dict(config); kwargs['kind']=kwargs.pop('model')
    return Field(**kwargs).to(device)


def state_hash(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str): h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (list,tuple)):
            for x in v: visit(x)
        else: h.update(repr(v).encode())
    visit(value); return h.hexdigest()


class FamilyScene(Scene):
    """Box union preserving legacy plate behavior and exact reserved stack boxes."""
    def __init__(self, case):
        object.__setattr__(self, 'case', dict(case))
        object.__setattr__(self, 'name', case['scene'])
        object.__setattr__(self, 'feature', case.get('feature', min(case.get('full_gaps',[.02]))))
        object.__setattr__(self, 'angle', case['angle'])
        if self.name != 'multiscale_plate_stack': Scene.__post_init__(self)

    def boxes(self, device='cpu', dtype=torch.float32):
        if self.name != 'multiscale_plate_stack': return super().boxes(device,dtype)
        return (torch.tensor(self.case['centers'],device=device,dtype=dtype),
                torch.tensor(self.case['half_extents'],device=device,dtype=dtype))

    @property
    def expected(self):
        n=3 if self.name=='multiscale_plate_stack' else 2
        return dict(components=n,euler=2*n,closed_edges=True)

    @property
    def scale_bounds(self):
        if self.name!='multiscale_plate_stack': return super().scale_bounds
        return min(self.case['full_gaps'])/2, .12

    def gaps(self):
        centers,half=self.boxes(); axis=0 if self.name=='mixed_thickness' else 2
        result=[]
        for i in range(len(centers)-1):
            low=torch.maximum(centers[i]-half[i],centers[i+1]-half[i+1])
            high=torch.minimum(centers[i]+half[i],centers[i+1]+half[i+1])
            low[axis]=centers[i,axis]+half[i,axis]
            high[axis]=centers[i+1,axis]-half[i+1,axis]
            if not bool((high>low).all()): raise ValueError('Nonpositive gap/overlap')
            result.append((low,high,axis,i,i+1))
        return result

    def probes(self, device='cpu'):
        if self.name!='multiscale_plate_stack': return super().probes(device)
        centers,half=self.boxes(device); v=torch.linspace(-.8,.8,16,device=device)
        x,y=torch.meshgrid(v,v,indexing='ij'); p=torch.stack((x.flatten(),y.flatten(),x.flatten()*0),-1)
        result={f'component_{i}':(p*half[i]+centers[i])@self.rotation(p).T for i in range(3)}
        for i,(lo,hi,axis,_,_) in enumerate(self.gaps()):
            local=p*((hi-lo)/2).to(device)+(hi+lo).to(device)/2
            result[f'gap_{i}']=local@self.rotation(local).T
        return result


def sample_regions(scene,n,g):
    """Uniform complete gaps; balanced interiors including all facing surfaces."""
    gap_parts=[]; gap_ids=[]; interiors=[]; component_ids=[]
    gaps=scene.gaps(); centers,half=scene.boxes()
    for i,(low,high,axis,left,right) in enumerate(gaps):
        count=n//len(gaps)+(i<n%len(gaps))
        gap_parts.append(low+(high-low)*torch.rand(count,3,generator=g)); gap_ids.extend([i]*count)
    for i in range(len(centers)):
        count=n//len(centers)+(i<n%len(centers))
        p=(torch.rand(count,3,generator=g)*2-1)*half[i]+centers[i]
        faces=[(axis,1 if i==left else -1) for _,_,axis,left,right in gaps if i in [left,right]]
        for j,(axis,sign) in enumerate(faces):
            mask=torch.arange(count)%len(faces)==j
            depth=min(.01,float(half[i,axis]))
            p[mask,axis]=centers[i,axis]+sign*(half[i,axis]-depth*torch.rand(int(mask.sum()),generator=g))
        interiors.append(p); component_ids.extend([i]*count)
    return torch.cat(gap_parts),torch.tensor(gap_ids),torch.cat(interiors),torch.tensor(component_ids)


def pool(case,seed,independent=False,size=65536):
    scene=FamilyScene(case); actual_seed=91002 if independent else seed+91000
    if scene.name=='parallel_plates':
        c=dict(case,seed=seed,policy='global_narrow')
        data=legacy_pool(c,actual_seed)
        local=data['points']@scene.rotation(data['points'])
        data['gap_ids']=torch.where(data['groups']==1,0,-1)
        data['component_ids']=torch.where(data['groups']==2,(local[:,2]>0).long(),-1)
        return data
    g=torch.Generator().manual_seed(actual_seed); blocks=[]; gids=[]; cids=[]
    for _ in range(size//4096):
        original=batch(scene,2048,'global_narrow',g)[0]
        gap,gi,inside,ci=sample_regions(scene,1024,g)
        blocks.append(torch.cat([original,gap@scene.rotation(gap).T,inside@scene.rotation(inside).T]))
        gids.append(torch.cat([torch.full((2048,),-1),gi,torch.full((1024,),-1)]))
        cids.append(torch.cat([torch.full((3072,),-1),ci]))
    points=torch.cat(blocks)
    return dict(points=points,target=scene.sdf(points),groups=torch.tensor([0]*2048+[1]*1024+[2]*1024).repeat(size//4096),
                gap_ids=torch.cat(gids),component_ids=torch.cat(cids))


@torch.no_grad()
def region_metrics(model,data,device):
    truth=data['target'].flatten().double(); pred=predict(model,data['points'].to(device)).cpu().flatten().double()
    masks={'all':torch.ones(len(truth),dtype=torch.bool)}
    for i,name in enumerate(['original','gap','interior']): masks[name]=data['groups']==i
    for key,label in [('gap_ids','gap'),('component_ids','component')]:
        for i in sorted(set(data[key].tolist())-{-1}): masks[f'{label}_{i}']=data[key]==i
    result={}
    for name,mask in masks.items():
        if not bool(mask.any()): continue
        valid=mask&(truth.abs()>1e-7); error=pred[mask]-truth[mask]
        result[name]=dict(samples=int(mask.sum()),rmse=float(error.square().mean().sqrt()),
                          sign_accuracy=float((pred[valid]*truth[valid]>0).double().mean()),
                          min_signed_prediction=float((pred[valid]*truth[valid].sign()).min()),
                          min_true_margin=float(truth[valid].abs().min()))
    return result


@torch.no_grad()
def geometry(model,scene,device,resolution,count=20000,seed=91000):
    axis=torch.linspace(-1,1,resolution,device=device); x,y=torch.meshgrid(axis,axis,indexing='ij')
    points=torch.stack([x.flatten(),y.flatten(),torch.zeros_like(x).flatten()],-1)
    pv=[];tv=[]
    for z in axis:
        points[:,2]=z; pv.append(predict(model,points).reshape(resolution,resolution).cpu().numpy())
        tv.append(scene.sdf(points).reshape(resolution,resolution).cpu().numpy())
    volume,truth=np.stack(pv,2),np.stack(tv,2)
    vertices,faces=mesh_from_volume(volume); rv,rf=mesh_from_volume(truth)
    target=scene.surface(count,torch.Generator().manual_seed(seed))[0].numpy()
    sampled=mesh_surface_samples(vertices,faces,count,np.random.default_rng(seed+2)) if len(faces) else np.empty((0,3))
    pt,rt=topology(vertices,faces),topology(rv,rf)
    chamfer=symmetric_chamfer(sampled,target)
    return dict(resolution=resolution,mesh_present=bool(len(faces)),mesh=pt,reference_mesh=rt,
                reference_resolved=rt==scene.expected,structure_match=pt==scene.expected and rt==scene.expected,
                symmetric_chamfer=float(chamfer) if chamfer is not None and math.isfinite(chamfer) else None)


def persistent(rows):
    by={r['step']:r for r in rows}
    passing=lambda step: by.get(step,{}).get('primary_gate') is True and valid_row(by[step])
    return dict(persistent=passing(26000) and passing(30000),
                onset=23000 if all(passing(x) for x in ENDS) else (26000 if passing(26000) and passing(30000) else None))


def valid_row(row):
    try:
        required=[row['independent']['gap']['rmse'],row['full_gap']['gap']['sign_accuracy']]
        for res in ['192','384']:
            e=row['evaluations'][res]
            if not e['reference_resolved'] or not e['mesh_present']: return False
            required.append(e['symmetric_chamfer'])
        for name,value in row['field'].items():
            if isinstance(value,dict): required.extend([value['sign_accuracy'],value['mae'],value['error_margin_ratio_max']])
        return all(isinstance(v,(float,int)) and math.isfinite(v) for v in required)
    except (KeyError,TypeError): return False


def guard(candidate,dense):
    try:
        comparisons=[]
        for step in [26000,30000]:
            c=next(r for r in candidate if r['step']==step); d=next(r for r in dense if r['step']==step)
            cr=c['independent']['gap']['rmse']; dr=d['independent']['gap']['rmse']
            delta=c['full_gap']['gap']['sign_accuracy']-d['full_gap']['gap']['sign_accuracy']
            values=[cr,dr,delta]
            passed=cr<=1.10*dr and delta>=-.005
            for resolution in ['192','384']:
                cv=c['evaluations'][resolution]['symmetric_chamfer']; dv=d['evaluations'][resolution]['symmetric_chamfer']
                values.extend([cv,dv]); passed=passed and cv<=1.05*dv
            comparisons.append(passed and all(v is not None and math.isfinite(v) for v in values))
        return all(comparisons)
    except (KeyError,StopIteration,TypeError): return False


def aggregate_guards(pairs):
    """Frozen historical error guards compare means across complete paired sets."""
    if not pairs: return dict(passes=False,reason='No complete paired set',endpoints=[])
    endpoints=[]
    for step in [26000,30000]:
        try:
            candidates=[next(x for x in c if x['step']==step) for c,d in pairs]
            dense=[next(x for x in d if x['step']==step) for c,d in pairs]
            if not all(valid_row(r) for r in candidates+dense): raise ValueError('Invalid paired endpoint')
            mean=lambda rows,fun:sum(fun(r) for r in rows)/len(rows)
            cr=mean(candidates,lambda r:r['independent']['gap']['rmse']);dr=mean(dense,lambda r:r['independent']['gap']['rmse'])
            delta=mean(candidates,lambda r:r['full_gap']['gap']['sign_accuracy'])-mean(dense,lambda r:r['full_gap']['gap']['sign_accuracy'])
            row=dict(step=step,gap_rmse_ratio=cr/dr if dr else None,sign_delta=delta,passes=cr<=1.10*dr and delta>=-.005)
            for res in ['192','384']:
                cv=mean(candidates,lambda r:r['evaluations'][res]['symmetric_chamfer']);dv=mean(dense,lambda r:r['evaluations'][res]['symmetric_chamfer'])
                row['chamfer_ratio_'+res]=cv/dv if dv else None;row['passes']&=cv<=1.05*dv
            endpoints.append(row)
        except (KeyError,StopIteration,TypeError,ValueError):
            endpoints.append(dict(step=step,passes=False,reason='Missing/invalid/nonfinite paired endpoint'))
    return dict(passes=all(r['passes'] for r in endpoints),endpoints=endpoints,pairs=len(pairs))


def check_manifest(path):
    hash_path=Path(str(path)+'.sha256')
    if not hash_path.exists() or digest(path)!=hash_path.read_text().strip(): raise RuntimeError('Immutable seal drift')
    m=read(path)
    if m['sources']!=scientific_hashes(): raise RuntimeError('Frozen scientific source drift')
    if m['environment']!=environment(): raise RuntimeError('Frozen environment drift')
    if digest(RESERVATION)!=m['reservation_sha256'] or digest(OLD_CONSUMPTION)!=m['old_consumption_sha256']:
        raise RuntimeError('Original reservation or consumption drift')
    if digest(DEST/'protocol.json')!=m['protocol_sha256']: raise RuntimeError('Protocol drift')
    if m['stage']=='confirmation':
        cases=read(RESERVATION)['cases']
        expected={(i,s,a) for i in range(2,8) for s in [71,72,73] for a in ARMS}
        actual={(t['case_index'],t['seed'],t['arm']) for t in m['tasks']}
        if actual!=expected or len(m['tasks'])!=54 or any(t['case']!=cases[t['case_index']] or t['config']!=m['selected_config'] for t in m['tasks']): raise RuntimeError('Reserved inventory drift')
    return m


def exposure(m,task,action):
    if m['stage']!='confirmation': return
    path=DEST/'reservation-events.jsonl'
    row=dict(utc=utc(),seal_sha256=digest(DEST/'confirmation-manifest.json'),task=task['id'],
             original_case_index=task['case_index'],seed=task['seed'],arm=task['arm'],action=action)
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())


def save_checkpoint(path,model,opt,g,step,config):
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),sample_rng=g.get_state(),step=step,config=config)
    torch.save(payload,path.with_suffix('.tmp')); os.replace(path.with_suffix('.tmp'),path)
    return dict(sha256=digest(path),model=state_hash(payload['model']),optimizer=state_hash(payload['optimizer']),sample_rng=tensor_hash(payload['sample_rng']))


def worker(manifest,task_id):
    deterministic(); m=check_manifest(manifest); task=next(t for t in m['tasks'] if t['id']==task_id)
    out=RUNS/m['stage']/task_id; out.mkdir(parents=True,exist_ok=False)
    for name,sha in m['sources'].items():
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
        if digest(dest)!=sha: raise RuntimeError('Source snapshot drift')
    start=float(os.environ.get('THIN_FEATURE_LAUNCH_CLOCK',time.perf_counter())); device=torch.device('cuda'); case=task['case']; scene=None
    exposure(m,task,'generate_geometry');scene=FamilyScene(case)
    exposure(m,task,'generate_training_pool');data=pool(case,task['seed'])
    exposure(m,task,'generate_independent_pool');independent=pool(case,task['seed'],True)
    if overlap(data['points'],independent['points']): raise RuntimeError('Independent overlap')
    exposure(m,task,'generate_full_gap');fullg=torch.Generator().manual_seed(91001); gap,gids,inside,cids=sample_regions(scene,131072,fullg)
    wp=gap@scene.rotation(gap).T
    fullgap=dict(points=wp,target=scene.sdf(wp),groups=torch.ones(len(wp),dtype=torch.long),
                 gap_ids=gids,component_ids=torch.full((len(wp),),-1))
    for name,value in [('pool',data),('independent',independent),('full-gap',fullgap)]: torch.save(value,out/(name+'.pt'))
    write(out/'data.json',dict(hashes={n:digest(out/(n+'.pt')) for n in ['pool','independent','full-gap']},overlap=0,
                              preparation_seconds=time.perf_counter()-start,oracle_regions=True))
    exposure(m,task,'initialize_model');model=factory(task['config'],task['seed'],device); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    g=torch.Generator(device=device).manual_seed(task['seed']+1000)
    initial=save_checkpoint(out/'initial.pt',model,opt,g,0,task['config'])
    write(out/'model.json',dict(parameters=sum(p.numel() for p in model.parameters()),initial=initial,
                              basis_sha256=tensor_hash(model.basis) if hasattr(model,'basis') else None,config=task['config']))
    points,target=data['points'].to(device),data['target'].to(device)
    random_plan=None
    if task['arm']=='random':
        exposure(m,task,'generate_random_schedule')
        plan_path=out/'random-indices.npy'; plan=np.lib.format.open_memmap(plan_path,mode='w+',dtype=np.int32,shape=(10000,16384))
        rg=torch.Generator().manual_seed(800000+task['case_index']*100+task['seed'])
        for j in range(10000): plan[j]=torch.randperm(65536,generator=rg)[:16384].numpy()
        plan.flush(); del plan; random_plan=np.load(plan_path,mmap_mode='r')
        write(out/'random-schedule.json',dict(sha256=digest(plan_path),seed=800000+task['case_index']*100+task['seed'],unstratified=True))
    records=[]; training_seconds=0.
    # Schedule hashes are computed without device readback in the training loop.
    stream_hash=state_hash(dict(pool=tensor_hash(data['points'],data['target']),arm=task['arm'],
                                updates=10000,cycling_origin=20000,random_sha256=digest(out/'random-indices.npy') if random_plan is not None else None))
    sync(device); segment=time.perf_counter()
    exposure(m,task,'train_prefix')
    with (out/'training.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,30001):
            for group in opt.param_groups: group['lr']=1e-4 if step<=10000 else lr_at((step-1)%10000+1)
            if step<=20000:
                p,t,_=batch(scene,2048,'global_narrow',g,device)
                opt.zero_grad(set_to_none=True); loss=(model(p)-t).square().mean(); loss.backward()
            else:
                j=step-20001
                if j==0: exposure(m,task,'train_recovery')
                if task['arm']=='random': idx=torch.from_numpy(np.array(random_plan[j],copy=True)).to(device=device,dtype=torch.long)
                elif task['arm']=='cycling': idx=torch.arange((j%4)*16384,(j%4+1)*16384,device=device)
                else: idx=torch.arange(65536,device=device)
                opt.zero_grad(set_to_none=True); loss=full_gradient(model,points[idx],target[idx],16384)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            opt.step()
            if step%100==0:
                log.write(json.dumps(dict(step=step,mse=loss.item(),elapsed=time.perf_counter()-start))+'\n'); log.flush()
                write(out/'progress.json',dict(step=step,elapsed_seconds=time.perf_counter()-start,arm=task['arm']))
            if step==20000:
                parent=save_checkpoint(out/'prefix-20000.pt',model,opt,g,step,task['config'])
                write(out/'prefix.json',dict(**parent,initial=initial))
            if step in m['checkpoints']:
                sync(device); training_seconds+=time.perf_counter()-segment
                cp=save_checkpoint(out/f'checkpoint-{step}.pt',model,opt,g,step,task['config'])
                exposure(m,task,f'evaluate_checkpoint_{step}')
                began=time.perf_counter(); field=field_metrics(model,scene,device,91000,65536)
                fit=region_metrics(model,independent,device); fg=region_metrics(model,fullgap,device)
                evaluations={str(r):geometry(model,scene,device,r) for r in m['resolutions']}
                sync(device)
                primary=field['all_probe_signs_correct'] and all(e['mesh_present'] and e['structure_match'] and e['reference_resolved'] and e['symmetric_chamfer'] is not None for e in evaluations.values())
                reasons=[]
                if not field['all_probe_signs_correct']: reasons.append('original_probe_sign_failure')
                for res,e in evaluations.items():
                    if not e['reference_resolved']: reasons.append(f'grid{res}:unresolved_reference')
                    if not e['mesh_present']: reasons.append(f'grid{res}:missing_mesh')
                    if e['mesh']!=scene.expected: reasons.append(f'grid{res}:topology_mismatch:{e["mesh"]}')
                    if e['symmetric_chamfer'] is None: reasons.append(f'grid{res}:nonfinite_surface_error')
                row=dict(step=step,primary_gate=bool(primary),field=field,independent=fit,full_gap=fg,evaluations=evaluations,
                         checkpoint=cp,evaluation_seconds=time.perf_counter()-began,training_seconds=training_seconds,
                         cumulative_pipeline_seconds=time.perf_counter()-start,failure_reasons=reasons)
                if not valid_row(row): row['primary_gate']=False;row['failure_reasons'].append('invalid_or_nonfinite_metrics')
                records.append(row); write(out/f'metrics-{step}.json',row); write(out/'summary.json',dict(task=task,records=records))
                print(json.dumps(dict(task=task_id,step=step,primary=primary,elapsed=row['cumulative_pipeline_seconds'])),flush=True)
                segment=time.perf_counter()
    check_manifest(manifest)
    write(out/'complete.json',dict(task=task,records=records,persistence=persistent(records),worker_seconds=time.perf_counter()-start,
                                 selection_stream_sha256=stream_hash,files={p.name:digest(p) for p in out.iterdir() if p.is_file()}))


def kill_tree(process):
    if process.poll() is not None: return
    if os.name=='nt': subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    else: os.killpg(process.pid,9)
    process.wait()


def allowance(ledger,stage):
    if any(not a.get('finished') for a in ledger['attempts']): raise RuntimeError('Unclosed external attempt')
    total=sum(a['charged_seconds'] for a in ledger['attempts'])
    stage_total=sum(a['charged_seconds'] for a in ledger['attempts'] if a['stage']==stage)
    return min(LIMITS[stage]-stage_total,LIMITS['global_seconds']-total)


def supervise(command,stage,task=None,manifest=None):
    ledger_path=DEST/'budget.json'; ledger=read(ledger_path); remaining=allowance(ledger,stage)
    if remaining<=5: return False
    if manifest: check_manifest(manifest)
    number=len(ledger['attempts'])+1; log=DEST/f'attempt-{number:04d}.log'
    row=dict(number=number,stage=stage,task=task,command=command,started_utc=utc(),finished=False,
             allowance_seconds=remaining,log=log.relative_to(ROOT).as_posix())
    ledger['attempts'].append(row); write(ledger_path,ledger)
    start=time.perf_counter(); process=None; code=-1
    try:
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',THIN_FEATURE_LAUNCH_CLOCK=str(start))
        with log.open('x',encoding='utf-8') as f:
            process=subprocess.Popen([sys.executable]+command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,
                                     start_new_session=os.name!='nt')
            row['pid']=process.pid; write(ledger_path,ledger)
            print(json.dumps(dict(pid=process.pid,stage=stage,task=task,log=str(log))),flush=True)
            try: code=process.wait(timeout=max(1,remaining-2))
            except subprocess.TimeoutExpired: row['timed_out']=True; kill_tree(process); code=process.returncode
    finally:
        if process and process.poll() is None: kill_tree(process)
        row.update(finished=True,exit_code=code,finished_utc=utc(),charged_seconds=time.perf_counter()-start)
        write(ledger_path,ledger)
    if manifest: check_manifest(manifest)
    return code==0


def task(case,index,seed,arm,config,stage):
    key=f'{config["model"]}-case{index}-seed{seed}-{arm}'
    return dict(id=key,case=case,case_index=index,seed=seed,arm=arm,config=config,stage=stage)


def init():
    DEST.mkdir(parents=True,exist_ok=True); RUNS.mkdir(parents=True,exist_ok=True)
    write(DEST/'budget.json',dict(limits=LIMITS,attempts=[]),True)
    protocol=dict(
                  configs=CONFIGS,development='Exactly four crossed parallel-plate cells .012/.03 by17/37 seed71; dense+cycling. Selected narrow17 seeds72/73 only if development budget remains. No optional random.',
                  confirmation='Original reserved indices2..7; seeds71/72/73; independent initialization-to30k dense/random/cycling trajectories; fixed rotated order.',
                  objective='Unweighted exact ordinary MSE; Adam; stochastic prefix10k constant1e-4 then10k cosine1e-4to1e-6; recovery10k same cosine.',
                  pool='65536 balanced50/25/25; cycling16384; original global_narrow; oracle analytic regions and labels.',
                  criteria='Persistent26k+30k both192/384 all original probes closed mesh expected components/Euler resolved reference; missing/nonfinite fail closed.',
                  ranking='fewest lost dense successes; most cycling narrow persistent; most cycling all persistent; fewest guard violations; lowest attributed pipeline time; SIREN tie first. Incomplete comparison fallback SIREN inconclusive.',
                  guards='Historical guards compare paired endpoint means across all cases: independent gap RMSE<=1.10dense; fullgap sign delta>=-.005; each grid Chamfer<=1.05dense. Narrow-stack pairs also individually satisfy every guard.',
                  target='Complete valid confirmation, preserve every dense pass; narrow stacks all6 persistent with guards; aggregate actual cycling pipeline<=.75dense including failed attempts.',
                  time='External launch-through-durable-exit charged incl startup/data/labels/randomschedule/prefix/train/eval/transfers/checkpoint/hash/failures. Development prefixes are independently repeated; configurations interleaved case/arm. Retrospective onset suffix length>=2 ending30k; onset pipeline separate from later verification.',
                  limits=LIMITS,scope='Descriptive oracle analytic evidence for three box-union families; not arbitrary unseen shapes or topology certification.',
                  rng=dict(model='seed',prefix='CUDA seed+1000',training_pool='CPU seed+91000',independent='CPU91002',full_gap='CPU91001',evaluation='CPU91000',random='CPU800000+original_case_index*100+seed'))
    write(DEST/'protocol.json',protocol,True)


def seal(stage,tasks,config=None):
    manifest=dict(schema=1,stage=stage,tasks=tasks,selected_config=config,sources=scientific_hashes(),
                  environment=environment(),created_utc=utc(),reservation_sha256=digest(RESERVATION),
                  old_consumption_sha256=digest(OLD_CONSUMPTION),protocol_sha256=digest(DEST/'protocol.json'),
                  checkpoints=[26000,30000] if stage=='development' else ENDS,resolutions=[192,384],limits=LIMITS)
    if stage=='confirmation':
        manifest.update(selection_sha256=digest(DEST/'selection.json'),development_manifest_sha256=digest(DEST/'development-manifest.json'),
                        development_results_sha256=state_hash(load_completed('development')))
    write(DEST/f'{stage}-manifest.json',manifest,True)
    path=DEST/f'{stage}-manifest.json'
    with Path(str(path)+'.sha256').open('x',encoding='utf-8') as f: f.write(digest(path)+'\n')


def freeze_development():
    cases=[dict(scene='parallel_plates',feature=w,angle=a) for w in [.012,.03] for a in [17,37]]
    tasks=[task(c,i,71,arm,config,'development') for i,c in enumerate(cases) for arm in ['dense','cycling'] for config in CONFIGS]
    seal('development',tasks)


def load_completed(stage):
    rows=[]
    for path in sorted((RUNS/stage).glob('*/complete.json')):
        value=read(path)
        for name,sha in value['files'].items():
            if digest(path.parent/name)!=sha: raise RuntimeError('Completed artifact drift')
        rows.append(value)
    return rows


def run_stage(stage):
    path=DEST/f'{stage}-manifest.json'; m=check_manifest(path)
    for t in m['tasks']:
        folder=RUNS/stage/t['id']
        if folder.exists(): continue
        if allowance(read(DEST/'budget.json'),stage)<=5: break
        supervise(['-m','scripts.thin_feature_campaign','worker','--manifest',str(path),'--task',t['id']],stage,t['id'],path)
    check_manifest(path)
    write(DEST/f'{stage}-status.json',dict(expected=len(m['tasks']),completed=len(load_completed(stage)),
                                         budget_remaining_seconds=allowance(read(DEST/'budget.json'),stage)))


def select():
    m=check_manifest(DEST/'development-manifest.json'); rows=load_completed('development')
    complete=len(rows)==16 and {r['task']['id'] for r in rows}=={t['id'] for t in m['tasks']} and all(len(r['records'])==2 and all(valid_row(x) for x in r['records']) for r in rows); ranks=[]
    for config in CONFIGS:
        model=config['model']; own=[r for r in rows if r['task']['config']==config]; lost=0;narrow=0;allpass=0;violations=0
        for i in range(4):
            dense=next((r for r in own if r['task']['case_index']==i and r['task']['arm']=='dense'),None)
            cycle=next((r for r in own if r['task']['case_index']==i and r['task']['arm']=='cycling'),None)
            if not dense or not cycle: continue
            dp=dense['persistence']['persistent']; cp=cycle['persistence']['persistent']
            lost+=int(dp and not cp); allpass+=int(cp); narrow+=int(cp and cycle['task']['case']['feature']==.012)
            violations+=int(not guard(cycle['records'],dense['records']))
        agg_pairs=[]
        for i in range(4):
            d=next((r for r in own if r['task']['case_index']==i and r['task']['arm']=='dense'),None)
            cc=next((r for r in own if r['task']['case_index']==i and r['task']['arm']=='cycling'),None)
            if d and cc: agg_pairs.append((cc['records'],d['records']))
        aggregate=aggregate_guards(agg_pairs)
        violations=sum(not e['passes'] for e in aggregate['endpoints']) if len(agg_pairs)==4 else 2
        seconds=sum(r['worker_seconds'] for r in own if r['task']['arm']=='cycling')
        ranks.append(dict(config=config,rank=[lost,-narrow,-allpass,violations,seconds,CONFIGS.index(config)],
                          lost_dense=lost,narrow_persistent=narrow,all_persistent=allpass,guard_violations=violations,aggregate_guards=aggregate))
    chosen=min(ranks,key=lambda r:r['rank'])['config'] if complete else CONFIGS[0]
    write(DEST/'selection.json',dict(selected_config=chosen,comparison_complete=complete,
                                    selection_inconclusive=not complete,ranks=ranks,no_switch_after_validation=True),True)


def freeze_confirmation():
    selected=read(DEST/'selection.json')['selected_config']; reservation=read(RESERVATION)
    tasks=[]
    for i in range(2,8):
        for si,seed in enumerate([71,72,73]):
            rotate=(i+si)%3; order=ARMS[rotate:]+ARMS[:rotate]
            for arm in order: tasks.append(task(reservation['cases'][i],i,seed,arm,selected,'confirmation'))
    seal('confirmation',tasks,selected)


def validate_development():
    path=DEST/'development-manifest.json'; m=check_manifest(path)
    config=read(DEST/'selection.json')['selected_config']; case=dict(scene='parallel_plates',feature=.012,angle=17)
    validation=[task(case,0,seed,arm,config,'development') for seed in [72,73] for arm in ['dense','cycling']]
    # Separate immutable validation manifest; development seal never rewritten.
    vp=DEST/'validation-manifest.json'
    if not vp.exists():
        write(vp,dict(m,tasks=validation),True)
        with Path(str(vp)+'.sha256').open('x',encoding='utf-8') as f:f.write(digest(vp)+'\n')
    for t in validation:
        if (RUNS/'development'/t['id']).exists(): continue
        if allowance(read(DEST/'budget.json'),'development')<=5: break
        supervise(['-m','scripts.thin_feature_campaign','worker','--manifest',str(vp),'--task',t['id']],'development',t['id'],vp)


def verify():
    for stage in ['development','confirmation']:
        if (DEST/f'{stage}-manifest.json').exists(): check_manifest(DEST/f'{stage}-manifest.json')
    ledger=read(DEST/'budget.json')
    for stage in ['development','confirmation','qa']: allowance(ledger,stage)
    confirmation=load_completed('confirmation'); pairs={}
    for row in confirmation:
        t=row['task']; prefix=read(RUNS/'confirmation'/t['id']/'prefix.json')
        folder=RUNS/'confirmation'/t['id']
        for name,sha in read(DEST/'confirmation-manifest.json')['sources'].items():
            if digest(folder/'source'/name)!=sha: raise RuntimeError('Source snapshot drift')
        if {r['step'] for r in row['records']}!=set(ENDS): raise RuntimeError('Missing completed checkpoint')
        pairs.setdefault((t['case_index'],t['seed']),[]).append(prefix)
    matched=all(len(v)==3 and all(all(x[k]==v[0][k] for k in ['model','optimizer','sample_rng']) for x in v) for v in pairs.values())
    events=[json.loads(line) for line in (DEST/'reservation-events.jsonl').read_text().splitlines()] if (DEST/'reservation-events.jsonl').exists() else []
    for r in confirmation:
        own=[e['action'] for e in events if e['task']==r['task']['id']]
        expected_events=['generate_geometry','generate_training_pool','generate_independent_pool','generate_full_gap','initialize_model']
        if r['task']['arm']=='random': expected_events.append('generate_random_schedule')
        expected_events+=['train_prefix','train_recovery']+[f'evaluate_checkpoint_{s}' for s in ENDS]
        if own!=expected_events: raise RuntimeError('Exposure event audit mismatch')
    partial=[p.parent.name for p in (RUNS/'confirmation').glob('*/progress.json') if not (p.parent/'complete.json').exists()]
    audit=dict(confirmation_completed=len(confirmation),expected=54,complete=len(confirmation)==54,
               prefix_pairing=matched and len(pairs)==18,failed_attempts=sum(a['exit_code']!=0 for a in ledger['attempts']),
               stage_charges={s:sum(a['charged_seconds'] for a in ledger['attempts'] if a['stage']==s) for s in ['development','confirmation','qa']},
               events=len(events),completed_source_snapshots_verified=True,partial_trajectories=partial,
               original_reservation_unchanged=True,old_consumption_unchanged=True)
    write(DEST/'audit.json',audit); print(json.dumps(audit),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['init','test','freeze-development','run-development','select','validate-development','freeze-confirmation','run-confirmation','worker','report','verify'])
    p.add_argument('--manifest',type=Path);p.add_argument('--task');a=p.parse_args()
    if a.command=='init': init()
    elif a.command=='test':
        if not supervise(['-m','pytest','tests/test_gap_sampling_controls.py','tests/test_gap_confirmation.py','tests/test_gap_reserved_confirmation.py','tests/test_thin_feature_campaign.py','-q','-p','no:cacheprovider','--basetemp',str(RUNS/f'test-temp-{len(read(DEST/"budget.json")["attempts"])+1}')],'qa'): raise SystemExit(1)
    elif a.command=='freeze-development': freeze_development()
    elif a.command=='run-development': run_stage('development')
    elif a.command=='select': select()
    elif a.command=='validate-development': validate_development()
    elif a.command=='freeze-confirmation': freeze_confirmation()
    elif a.command=='run-confirmation': run_stage('confirmation')
    elif a.command=='worker': worker(a.manifest,a.task)
    elif a.command=='report':
        from scripts.report_thin_feature_campaign import report
        report()
    else: verify()


if __name__=='__main__': main()
