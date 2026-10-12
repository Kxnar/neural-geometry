"""Two wholly preregistered rotated-plate confirmation batches; no retuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from neural_geometry_scale.geometry import Scene,batch
from neural_geometry_scale.evaluation import predict,field_metrics,geometry_metrics
from neural_geometry_scale.study import ROOT,read,digest,identity,model_for,sync,check_files
from scripts.gap_lr_controls import lr_at,tensor_hash,fixed_metrics,train_segment
from scripts.gap_training_controls import train_segment as initial_train
from scripts.gap_density_controls import full_gradient,nested_data,overlap
from scripts.gap_confirmation import DEST,INPUTS,utc,allowance,code_hashes,same
from scripts.recover_gap_confirmation_io import resilient_json as atomic_json,install_retry

ARMS=[dict(name='dense',points=65536),dict(name='canonical',points=16384),dict(name='static',points=16384),dict(name='random',points=16384)]
ENDS=[26000,30000]


def sources():
    return {**code_hashes(),**{p:digest(ROOT/p) for p in ['scripts/gap_reserved_confirmation.py','scripts/recover_gap_confirmation_io.py','tests/test_gap_reserved_confirmation.py']}}


def pool(c,seed=None):
    scene=Scene(c['scene'],c['feature'],c['angle']); assert scene.name=='parallel_plates'
    g=torch.Generator().manual_seed(c['seed']+91000 if seed is None else seed); blocks=[]
    half=c['feature']/2
    for _ in range(16):
        original=batch(scene,2048,c['policy'],g)[0]
        gap=(torch.rand(1024,3,generator=g)*2-1)*torch.tensor([.6,.6,half])
        inside=(torch.rand(1024,3,generator=g)*2-1)*torch.tensor([.6,.6,1.])
        inside[:,2]=inside[:,2].sign()*(half+.01*torch.rand(1024,generator=g))
        blocks.append(torch.cat([original,gap@scene.rotation(gap).T,inside@scene.rotation(inside).T]))
    p=torch.cat(blocks)
    groups=torch.cat([torch.zeros(2048,dtype=torch.long),torch.ones(1024,dtype=torch.long),torch.full((1024,),2,dtype=torch.long)]).repeat(16)
    return dict(points=p,target=scene.sdf(p),groups=groups)


@torch.no_grad()
def full_gap(model,scene,device,seed,count):
    g=torch.Generator().manual_seed(seed)
    local=(torch.rand(count,3,generator=g)*2-1)*torch.tensor([.6,.6,scene.feature/2])
    points=local@scene.rotation(local).T; truth=scene.sdf(points).flatten(); valid=truth>1e-7
    points,truth,local=points[valid],truth[valid],local[valid]
    pred=predict(model,points.to(device)).cpu().flatten(); outer=(local[:,:2].abs()>.48).any(-1)
    regions={}
    edges=[0,.001,.0025,.005,.011]
    if scene.feature/2>.011: edges.append(scene.feature/2+1e-7)
    for name,region in [('whole_gap',torch.ones(len(points),dtype=torch.bool)),('central_footprint',~outer),('outer_footprint',outer)]:
        masks={'all_margins':region}
        for lo,hi in zip(edges,edges[1:]): masks[f'margin_{lo:g}_{hi:g}']=region&(truth>lo)&(truth<=hi)
        regions[name]={}
        for label,mask in masks.items():
            n=int(mask.sum()); regions[name][label]=dict(samples=n,errors=int((pred[mask]<=0).sum()),
                sign_accuracy=float((pred[mask]>0).float().mean()) if n else None,
                sdf_mae=float((pred[mask]-truth[mask]).abs().mean()) if n else None)
    return dict(seed=seed,excluded=int((~valid).sum()),points_sha256=tensor_hash(points,truth),regions=regions,
        coordinate_frame='Gap and footprint sampled in local frame, rotated to world; bins cover complete gap')


@torch.no_grad()
def stratified(model,data,c):
    pred=predict(model,data['points'].to(c['device'])).cpu().flatten().double(); target=data['target'].flatten().double()
    local=data['points']@Scene(c['scene'],c['feature'],c['angle']).rotation(data['points'])
    outer=(local[:,:2].abs()>.48).any(-1); error=pred-target; dist=target.abs(); result={}
    masks={'all':torch.ones(len(target),dtype=torch.bool)}
    for i,name in enumerate(['original','gap','interior']):
        masks[name]=data['groups']==i
        if i: masks[name+'_outer']=masks[name]&outer; masks[name+'_central']=masks[name]&~outer
    for name,mask in masks.items():
        bins={'all':mask}
        for lo,hi in [(0,.001),(.001,.0025),(.0025,.005),(.005,.01),(.01,.05),(.05,float('inf'))]: bins[f'{lo:g}_{hi:g}']=mask&(dist>lo)&(dist<=hi)
        result[name]={}
        for label,sel in bins.items():
            n=int(sel.sum()); valid=sel&(dist>1e-7)
            result[name][label]=dict(samples=n,mse=float(error[sel].square().mean()) if n else None,
                rmse=float(error[sel].square().mean().sqrt()) if n else None,sign_errors=int((pred[valid]*target[valid]<=0).sum()),sign_samples=int(valid.sum()))
    return result


def freeze():
    assert read(DEST/'decision.json')['gate_pass']
    reservation=read(INPUTS/'fresh-cases-reserved.json'); selected=reservation['cases'][:2]
    hashes=sources(); config=read(DEST/'batch-1-manifest.json')['pairs'][0]['config']
    m=dict(schema=1,created_utc=utc(),source_hashes=hashes,case_allowlist=selected,seeds=[71,72,73],policy='global_narrow',arms=ARMS,
        reservation_sha256=digest(INPUTS/'fresh-cases-reserved.json'),order_gate_manifest_sha256=digest(DEST/'batch-1-manifest.json'),
        gate_decision_sha256=digest(DEST/'decision.json'),checkpoints=ENDS,resolutions=[192,384],gap_seed=91001,gap_samples=131072,
        independent_seed=91002,independent_points=65536,
        config_template=dict(config,eval_seed=91000,resolutions=[192,384]),
        prefix='Per geometry/seed ONCE:10k stochastic ordinaryMSE constant1e-4/batch2048, then10k identical cosine continuation retainingAdam/RNG. Fork20k parent.',
        recovery='10k additional updates samecosine1e-4->1e-6. Dense65536, canonical16384 cycle, staticbalancedprefix16384, uniformrandom16384 withoutreplacement eachupdate CPUgen800000+caseindex*100+seed; precompute int32 schedule.',
        objective='Ordinary equal-point MSE all arms; identical pool, labels, parent model/Adam/samplingRNG, independent data withinpair. No evaluation-based selection.',
        geometry='Localgap halfwidthfeature/2 and facinginterior depth.01; rotate localpointstoworld. Footprint split localabs(xy)>.48. Fullmarginbinscovercompletegap.',
        criterion='Canonical persistent ALL6 pairs; no densepass/candidatefail; atboth26k/30k meanindependentgapRMSE<=1.10xdense, meanChamferatEACHgrid<=1.05xdense, fullgapsigndelta>=-.005; aggregaterecoverytraining<=.90xdense. Missing/nonfinite/unresolved/incomplete cannot pass.',
        simple_controls='Staticbalanced and genuinelyunstratified random. If randommatches, no balancingnecessity/superiorityclaim.',
        scope='Two unseen angled parallelplate geometries,3pairedseeds,ONEfamily. Six othercompletecases untouched. Persistentonly26/30 and192/384 finitegrids.',
        stopping='Complete both frozen batches budgetpermitting regardlessbatch2outcome. Retainallfailures; NOretuning/replacements/resolutionchange/seedsubset.',
        budget='Scientific batches1,2,3 ONLY; totalchargedwholeexperiment+modeltestsubprocesswall<=7200seconds. LinkedI/Orecovery is same logicalbatch1.')
    with (DEST/'confirmation-manifest.json').open('x',encoding='utf-8') as f: json.dump(m,f,indent=2,allow_nan=False); f.write('\n')
    atomic_json(DEST/'reservation-consumption.json',dict(sha256=m['reservation_sha256'],selected_cases=selected,untouched_cases=reservation['cases'][2:],events=[]))
    print('Confirmation freeze:',digest(DEST/'confirmation-manifest.json'),flush=True)


def event(m,case_index,seed,action):
    ledger=read(DEST/'reservation-consumption.json'); assert m['case_allowlist'][case_index]==ledger['selected_cases'][case_index]
    ledger['events'].append(dict(utc=utc(),case_index=case_index,case=m['case_allowlist'][case_index],seed=seed,action=action))
    atomic_json(DEST/'reservation-consumption.json',ledger)


def random_schedule(path,seed,updates=10000):
    began=time.perf_counter(); g=torch.Generator().manual_seed(seed)
    plan=np.lib.format.open_memmap(path,mode='w+',dtype=np.int32,shape=(updates,16384))
    counts=torch.zeros(3,dtype=torch.long); low=torch.full((3,),16384,dtype=torch.long); high=torch.zeros(3,dtype=torch.long)
    groups=torch.cat([torch.zeros(2048,dtype=torch.long),torch.ones(1024,dtype=torch.long),torch.full((1024,),2,dtype=torch.long)]).repeat(16)
    for j in range(updates):
        idx=torch.randperm(65536,generator=g)[:16384]; plan[j]=idx.numpy().astype(np.int32)
        c=torch.bincount(groups[idx],minlength=3); counts+=c; low=torch.minimum(low,c); high=torch.maximum(high,c)
    plan.flush(); del plan
    return dict(seed=seed,shape=[updates,16384],dtype='int32',sha256=digest(path),preparation_seconds=time.perf_counter()-began,
        group_counts=counts.tolist(),minimum_group_counts=low.tolist(),maximum_group_counts=high.tolist())


def selection_stream(data,arm,start,end,plan=None):
    h=hashlib.sha256(tensor_hash(data['points'],data['target']).encode())
    for step in range(start,end):
        j=step-20000
        if arm['name']=='random': idx=np.asarray(plan[j],dtype=np.int32)
        else:
            offset=(j%4)*16384 if arm['name']=='canonical' else 0
            idx=np.arange(offset,offset+arm['points'],dtype=np.int32)
        h.update(idx.tobytes())
    return h.hexdigest()


def train(c,parent,end,folder,data,arm,plan):
    folder.mkdir(parents=True,exist_ok=False); device=torch.device(c['device'])
    saved=torch.load(parent,map_location=device,weights_only=True); assert saved['config']==c and 20000<=saved['step']<end<=30000
    model=model_for(c,device); model.load_state_dict(saved['model']); opt=torch.optim.Adam(model.parameters(),lr=1e-4); opt.load_state_dict(saved['optimizer'])
    initial=fixed_metrics(model,data,device); points=data['points'].to(device); target=data['target'].to(device)
    indexes=torch.from_numpy(np.array(plan[saved['step']-20000:end-20000],copy=True)).to(device=device,dtype=torch.long) if arm['name']=='random' else None
    start=saved['step']; history=[]; sync(device); began=time.perf_counter()
    with (folder/'training.jsonl').open('x',encoding='utf-8') as log:
        for step in range(start+1,end+1):
            for group in opt.param_groups: group['lr']=lr_at(step-20000)
            if indexes is not None: p,t=points[indexes[step-start-1]],target[indexes[step-start-1]]
            else:
                offset=((step-20001)%4)*16384 if arm['name']=='canonical' else 0
                p,t=points[offset:offset+arm['points']],target[offset:offset+arm['points']]
            opt.zero_grad(set_to_none=True); loss=full_gradient(model,p,t,16384)
            if not torch.isfinite(loss): raise RuntimeError('Nonfiniteloss;retainfailure')
            opt.step()
            if step%100==0:
                row=dict(step=step,batch_mse=loss.item(),lr=lr_at(step-20000),elapsed_seconds=time.perf_counter()-began)
                history.append(row); log.write(json.dumps(row)+'\n'); log.flush()
                atomic_json(folder/'progress.json',dict(completed_update=step,completed_presentations=(step-start)*arm['points']))
    sync(device); seconds=time.perf_counter()-began
    torch.save(dict(config=c,step=end,model=model.state_dict(),optimizer=opt.state_dict(),sample_rng=saved['sample_rng'],arm=arm,schedule='cosine',origin=20000,history=history),folder/'checkpoint.pt.tmp')
    (folder/'checkpoint.pt.tmp').replace(folder/'checkpoint.pt')
    atomic_json(folder/'training.json',dict(start=start,end=end,seconds=seconds,arm=arm,point_presentations=(end-start)*arm['points'],
        parent_sha256=digest(parent),pool_sha256=tensor_hash(data['points'],data['target']),stream_sha256=selection_stream(data,arm,start,end,plan),
        initial_sample_rng_sha256=tensor_hash(saved['sample_rng']),sample_rng_sha256=tensor_hash(saved['sample_rng']),
        initial_fixed=initial,final_fixed=fixed_metrics(model,data,device),history=history))
    atomic_json(folder/'training-complete.json',{n:digest(folder/n) for n in ['checkpoint.pt','training.json','training.jsonl']})


def measure(c,folder,m,data,independent):
    began=time.perf_counter(); check_files(folder,read(folder/'training-complete.json'))
    saved=torch.load(folder/'checkpoint.pt',map_location=c['device'],weights_only=True); model=model_for(c,torch.device(c['device'])); model.load_state_dict(saved['model']); model.eval()
    scene=Scene(c['scene'],c['feature'],c['angle']); metadata=dict(config=c,step=saved['step'],checkpoint_sha256=digest(folder/'checkpoint.pt'),source_hashes=m['source_hashes'])
    r=dict(metadata=metadata,training=read(folder/'training.json'),field=field_metrics(model,scene,c['device'],c['eval_seed'],c['eval_points']),
        full_gap=full_gap(model,scene,c['device'],m['gap_seed'],m['gap_samples']),independent_fit=fixed_metrics(model,independent,c['device']),
        independent_points_sha256=tensor_hash(independent['points'],independent['target']),evaluations={},
        surface_distance_errors=dict(training=stratified(model,data,c),independent=stratified(model,independent,c)))
    for resolution in m['resolutions']:
        metrics=geometry_metrics(model,scene,c['device'],resolution,c['surface_samples'],c['eval_seed'])
        atomic_json(folder/f'evaluation-r{resolution}.json',dict(metadata=metadata,metrics=metrics)); r['evaluations'][str(resolution)]=metrics
    r['primary_gate']=r['field']['all_probe_signs_correct'] and all(e['structure_match'] and e['mesh_present'] for e in r['evaluations'].values())
    sync(torch.device(c['device'])); r['evaluation_seconds']=time.perf_counter()-began
    return r


def run(case_index):
    install_retry()
    import scripts.gap_training_controls as original
    original.atomic_json=atomic_json
    m=read(DEST/'confirmation-manifest.json'); assert m['source_hashes']==sources() and case_index in [0,1]
    assert digest(INPUTS/'fresh-cases-reserved.json')==m['reservation_sha256']
    batch_id=case_index+2; out=ROOT/f'runs/gap-confirmation/batch-{batch_id}'; out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'manifest.json',m)
    for n,sha in m['source_hashes'].items():
        p=out/'source'/n; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((ROOT/n).read_bytes()); assert digest(p)==sha
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    atomic_json(out/'environment.json',dict(python=platform.python_version(),executable=sys.executable,python_executable_sha256=digest(sys.executable),
        torch=str(torch.__version__),numpy=np.__version__,gpu=torch.cuda.get_device_name(),tf32=False,deterministic=True,cublas=os.getenv('CUBLAS_WORKSPACE_CONFIG')))
    records=[]; prefixes=[]
    for seed_index,seed in enumerate(m['seeds']):
        c=dict(m['config_template'],**m['case_allowlist'][case_index],seed=seed,policy='global_narrow'); root=out/identity(c); root.mkdir()
        event(m,case_index,seed,'generate_training_pool'); began=time.perf_counter(); data=pool(c)
        event(m,case_index,seed,'generate_independent_pool'); independent=pool(c,m['independent_seed']); assert overlap(data['points'],independent['points'])==0
        for n,d in [('pool.pt',data),('independent.pt',independent)]: torch.save(d,root/n)
        prep=dict(seconds=time.perf_counter()-began,pool_sha256=tensor_hash(data['points'],data['target']),independent_sha256=tensor_hash(independent['points'],independent['target']),overlap=0)
        atomic_json(root/'data-audit.json',prep)
        event(m,case_index,seed,'train_shared_prefix'); began=time.perf_counter()
        initial_train(c,None,10000,root/'prefix-10000')
        train_segment(c,root/'prefix-10000/checkpoint.pt',20000,root/'prefix-20000','cosine',10000)
        parent=root/'prefix-20000/checkpoint.pt'; saved=torch.load(parent,map_location='cpu',weights_only=True)
        prefix=dict(config=c,parent_sha256=digest(parent),sampling_rng_sha256=tensor_hash(saved['sample_rng']),point_presentations=40960000,
            training_seconds=read(root/'prefix-10000/training.json')['train_seconds']+read(root/'prefix-20000/training.json')['seconds'],stage_seconds=time.perf_counter()-began,
            data_preparation_seconds=prep['seconds'])
        prefixes.append(prefix); atomic_json(out/'prefixes.json',dict(pairs=prefixes))
        event(m,case_index,seed,'generate_random_selection_schedule')
        random_path=root/'random-indices.npy'; random_meta=random_schedule(random_path,800000+case_index*100+seed); atomic_json(root/'random-schedule.json',random_meta)
        plan=np.load(random_path,mmap_mode='r')
        rotated=ARMS[(seed_index+case_index)%4:]+ARMS[:(seed_index+case_index)%4]
        for arm in rotated:
            previous=parent
            for end in ENDS:
                folder=root/arm['name']/f'step-{end}'/'attempt-0001'; began=time.perf_counter()
                train(c,previous,end,folder,data,arm,plan)
                event(m,case_index,seed,f'evaluate_{arm["name"]}_{end}')
                r=measure(c,folder,m,data,independent); r['stage_seconds']=time.perf_counter()-began
                atomic_json(folder/'metrics.json',r); atomic_json(folder/'complete.json',{p.name:digest(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json' and not p.name.endswith('.tmp')})
                records.append(dict(folder=folder.relative_to(ROOT).as_posix(),**r)); atomic_json(out/'summary.json',dict(runs=records)); previous=folder/'checkpoint.pt'
                print(json.dumps(dict(batch=batch_id,seed=seed,arm=arm['name'],step=end,primary=r['primary_gate'],
                    references={res:e['reference_resolved'] for res,e in r['evaluations'].items()},train_seconds=r['training']['seconds'],eval_seconds=r['evaluation_seconds'],completed=len(records),expected=24)),flush=True)
        del plan
    assert len(records)==24 and m['source_hashes']==sources() and digest(INPUTS/'fresh-cases-reserved.json')==m['reservation_sha256']
    atomic_json(out/'run-complete.json',dict(checkpoints=24,summary_sha256=digest(out/'summary.json'),case_index=case_index,untouched_cases=6))


def charged(command,batch=None):
    ledger=DEST/'budget.json'; state=read(ledger); remaining=allowance(state)
    ids={a['batch'] for a in state['attempts'] if a.get('batch') is not None}
    assert remaining>15 and (batch is None or batch not in ids and len(ids)<3)
    row=dict(batch=batch,started_utc=utc(),finished=False,command=command,source_sha256=digest(__file__))
    if batch: row['manifest_sha256']=digest(DEST/'confirmation-manifest.json')
    number=len(state['attempts'])+1; log=DEST/(f'batch-{batch}-stdout.log' if batch else f'confirmation-tests-{number}.log'); row['log']=log.relative_to(ROOT).as_posix()
    state['attempts'].append(row); atomic_json(ledger,state); began=time.perf_counter(); code=-1
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    try:
        with log.open('x',encoding='utf-8') as f: code=subprocess.run([sys.executable]+command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining-5).returncode
    except subprocess.TimeoutExpired: row['timed_out']=True
    finally:
        row.update(finished=True,finished_utc=utc(),charged_seconds=time.perf_counter()-began,exit_code=code); atomic_json(ledger,state)
    print(json.dumps(row),flush=True)
    if code: raise SystemExit(code)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['freeze','run','budget-run','test']); p.add_argument('--case-index',type=int); a=p.parse_args()
    if a.command=='freeze': freeze()
    elif a.command=='run': run(a.case_index)
    elif a.command=='budget-run': charged(['-m','scripts.gap_reserved_confirmation','run','--case-index',str(a.case_index)],batch=a.case_index+2)
    else:
        number=len(read(DEST/'budget.json')['attempts'])+1
        charged(['-m','pytest','tests/test_gap_reserved_confirmation.py','-q','--basetemp',str(ROOT/f'runs/gap-confirmation/confirmation-test-temp-{number}')])
