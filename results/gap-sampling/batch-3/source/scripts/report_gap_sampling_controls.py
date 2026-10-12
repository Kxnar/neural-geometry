"""Audit all sampling arms, expose every failure, and export reproducible decisions."""
import argparse
import csv
import shutil
import statistics
from pathlib import Path
import torch

from neural_geometry_scale.study import ROOT, atomic_json, check_files, digest, identity, read
from scripts.gap_density_controls import nested_data, overlap
from scripts.gap_sampling_controls import STEPS, selections, stream_hash
from scripts.gap_lr_controls import tensor_hash


def same(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def mean(values):
    v=list(values)
    return statistics.mean(v) if v and all(x is not None for x in v) else None


def csv_write(path, rows):
    rows=list(rows)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def audit(out):
    torch.set_num_threads(4)
    m=read(out/'manifest.json'); records=read(out/'summary.json')['runs']
    done=read(out/'run-complete.json')
    assert done['summary_sha256']==digest(out/'summary.json')
    assert len(records)==done['checkpoints']==6*len(m['protocol']['arms'])*3
    assert len({(r['metadata']['config']['policy'],r['metadata']['config']['seed'],r['training']['arm']['name'],r['metadata']['step']) for r in records})==len(records)
    assert digest(ROOT/'docs/gap-controls/fresh-cases.json')==m['fresh_cases_sha256']
    for n,h in m['source_hashes'].items(): assert digest(out/'source'/n)==h
    historical=ROOT/'runs/gap-density-sweep'
    assert digest(historical/'manifest.json')==m['historical_manifest_sha256']
    assert digest(historical/'summary.json')==m['historical_summary_sha256']
    old=read(historical/'summary.json')['runs']
    exact=[]; artifacts={}; streams=[]
    for pair in m['pairs']:
        c=pair['config']; root=out/identity(c)
        assert digest(pair['checkpoint'])==pair['sha256']
        baseline=torch.load(pair['checkpoint'],map_location='cpu',weights_only=True)
        assert baseline['step']==20000 and all(int(s['step'])==20000 for s in baseline['optimizer']['state'].values())
        d=read(root/'data-audit.json');check_files(root,d['files'])
        data=torch.load(root/'pool.pt',weights_only=True); independent=torch.load(root/'independent.pt',weights_only=True)
        assert same(data,nested_data(c)) and same(independent,nested_data(c,65536,m['independent_seed']))
        assert overlap(data['points'],independent['points'])==0
        for arm in m['protocol']['arms']:
            choices=selections(arm['points'],arm['phase'])
            assert torch.equal(torch.cat(choices).sort().values,torch.arange(65536))
            assert all(torch.bincount(data['groups'][i]).tolist()==[len(i)//2,len(i)//4,len(i)//4] for i in choices)
            rows=sorted([r for r in records if r['metadata']['config']==c and r['training']['arm']==arm],key=lambda r:r['metadata']['step'])
            assert [r['metadata']['step'] for r in rows]==STEPS
            previous=pair['sha256']; start=20000
            for r in rows:
                folder=Path(r['folder']);check_files(folder,read(folder/'complete.json'))
                assert read(folder/'metrics.json')=={k:v for k,v in r.items() if k!='folder'}
                assert r['metadata']['source_hashes']==m['source_hashes']
                assert r['metadata']['checkpoint_sha256']==digest(folder/'checkpoint.pt')
                t=r['training']; assert t==read(folder/'training.json')
                assert t['parent_sha256']==previous and t['start']==start
                assert t['point_presentations']==(t['end']-start)*arm['points']
                assert t['pool_sha256']==tensor_hash(data['points'],data['target'])
                assert t['stream_sha256']==stream_hash(data,arm['points'],arm['phase'],start,t['end'])
                assert t['sample_rng_sha256']==t['initial_sample_rng_sha256']==tensor_hash(baseline['sample_rng'])
                saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
                assert saved['config']==c and saved['arm']==arm and saved['step']==t['end']
                assert all(int(s['step'])==t['end'] for s in saved['optimizer']['state'].values())
                assert torch.equal(saved['sample_rng'],baseline['sample_rng'])
                assert r['independent_points_sha256']==tensor_hash(independent['points'],independent['target'])
                assert set(r['evaluations'])=={'192','384'}
                expect=dict(components=2,euler=4,closed_edges=True)
                gate=r['field']['all_probe_signs_correct'] and all(e['mesh_present'] and e['mesh']==expect and e['reference_mesh']==expect for e in r['evaluations'].values())
                assert gate==r['primary_gate']
                if arm['points']==65536:
                    ref=next(x for x in old if x['metadata']['config']==c and x['training']['density']==65536 and x['metadata']['step']==t['end'])
                    base=torch.load(Path(ref['folder'])/'checkpoint.pt',map_location='cpu',weights_only=True)
                    assert same(saved['model'],base['model']) and same(saved['optimizer'],base['optimizer'])
                    assert r['primary_gate']==ref['primary_gate'] and r['evaluations']==ref['evaluations']
                    exact.append(dict(policy=c['policy'],seed=c['seed'],step=t['end'],model_exact=True,adam_exact=True,geometry_exact=True))
                artifacts[str(folder)]=read(folder/'complete.json')
                previous=digest(folder/'checkpoint.pt');start=t['end']
            streams.append(dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],complete_cycle_coverage=True,balanced_each_update=True))
    return dict(checkpoints_verified=len(records),parent_chains_verified=True,adam_steps_verified=True,
        frozen_source_verified=True,pools_regenerated=True,independent_overlap=0,reservation_unchanged=True,
        streams_verified=streams,dense_exact_reproduction=exact,artifacts=artifacts)


def report(out,dest):
    verification=audit(out)
    m=read(out/'manifest.json');records=read(out/'summary.json')['runs']; dest.mkdir(parents=True,exist_ok=True)
    atomic_json(dest/'audit.json',verification)
    for name in ['manifest.json','summary.json','environment.json','run-complete.json']:
        shutil.copyfile(out/name,dest/name)
    shutil.copytree(out/'source',dest/'source',dirs_exist_ok=True)
    shutil.copyfile(__file__,dest/'source/scripts/report_gap_sampling_controls.py')
    rows=[]; trajectories=[]; compute=[]; pairs=[]; bins=[]
    for pair in m['pairs']:
        c=pair['config']
        for arm in m['protocol']['arms']:
            rr=sorted([r for r in records if r['metadata']['config']==c and r['training']['arm']==arm],key=lambda r:r['metadata']['step'])
            gates=[r['primary_gate'] for r in rr]
            trajectories.append(dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],gates=gates,persistent=all(gates[-2:]),regression=any(a and not b for a,b in zip(gates,gates[1:]))))
            compute.append(dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],points_per_update=arm['points'],
                presentations=sum(r['training']['point_presentations'] for r in rr),training_seconds=sum(r['training']['seconds'] for r in rr),evaluation_seconds=sum(r['evaluation_seconds'] for r in rr)))
            for r in rr:
                row=dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],step=r['metadata']['step'],gate=r['primary_gate'],pool_fit=r['training']['final_fixed']['fit_gate'],
                    gap_signs=r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'],independent_gap_rmse=r['independent_fit']['gap']['rmse'])
                for res,e in r['evaluations'].items():
                    row.update({f'chamfer_{res}':e['symmetric_chamfer'],f'components_{res}':e['mesh']['components'],f'euler_{res}':e['mesh']['euler'],f'closed_{res}':e['mesh']['closed_edges']})
                rows.append(row)
                for split,groups in r['surface_distance_errors'].items():
                    for group,bb in groups.items():
                        for bucket,v in bb.items(): bins.append(dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],step=r['metadata']['step'],split=split,group=group,distance_bin=bucket,**v))
    # Batch 1 control remains the same-machine contemporary comparison for later batches.
    control_out=out.parent/'batch-1'
    control_records=read(control_out/'summary.json')['runs']
    controls=[r for r in control_records if r['training']['arm']['name']=='dense']
    assert len(controls)==18
    dense_time=sum(r['training']['seconds'] for r in controls)
    decisions=[]
    for arm in m['protocol']['arms']:
        if arm['name']=='dense':continue
        comparisons=[]
        for step in [26000,30000]:
            a=[r for r in records if r['training']['arm']==arm and r['metadata']['step']==step]
            b=[r for r in controls if r['metadata']['step']==step]
            d=dict(step=step,gap_rmse_ratio=mean(r['independent_fit']['gap']['rmse'] for r in a)/mean(r['independent_fit']['gap']['rmse'] for r in b),
                gap_sign_delta=mean(r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'] for r in a)-mean(r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'] for r in b))
            for res in ['192','384']:
                av=mean(r['evaluations'][res]['symmetric_chamfer'] for r in a);bv=mean(r['evaluations'][res]['symmetric_chamfer'] for r in b)
                d['chamfer_ratio_'+res]=av/bv if av is not None and bv else None
            d['guards_pass']=d['gap_rmse_ratio']<=1.10 and d['gap_sign_delta']>=-.005 and all(d['chamfer_ratio_'+res] is not None and d['chamfer_ratio_'+res]<=1.05 for res in ['192','384'])
            comparisons.append(d)
            for x in a:
                y=next(y for y in b if y['metadata']['config']==x['metadata']['config'])
                c=x['metadata']['config']
                pairs.append(dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],step=step,gate=x['primary_gate'],dense_gate=y['primary_gate'],
                    gap_rmse_delta=x['independent_fit']['gap']['rmse']-y['independent_fit']['gap']['rmse'],
                    gap_sign_delta=x['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy']-y['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'],
                    **{'chamfer_delta_'+res:x['evaluations'][res]['symmetric_chamfer']-y['evaluations'][res]['symmetric_chamfer'] if x['evaluations'][res]['symmetric_chamfer'] is not None and y['evaluations'][res]['symmetric_chamfer'] is not None else None for res in ['192','384']}))
        tt=[t for t in trajectories if t['arm']==arm['name']]
        elapsed=sum(t['training_seconds'] for t in compute if t['arm']==arm['name'])
        passes=sum(t['persistent'] for t in tt);ratio=elapsed/dense_time
        decisions.append(dict(arm=arm['name'],persistent_passes=passes,training_time_ratio=ratio,presentation_ratio=arm['points']/65536,
            comparisons=comparisons,qualifies=passes==6 and ratio<=.9 and arm['points']<65536 and all(d['guards_pass'] for d in comparisons)))
    atomic_json(dest/'decision.json',dict(batch=m['protocol']['batch'],arms=decisions,fresh_cases_evaluated=False))
    atomic_json(dest/'persistence.json',dict(trajectories=trajectories))
    csv_write(dest/'checkpoints.csv',rows);csv_write(dest/'compute.csv',compute);csv_write(dest/'paired-differences.csv',pairs);csv_write(dest/'surface-distance-errors.csv',bins)
    lines=[f"# Sampling batch {m['protocol']['batch']}",'',m['protocol']['hypothesis'],'',
        'Frozen before execution: [manifest](manifest.json). Audit: [audit](audit.json). All results use ordinary MSE and the unchanged two-resolution structural gate. Fresh cases remain reserved.','',
        '| Arm | Persistent passes | Presentations per trajectory | Training seconds (six trajectories) | Evaluation seconds |',
        '|---|---:|---:|---:|---:|']
    for arm in m['protocol']['arms']:
        cc=[c for c in compute if c['arm']==arm['name']];tt=[t for t in trajectories if t['arm']==arm['name']]
        lines.append(f"| {arm['name']} | {sum(t['persistent'] for t in tt)}/6 | {10000*arm['points']:,} | {sum(c['training_seconds'] for c in cc):.2f} | {sum(c['evaluation_seconds'] for c in cc):.2f} |")
    lines+=['','| Sampler | Seed | Arm | Gate 23k / 26k / 30k | Persistent |','|---|---:|---|---|---|']
    for t in trajectories:lines.append(f"| {t['policy']} | {t['seed']} | {t['arm']} | {' / '.join('pass' if x else 'fail' for x in t['gates'])} | {t['persistent']} |")
    lines+=['','## Frozen decision','']
    for d in decisions:
        lines.append(f"- {d['arm']}: qualifies={d['qualifies']}; {d['persistent_passes']}/6 persistent; training-time ratio {d['training_time_ratio']:.3f}; presentation ratio {d['presentation_ratio']:.4f}. Guardrail values at both later checkpoints are in [decision.json](decision.json).")
    lines+=['','All comparisons to dense use the six batch-1 controls; later batches are sequential, not simultaneous timing measurements. Training times include synchronized updates and logging; evaluation is measured separately. Full-process budget charges additionally include preparation, hashing and I/O. Pool fit is evaluated on all 65,536 pool points even when only a subset is presented each update.','',
        '## Limits and reproduction','',
        'One axis-aligned development geometry, two original samplers and three paired seeds each. The same seeds are reused adaptively across batches: this is development evidence. Cycling changes gradient noise and order; it retains all dense labels and storage. Persistent means observed at 26k and 30k, not a guarantee between checkpoints or in continuous space. Small independent sign errors and strict pool-fit failures remain visible in the complete metrics.','',
        'From the original neural-geometry checkout, use a fresh output directory and the frozen source snapshot. Parent checkpoints and saved pools are indexed by SHA-256. The budget-run command enforces the campaign limit; run is available for a separately authorised reproduction.','',
        '```powershell',"$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'",f".venv/Scripts/python.exe -m scripts.gap_sampling_controls run --manifest docs/gap-sampling/batch-{m['protocol']['batch']}.json --out runs/reproduce-sampling-batch-{m['protocol']['batch']}",f".venv/Scripts/python.exe -m scripts.report_gap_sampling_controls --out runs/gap-sampling/batch-{m['protocol']['batch']} --dest docs/gap-sampling/batch-{m['protocol']['batch']}",'```','']
    (dest/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(dest/'export-audit.json',dict(files={p.relative_to(dest).as_posix():digest(p) for p in dest.rglob('*') if p.is_file() and p.name!='export-audit.json'}))
    print(read(dest/'decision.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();report(a.out,a.dest)
