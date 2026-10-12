"""Audit and export the completed LR experiment, including negative results."""
import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from neural_geometry_scale.study import ROOT, read, digest, check_files, atomic_json
from scripts.gap_lr_controls import tensor_hash


def mean_valid(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def number(value):
    return 'missing' if value is None else f'{value:.6f}'


def difference(a,b):
    return None if a is None or b is None else a-b


def audit(out):
    m=read(out/'manifest.json')
    source_root=(out/'source').resolve()
    for name,expected in m['source_hashes'].items():
        path=(source_root/name).resolve()
        assert path.is_relative_to(source_root) and digest(path)==expected
    assert digest(ROOT/'docs/gap-controls/fresh-cases.json')==m['fresh_cases_sha256']
    rows=read(out/'summary.json')['runs']
    fixed=read(out/'fixed-summary.json')['runs'] if (out/'fixed-summary.json').exists() else []
    assert len(rows)==36
    if not read(out/'decision.json')['consistent_repair']:
        assert len(fixed)==18
    evidence={}
    for r in rows+fixed:
        folder=Path(r['folder'])
        hashes=read(folder/'complete.json')
        check_files(folder,hashes)
        assert r['metadata']['checkpoint_sha256']==digest(folder/'checkpoint.pt')
        assert r['metadata']['source_hashes']==m['source_hashes']
        saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert saved['config']==r['metadata']['config'] and saved['step']==r['metadata']['step']
        assert all(int(s['step'])==saved['step'] for s in saved['optimizer']['state'].values())
        assert saved['schedule']==r['training']['schedule']
        evidence[folder.relative_to(out).as_posix()]=hashes
        if r['training']['origin']==20000:
            dataset=folder.parent.parent
            d=read(dataset/'data-audit.json')
            assert digest(dataset/'data.pt')==d['sha256']
            saved_data=torch.load(dataset/'data.pt',map_location='cpu',weights_only=True)
            assert tensor_hash(saved_data['points'],saved_data['target'])==d['points_targets_sha256']==r['training']['stream_sha256']
            evidence[dataset.relative_to(out).as_posix()]={n:digest(dataset/n) for n in ['data.pt','data-audit.json']}
    assert len({r['full_gap']['points_sha256'] for r in rows+fixed})==1
    for pair in m['pairs']:
        c=pair['config']
        assert digest(pair['checkpoint'])==pair['sha256']
        arms={s:sorted([r for r in rows if r['metadata']['config']==c and r['training']['schedule']==s],key=lambda r:r['metadata']['step']) for s in m['schedules']}
        for s,arm in arms.items():
            assert [r['metadata']['step'] for r in arm]==[13000,16000,20000]
            previous=pair['sha256']
            for r in arm:
                assert r['training']['parent_sha256']==previous
                previous=r['metadata']['checkpoint_sha256']
        for a,b in zip(arms['constant'],arms['cosine']):
            for key in ['stream_sha256','sample_rng_sha256','initial_sample_rng_sha256']:
                assert a['training'][key]==b['training'][key]
        diagnostic=sorted([r for r in fixed if r['metadata']['config']==c],key=lambda r:r['metadata']['step'])
        if diagnostic:
            assert [r['metadata']['step'] for r in diagnostic]==[23000,26000,30000]
            previous=arms['cosine'][-1]['metadata']['checkpoint_sha256']
            for r in diagnostic:
                assert r['training']['parent_sha256']==previous
                assert r['training']['initial_sample_rng_sha256']==r['training']['sample_rng_sha256']
                previous=r['metadata']['checkpoint_sha256']
            assert len({r['training']['stream_sha256'] for r in diagnostic})==1
    return m,rows,fixed,dict(completed_continuations=12,completed_checkpoints=len(rows),fixed_checkpoints=len(fixed),
        all_pairs_exact_stream_match=True,adam_step_counts_verified=True,checkpoint_parent_chains_verified=True,
        source_snapshots_verified=True,fresh_reservation_unchanged=True,artifacts=evidence)


def csv_rows(rows):
    for r in rows:
        c=r['metadata']['config']; t=r['training']
        row=dict(policy=c['policy'],seed=c['seed'],schedule=t['schedule'],origin=t['origin'],step=r['metadata']['step'],
            primary_gate=r['primary_gate'],full_gap_gate=r['full_gap_gate'],original_gap_accuracy=r['field']['gap']['sign_accuracy'])
        for region,groups in r['full_gap']['regions'].items():
            for margin,v in groups.items():
                for k in ['samples','errors','sign_accuracy','sdf_mae']:
                    row[f'{region}_{margin}_{k}']=v[k]
        for res,e in r['evaluations'].items():
            for k in ['symmetric_chamfer','reference_sampling_floor','structure_match','reference_resolved','mesh_present']:
                row[f'r{res}_{k}']=e[k]
            for k,v in e['mesh'].items():
                row[f'r{res}_{k}']=v
        if t['final_fixed']:
            row['training_fit_gate']=t['final_fixed']['fit_gate']
            for group in ['original','gap','interior']:
                for k,v in t['final_fixed'][group].items():
                    row[f'training_{group}_{k}']=v
        yield row


def write_csv(path,rows):
    rows=list(rows)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def charts(rows,fixed,dest):
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
    for ax,(policy,seed) in zip(axes.flat,[(p,s) for p in ['global_narrow','local'] for s in [31,32,33]]):
        for schedule,color in [('constant','#d45b36'),('cosine','#2875ba')]:
            group=sorted([r for r in rows if r['metadata']['config']['policy']==policy and r['metadata']['config']['seed']==seed and r['training']['schedule']==schedule],key=lambda r:r['metadata']['step'])
            hist=[h for r in group for h in r['training']['history']]
            ax.plot([h['step'] for h in hist],[h['gap']*100 for h in hist],color=color,alpha=.35,lw=1)
            ax.plot([r['metadata']['step'] for r in group],[r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy']*100 for r in group],'-o',color=color,label=schedule)
        ax.set_title(f'{policy}, seed {seed}');ax.grid(alpha=.2);ax.set_ylim(50,101)
        ax.set_xlabel('Total updates');ax.set_ylabel('Correct gap signs (%)')
    axes[0,0].legend()
    fig.suptitle('Faint: original mid-plane probes every 100 updates. Bold: full-footprint gap volume.\nPerfect probe signs alone do not imply correct mesh structure.')
    fig.tight_layout();fig.savefig(dest/'gap-stability.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    buckets=['margin_0_0.001','margin_0.001_0.0025','margin_0.0025_0.005','margin_0.005_0.011']
    for ax,policy in zip(axes,['global_narrow','local']):
        for schedule,color in [('constant','#d45b36'),('cosine','#2875ba')]:
            group=[r for r in rows if r['metadata']['config']['policy']==policy and r['metadata']['step']==20000 and r['training']['schedule']==schedule]
            ys=[[100*(1-r['full_gap']['regions']['whole_gap'][b]['sign_accuracy']) for b in buckets] for r in group]
            for y in ys: ax.plot(range(4),y,color=color,alpha=.25)
            ax.plot(range(4),[statistics.mean(y[i] for y in ys) for i in range(4)],'-o',color=color,label=schedule)
        ax.set_xticks(range(4),['0–.001','.001–.0025','.0025–.005','.005–.01']);ax.set_xlabel('Distance from true surface');ax.set_title(policy);ax.grid(alpha=.2)
    axes[0].set_ylabel('Wrong gap signs (%)');axes[0].legend()
    fig.suptitle('20,000 updates: full footprint, three seeds (faint) and mean (bold)')
    fig.tight_layout();fig.savefig(dest/'margin-errors.png',dpi=160);plt.close(fig)


def report(out,dest):
    m,rows,fixed,verification=audit(out)
    dest.mkdir(parents=True,exist_ok=True)
    for name in ['manifest.json','environment.json','summary.json','pair-audit.json','decision.json','fixed-summary.json']:
        if (out/name).exists(): shutil.copyfile(out/name,dest/name)
    shutil.copytree(out/'source',dest/'source',dirs_exist_ok=True)
    shutil.copyfile(__file__,dest/'source/scripts/report_gap_lr_controls.py')
    shutil.copyfile(ROOT/'tests/test_gap_lr_controls.py',dest/'source/test_gap_lr_controls.py')
    atomic_json(dest/'audit.json',verification)
    write_csv(dest/'checkpoints.csv',csv_rows(rows))
    if fixed: write_csv(dest/'fixed-checkpoints.csv',csv_rows(fixed))
    charts(rows,fixed,dest)
    lines=['# Does learning-rate decay stabilise gap geometry?','',
        'Development only: ordinary-MSE SIREN width 128, parallel plates with gap 0.02, both samplers and seeds 31–33. All 12 matched continuations and their failures are retained. Fresh cases remain untouched.','',
        'Each pair loads the same 10,000-update model, Adam moments/step counts and sampling RNG. Each receives 10,000 further updates on byte-identical coordinates and labels. Constant Adam uses 1e-4; cosine uses 1e-4 on update 1 and 1e-6 on update 10,000. No optimiser restart at evaluation checkpoints.','',
        '## Checkpoint results','',
        '| Sampler | Schedule | Updates | Full-gap signs | Near-surface signs (≤.001) | Outer-footprint signs | Distance 192 / 384 | Gate at both grids |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for policy in ['global_narrow','local']:
        for schedule in ['constant','cosine']:
            for step in [13000,16000,20000]:
                g=[r for r in rows if r['metadata']['config']['policy']==policy and r['training']['schedule']==schedule and r['metadata']['step']==step]
                avg=lambda region,margin: statistics.mean(r['full_gap']['regions'][region][margin]['sign_accuracy'] for r in g)
                cd=[mean_valid(r['evaluations'][res]['symmetric_chamfer'] for r in g) for res in ['192','384']]
                lines.append(f"| {policy} | {schedule} | {step} | {avg('whole_gap','all_margins'):.2%} | {avg('whole_gap','margin_0_0.001'):.2%} | {avg('outer_footprint','all_margins'):.2%} | {number(cd[0])} / {number(cd[1])} | {sum(r['primary_gate'] for r in g)}/3 |")
    lines+=['','## Persistence by trajectory','',
        'The gate retains all original probe signs and requires two components, Euler characteristic 4, closed edges and a resolved analytic reference at both grids. Full-gap signs are a separate, stricter sampled diagnostic. A sustained observed pass requires the 16,000 and 20,000 checkpoints; observations do not certify what happened between them.','',
        '| Sampler | Seed | Schedule | Gate 13k / 16k / 20k | Full-gap pass 13k / 16k / 20k | Sustained observed gate |',
        '|---|---:|---|---|---|---|']
    persistence=[]
    for pair in m['pairs']:
        c=pair['config']
        for schedule in m['schedules']:
            g=sorted([r for r in rows if r['metadata']['config']==c and r['training']['schedule']==schedule],key=lambda r:r['metadata']['step'])
            gates=[r['primary_gate'] for r in g]; gaps=[r['full_gap_gate'] for r in g]
            sustained=all(gates[-2:])
            first_suffix=13000 if all(gates) else 16000 if sustained else None
            persistence.append(dict(policy=c['policy'],seed=c['seed'],schedule=schedule,primary=gates,full_gap=gaps,sustained=sustained,first_observed_sustained_suffix=first_suffix))
            fmt=lambda a:' / '.join('pass' if x else 'fail' for x in a)
            lines.append(f"| {c['policy']} | {c['seed']} | {schedule} | {fmt(gates)} | {fmt(gaps)} | {'yes' if sustained else 'no'} |")
    atomic_json(dest/'persistence.json',dict(trajectories=persistence))
    paired=[]
    for pair in m['pairs']:
        c=pair['config']
        for step in m['checkpoints']:
            a=next(r for r in rows if r['metadata']['config']==c and r['metadata']['step']==step and r['training']['schedule']=='constant')
            b=next(r for r in rows if r['metadata']['config']==c and r['metadata']['step']==step and r['training']['schedule']=='cosine')
            paired.append(dict(policy=c['policy'],seed=c['seed'],step=step,
                gap_accuracy_delta=b['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy']-a['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'],
                chamfer192_delta=difference(b['evaluations']['192']['symmetric_chamfer'],a['evaluations']['192']['symmetric_chamfer']),
                chamfer384_delta=difference(b['evaluations']['384']['symmetric_chamfer'],a['evaluations']['384']['symmetric_chamfer'])))
    write_csv(dest/'paired-differences.csv',paired)
    repaired=read(out/'decision.json')['consistent_repair']
    lines+=['','## Decision','',
        'Decay consistently repaired the sampled gap and structural gate at both later checkpoints. Pursue optimisation stability and denser time-to-sustained-geometry measurements.' if repaired else
        'Decay did not consistently repair the gap under the frozen criterion (all six trajectories passing both the structural gate and sampled full-gap signs at both later checkpoints). The fixed-data diagnostic below was therefore run. No new loss is advanced.']
    final_pairs=[p for p in paired if p['step']==20000]
    lines+=['',f"At 20,000 updates, decay improves full-gap sign accuracy in {sum(p['gap_accuracy_delta']>0 for p in final_pairs)}/6 pairs, with a mean paired change of {100*statistics.mean(p['gap_accuracy_delta'] for p in final_pairs):+.2f} percentage points. Mean paired sampled-distance changes are {number(mean_valid(p['chamfer192_delta'] for p in final_pairs))} at 192 and {number(mean_valid(p['chamfer384_delta'] for p in final_pairs))} at 384 (negative is better). Across all 36 continuation checkpoints, {sum(r['primary_gate'] for r in rows)} satisfy the structural gate; {sum(p['sustained'] for p in persistence)}/12 trajectories show sustained observed structural success."]
    lines+=['',f"Missing meshes: {sum(not e['mesh_present'] for r in rows+fixed for e in r['evaluations'].values())} across {2*len(rows+fixed)} checkpoint-resolution evaluations. Missing distances are excluded from distance means and paired distance differences; missing meshes always fail the structural gate."]
    if fixed:
        lines+=['','## Fixed-data overfitting diagnostic','',
            'All six decayed 20,000-update checkpoints were continued with their Adam state on a saved 4,096-point dataset: 2,048 points from the original sampler, 1,024 uniform full-gap points, and 1,024 points inside the plates within 0.01 of the facing surfaces. Every update uses the entire fixed dataset and ordinary equal-point MSE, with the same 10,000-update cosine schedule. This intentionally changes coverage and removes stochastic sampling; it does not isolate either change alone.','',
            'Training-fit gate: every subset RMSE ≤0.001 and no wrong training signs where |true SDF|>0.001. Independent geometry and full-footprint checks reuse the frozen evaluation points, never the training dataset.','',
            '| Sampler | Seed | Original RMSE | Gap RMSE | Interior RMSE | Strong-margin training errors | Training fit gate | Independent gap signs | Geometry gate |',
            '|---|---:|---:|---:|---:|---:|---|---:|---|']
        for r in fixed:
            if r['metadata']['step']!=30000: continue
            c=r['metadata']['config']; f=r['training']['final_fixed']
            errors=sum(f[k]['margin_gt_001_errors'] for k in ['original','gap','interior'])
            lines.append(f"| {c['policy']} | {c['seed']} | {f['original']['rmse']:.6g} | {f['gap']['rmse']:.6g} | {f['interior']['rmse']:.6g} | {errors} | {f['fit_gate']} | {r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy']:.2%} | {r['primary_gate']} |")
        last=[r for r in fixed if r['metadata']['step']==30000]
        lines+=['',f"At the final diagnostic checkpoint, {sum(r['training']['final_fixed']['fit_gate'] for r in last)}/6 satisfy the fixed-training-set fit criterion and {sum(r['primary_gate'] for r in last)}/6 satisfy the independent structural gate. Training fit and geometry are distinct outcomes. Failure under this finite budget and Adam schedule does not prove representational impossibility; passing finite training points does not certify the continuous field."]
        initial=[r for r in fixed if r['metadata']['step']==23000]
        before=statistics.mean(r['training']['initial_fixed']['gap']['rmse'] for r in initial)
        after=statistics.mean(r['training']['final_fixed']['gap']['rmse'] for r in last)
        lines+=['',f"Mean fixed gap-training RMSE changes from {before:.6g} before the diagnostic to {after:.6g} after 10,000 full-batch updates. Full per-subset errors, sign margins and independent geometry at all three diagnostic checkpoints are in [fixed-data measurements](fixed-checkpoints.csv)."]
    lines+=['','## Evidence and limits','',
        '- [All per-checkpoint measurements](checkpoints.csv), [paired cosine-minus-constant differences](paired-differences.csv), [persistence](persistence.json), [pair stream audit](pair-audit.json), [artifact audit](audit.json).',
        '- [Gap stability curves](gap-stability.png), [distance-stratified errors](margin-errors.png), [frozen manifest](manifest.json), [exact source snapshots](source/scripts/gap_lr_controls.py).',
        '- Full-gap probes: 131,072 requested uniform points across the entire 1.2×1.2 footprint and 0.02 gap. Exclude only numerical true margins ≤1e-7. Every region is also split into 0–.001, .001–.0025, .0025–.005 and .005–.01 true-distance buckets. The outer footprint is outside the central 80% along either axis.',
        '- Surface distance is sampled unsquared symmetric Chamfer, half the two directed means, with 32,768 samples per surface. Sampling floor is retained per checkpoint; distances are not exact surface distances. Mesh diagnostics do not certify manifoldness or isotopy.',
        '- Three seeds per sampler are descriptive evidence. The two grids and three evaluation checkpoints bound what was observed; they do not prove continuous-time or continuum correctness. Timings include diagnostics/stream hashing and are not speed benchmarks.',
        '- Checkpoints, Adam state, RNG state, fixed datasets, partial attempts and source snapshots remain under `runs/gap-lr-continuations` in the original neural-geometry checkout. This export contains compact evidence and hashes.',
        '- The existing sign penalty remains unpromoted, and no fresh geometry, seed or reserved case was evaluated.','']
    lines+=['## Reproduction','',
        'From the original neural-geometry checkout, with the six hashed baseline checkpoints present. The runner resumes verified completed segments and retains interrupted attempts. Use a new output directory to rerun from scratch.','',
        '```powershell',"$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'",
        '.venv/Scripts/python.exe -m scripts.gap_lr_controls run --manifest docs/gap-controls/lr-continuations.json --out runs/gap-lr-continuations',
        '.venv/Scripts/python.exe -m scripts.report_gap_lr_controls --out runs/gap-lr-continuations --dest docs/gap-lr-continuations',
        '```','']
    (dest/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(dest/'export-audit.json',dict(files={p.relative_to(dest).as_posix():digest(p) for p in dest.rglob('*') if p.is_file() and p.name!='export-audit.json'},report_source_sha256=digest(__file__)))
    print(json.dumps(verification|{'artifacts':len(verification['artifacts'])}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();report(a.out,a.dest)
