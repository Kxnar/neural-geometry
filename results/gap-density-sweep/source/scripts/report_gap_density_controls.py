"""Verify and export the density experiment without selecting successful seeds."""
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

from neural_geometry_scale.study import ROOT, read, digest, atomic_json, check_files, identity
from scripts.gap_density_controls import SIZES, STEPS, GROUPS, nested_data, overlap
from scripts.gap_lr_controls import tensor_hash


def mean(xs):
    values=[x for x in xs if x is not None]
    return statistics.mean(values) if values else None


def write_csv(path,rows):
    rows=list(rows)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def audit(out):
    m=read(out/'manifest.json'); records=read(out/'summary.json')['runs']
    complete=read(out/'run-complete.json')
    assert complete['checkpoints']==len(records)==54
    assert complete['summary_sha256']==digest(out/'summary.json')
    assert digest(ROOT/'docs/gap-controls/fresh-cases.json')==m['fresh_cases_sha256']
    for name,sha in m['source_hashes'].items(): assert digest(out/'source'/name)==sha
    artifacts={}; rng_checks=0; baseline_comparisons=[]
    assert len({(r['metadata']['config']['policy'],r['metadata']['config']['seed'],r['training']['density'],r['metadata']['step']) for r in records})==54
    old=read(ROOT/'runs/gap-lr-continuations/fixed-summary.json')['runs']
    for pair in m['pairs']:
        c=pair['config']; root=out/identity(c)
        assert digest(pair['checkpoint'])==pair['sha256']
        baseline=torch.load(pair['checkpoint'],map_location='cpu',weights_only=True)
        assert baseline['step']==20000 and all(int(s['step'])==20000 for s in baseline['optimizer']['state'].values())
        assert digest(pair['original_data'])==pair['original_data_sha256']
        d=read(root/'data-audit.json'); check_files(root,d['files'])
        data=torch.load(root/'nested-data.pt',weights_only=True)
        independent=torch.load(root/'independent-data.pt',weights_only=True)
        prior=torch.load(pair['original_data'],weights_only=True)
        assert all(torch.equal(data[k][:4096],prior[k]) for k in data)
        regenerated=nested_data(c)
        assert all(torch.equal(data[k],regenerated[k]) for k in data)
        assert overlap(data['points'],independent['points'])==0
        regenerated_independent=nested_data(c,m['independent_points'],m['independent_seed'])
        assert all(torch.equal(independent[k],regenerated_independent[k]) for k in independent)
        artifacts[root.relative_to(out).as_posix()]={n:digest(root/n) for n in ['nested-data.pt','independent-data.pt','data-audit.json','baseline-errors.json']}
        for n in SIZES:
            group=sorted([r for r in records if r['metadata']['config']==c and r['training']['density']==n],key=lambda r:r['metadata']['step'])
            assert [r['metadata']['step'] for r in group]==STEPS
            assert torch.bincount(data['groups'][:n]).tolist()==[n//2,n//4,n//4]
            previous=pair['sha256']; start=20000
            for r in group:
                folder=Path(r['folder']); hashes=read(folder/'complete.json'); check_files(folder,hashes)
                assert read(folder/'metrics.json')=={k:v for k,v in r.items() if k!='folder'}
                assert r['metadata']['source_hashes']==m['source_hashes']
                assert r['metadata']['checkpoint_sha256']==digest(folder/'checkpoint.pt')
                t=r['training']; assert t==read(folder/'training.json')
                assert t['parent_sha256']==previous and t['start']==start
                assert t['point_presentations']==(t['end']-start)*n and t['end']==r['metadata']['step']
                assert t['stream_sha256']==tensor_hash(data['points'][:n],data['target'][:n])==d['prefixes'][str(n)]
                assert t['initial_sample_rng_sha256']==t['sample_rng_sha256']==tensor_hash(baseline['sample_rng'])
                assert r['independent_points_sha256']==tensor_hash(independent['points'],independent['target'])
                for split,expected_count in [('training',n),('independent',m['independent_points'])]:
                    strata=r['surface_distance_errors'][split]
                    assert strata['all']['all']['samples']==expected_count
                    assert sum(strata[name]['all']['samples'] for name in GROUPS)==expected_count
                    for name,bins in strata.items():
                        assert sum(v['samples'] for key,v in bins.items() if key!='all')==bins['all']['samples']
                        for v in bins.values():
                            assert 0<=v['sign_errors']<=v['sign_samples']<=v['samples']
                            if v['samples']: assert 0<=v['mae']<=v['rmse']+1e-15
                    for name in ['gap','interior']:
                        assert strata[name+'_central']['all']['samples']+strata[name+'_outer']['all']['samples']==strata[name]['all']['samples']
                saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
                assert saved['config']==c and saved['density']==n and saved['step']==t['end']
                assert all(int(s['step'])==t['end'] for s in saved['optimizer']['state'].values())
                assert torch.equal(saved['sample_rng'],baseline['sample_rng']); rng_checks+=1
                assert set(r['evaluations'])=={'192','384'}
                assert r['primary_gate']==(r['field']['all_probe_signs_correct'] and all(e['structure_match'] for e in r['evaluations'].values()))
                assert r['full_gap_gate']==(r['full_gap']['regions']['whole_gap']['all_margins']['errors']==0)
                artifacts[folder.relative_to(out).as_posix()]=hashes
                if n==4096:
                    reference=next(x for x in old if x['metadata']['config']==c and x['metadata']['step']==t['end'])
                    ref_saved=torch.load(Path(reference['folder'])/'checkpoint.pt',map_location='cpu',weights_only=True)
                    baseline_comparisons.append(dict(policy=c['policy'],seed=c['seed'],step=t['end'],
                        max_parameter_abs_difference=max(float((saved['model'][k]-ref_saved['model'][k]).abs().max()) for k in saved['model']),
                        optimizer_state_equal=all(torch.equal(value,ref_saved['optimizer']['state'][key][name]) for key,state in saved['optimizer']['state'].items() for name,value in state.items()),
                        gap_accuracy_difference=r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy']-reference['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy']))
                previous=r['metadata']['checkpoint_sha256']; start=t['end']
    assert len({r['full_gap']['points_sha256'] for r in records})==1
    for policy in ['local','global_narrow']:
        assert len({r['independent_points_sha256'] for r in records if r['metadata']['config']['policy']==policy})==1
    interrupted=[]
    for folder in out.glob('*/n-*/step-*/attempt-*'):
        if (folder/'training-complete.json').exists(): continue
        log=folder/'training.jsonl'
        if not log.exists(): continue
        history=[json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        end=int(folder.parent.name.split('-')[1]); n=int(folder.parent.parent.name.split('-')[1])
        start={23000:20000,26000:23000,30000:26000}[end]
        last=history[-1]['step'] if history else start
        interrupted.append(dict(folder=str(folder),start=start,last_logged_step=last,density=n,
            logged_discarded_updates=last-start,logged_discarded_point_presentations=(last-start)*n,
            unlogged_tail='At most 99 further completed updates before next flush, plus a possible partial update. Exact elapsed compute unavailable.'))
        artifacts[folder.relative_to(out).as_posix()]={p.name:digest(p) for p in folder.iterdir() if p.is_file()}
    return m,records,dict(checkpoints_verified=54,adam_steps_verified=True,nested_datasets_verified=True,
        exact_prior_4096_points=True,checkpoint_parent_chains_verified=True,rng_states_verified=rng_checks,
        independent_point_overlap=0,shared_independent_probes_verified=True,source_snapshots_verified=True,
        fresh_reservation_unchanged=True,prior_4096_reproduction=baseline_comparisons,interrupted_attempts=interrupted,artifacts=artifacts)


def flatten(r):
    c=r['metadata']['config']; t=r['training']
    row=dict(policy=c['policy'],seed=c['seed'],density=t['density'],step=r['metadata']['step'],
        primary_gate=r['primary_gate'],full_gap_gate=r['full_gap_gate'],training_fit_gate=t['final_fixed']['fit_gate'],
        train_seconds=t['seconds'],evaluation_seconds=r['evaluation_seconds'],point_presentations=t['point_presentations'],
        full_gap_sign_accuracy=r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'])
    for split,f in [('training',t['final_fixed']),('independent',r['independent_fit'])]:
        for name in GROUPS:
            for key,value in f[name].items(): row[f'{split}_{name}_{key}']=value
    for res,e in r['evaluations'].items():
        for key in ['symmetric_chamfer','reference_sampling_floor','mesh_present','reference_resolved','structure_match']:
            row[f'r{res}_{key}']=e[key]
        for key,value in e['mesh'].items(): row[f'r{res}_{key}']=value
    return row


def distance_rows(records):
    for r in records:
        c=r['metadata']['config']
        for split,groups in r['surface_distance_errors'].items():
            for group,bins in groups.items():
                for bucket,values in bins.items():
                    yield dict(policy=c['policy'],seed=c['seed'],density=r['training']['density'],step=r['metadata']['step'],
                               split=split,group=group,distance_bin=bucket,**values)


def charts(records,dest):
    colors=['#2875ba','#df8621','#5c9b51']
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    for ax,(policy,seed) in zip(axes.flat,[(p,s) for p in ['global_narrow','local'] for s in [31,32,33]]):
        for n,color in zip(SIZES,colors):
            g=sorted([r for r in records if r['metadata']['config']['policy']==policy and r['metadata']['config']['seed']==seed and r['training']['density']==n],key=lambda r:r['metadata']['step'])
            ax.plot(STEPS,[100*r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'] for r in g],'-o',color=color,label=f'{n:,}')
        ax.set_title(f'{policy}, seed {seed}'); ax.grid(alpha=.2);ax.set_xlabel('Total updates');ax.set_ylabel('Independent gap signs correct (%)')
    axes[0,0].legend(title='Fixed points')
    fig.suptitle('Density intervention: all seeds and checkpoints; sign accuracy is separate from the structural gate')
    fig.tight_layout();fig.savefig(dest/'density-curves.png',dpi=160);plt.close(fig)
    bins=['0_0.001','0.001_0.0025','0.0025_0.005','0.005_0.01']
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
    for row,policy in enumerate(['global_narrow','local']):
        for col,group in enumerate(GROUPS):
            ax=axes[row,col]
            for n,color in zip(SIZES,colors):
                rs=[r for r in records if r['metadata']['config']['policy']==policy and r['training']['density']==n and r['metadata']['step']==30000]
                for split,style in [('training','--'),('independent','-')]:
                    ys=[mean(r['surface_distance_errors'][split][group][b]['rmse'] for r in rs) for b in bins]
                    ax.plot(range(4),ys,style+'o',color=color,label=f'{n:,} {split}')
            ax.set_title(f'{policy}: {group}');ax.set_yscale('log');ax.grid(alpha=.2);ax.set_ylabel('SDF RMSE');ax.set_xticks(range(4),['0–.001','.001–.0025','.0025–.005','.005–.01'],rotation=15)
            ax.set_xlabel('Absolute true SDF / distance to surface')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=6,fontsize=8)
    fig.suptitle('30k: training (dashed) versus matched independent data (solid), mean of three seed RMSEs')
    fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(dest/'surface-distance-errors.png',dpi=160);plt.close(fig)


def report(out,dest):
    torch.set_num_threads(4)
    m,records,verification=audit(out)
    dest.mkdir(parents=True,exist_ok=True)
    for name in ['manifest.json','environment.json','summary.json','run-complete.json']:
        shutil.copyfile(out/name,dest/name)
    if (out/'validation.json').exists(): shutil.copyfile(out/'validation.json',dest/'validation.json')
    shutil.copyfile(ROOT/'docs/gap-density-protocol.md',dest/'protocol.md')
    shutil.copytree(out/'source',dest/'source',dirs_exist_ok=True)
    shutil.copyfile(__file__,dest/'source/scripts/report_gap_density_controls.py')
    shutil.copyfile(ROOT/'tests/test_gap_density_controls.py',dest/'source/test_gap_density_controls.py')
    atomic_json(dest/'audit.json',verification)
    atomic_json(dest/'interrupted-attempts.json',dict(attempts=verification['interrupted_attempts']))
    previous_records=read(ROOT/'runs/gap-lr-continuations/summary.json')['runs']
    starts=[]
    for pair in m['pairs']:
        r=next(r for r in previous_records if r['metadata']['config']==pair['config'] and r['metadata']['step']==20000 and r['training']['schedule']=='cosine')
        assert r['metadata']['checkpoint_sha256']==pair['sha256']
        check_files(Path(r['folder']),read(Path(r['folder'])/'complete.json'))
        starts.append(r)
    atomic_json(dest/'starting-geometry.json',dict(runs=starts))
    write_csv(dest/'checkpoints.csv',map(flatten,records))
    write_csv(dest/'surface-distance-errors.csv',distance_rows(records))
    baselines=[]
    for pair in m['pairs']:
        c=pair['config']; b=read(out/identity(c)/'baseline-errors.json')
        assert b['checkpoint_sha256']==pair['sha256']
        for n in SIZES:
            for split,groups in [('training',b['training'][str(n)]),('independent',b['independent'])]:
                for group,bins in groups.items():
                    for bucket,values in bins.items():
                        baselines.append(dict(policy=c['policy'],seed=c['seed'],density=n,step=20000,
                            split=split,group=group,distance_bin=bucket,**values))
    write_csv(dest/'baseline-errors.csv',baselines)
    charts(records,dest)
    trajectories=[]; comparisons=[]; compute=[]; regressions=[]
    for pair in m['pairs']:
        c=pair['config']; reference=None
        for n in SIZES:
            g=sorted([r for r in records if r['metadata']['config']==c and r['training']['density']==n],key=lambda r:r['metadata']['step'])
            last=g[-1]; gates=[r['primary_gate'] for r in g]
            fit=[r['training']['final_fixed']['fit_gate'] for r in g]
            sustained=all(gates[-2:])
            for before,after in zip(g,g[1:]):
                if before['primary_gate'] and not after['primary_gate']:
                    regressions.append(dict(policy=c['policy'],seed=c['seed'],density=n,
                        passing_step=before['metadata']['step'],failing_step=after['metadata']['step'],
                        probe_gate_at_failure=after['field']['all_probe_signs_correct'],
                        failing_meshes={res:e['mesh'] for res,e in after['evaluations'].items()}))
            trajectories.append(dict(policy=c['policy'],seed=c['seed'],density=n,primary=gates,training_fit=fit,
                sustained=sustained,first_observed_passing_suffix=23000 if all(gates) else 26000 if sustained else None,
                final_classification='persistent structural pass' if sustained else 'training fit passes; geometry unresolved' if fit[-1] else 'training fit unresolved; optimisation/capacity remain open'))
            compute.append(dict(policy=c['policy'],seed=c['seed'],density=n,updates=sum(r['training']['end']-r['training']['start'] for r in g),
                point_presentations=sum(r['training']['point_presentations'] for r in g),train_seconds=sum(r['training']['seconds'] for r in g),evaluation_seconds=sum(r['evaluation_seconds'] for r in g)))
            if reference is None: reference=last
            else:
                a=reference; b=last
                comparisons.append(dict(policy=c['policy'],seed=c['seed'],density=n,
                    gap_sign_accuracy_delta=b['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy']-a['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'],
                    independent_gap_rmse_delta=b['independent_fit']['gap']['rmse']-a['independent_fit']['gap']['rmse'],
                    training_gap_rmse_delta=b['training']['final_fixed']['gap']['rmse']-a['training']['final_fixed']['gap']['rmse'],
                    **{f'chamfer{res}_delta':None if b['evaluations'][res]['symmetric_chamfer'] is None or a['evaluations'][res]['symmetric_chamfer'] is None else b['evaluations'][res]['symmetric_chamfer']-a['evaluations'][res]['symmetric_chamfer'] for res in ['192','384']}))
    atomic_json(dest/'persistence.json',dict(trajectories=trajectories))
    atomic_json(dest/'regressions.json',dict(regressions=regressions))
    write_csv(dest/'paired-density-differences.csv',comparisons);write_csv(dest/'compute.csv',compute)
    dense=[t for t in trajectories if t['density']==65536]
    restored_sizes=[n for n in SIZES[1:] if all(t['sustained'] for t in trajectories if t['density']==n)]
    decision=dict(persistent_restoration_sizes=restored_sizes,
        persistent_restoration_sizes_by_sampler={policy:[n for n in SIZES if all(t['sustained'] for t in trajectories if t['policy']==policy and t['density']==n)] for policy in ['global_narrow','local']},
        dense_persistent_passes=sum(t['sustained'] for t in dense),dense_final_training_fit_passes=sum(t['training_fit'][-1] for t in dense),
        all_structural_checkpoints_passing=sum(r['primary_gate'] for r in records),fresh_cases_evaluated=False,new_loss_advanced=False,
        next_step='Pursue coverage and sampling efficiency with ordinary MSE; dense-data residual fit remains unresolved.' if restored_sizes else 'Separate fitted-data geometric failures from optimisation/capacity limits before changing the objective.',
        interpretation='Coverage is supported by persistent restoration across all six pairs.' if restored_sizes else
            'No consistent persistent restoration. Separate fitted-data geometry failures from unresolved training fit; density alone has not repaired geometry under this update budget.')
    atomic_json(dest/'decision.json',decision)
    lines=['# Does sparse training coverage cause the remaining geometry failures?','',decision['interpretation']+' '+decision['next_step'],'',
        'Development geometry only: parallel plates, gap 0.02; ordinary-MSE SIREN width 128/depth 3; global_narrow and local samplers; seeds 31, 32, 33. All 18 arms and 54 checkpoints are retained. No new loss is advanced; fresh cases remain untouched.','',
        '| Fixed points | Persistent global_narrow | Persistent local | Final training-fit passes | Mean independent gap signs |',
        '|---:|---:|---:|---:|---:|']
    for n in SIZES:
        g=[r for r in records if r['training']['density']==n and r['metadata']['step']==30000]
        counts=[sum(t['sustained'] for t in trajectories if t['density']==n and t['policy']==p) for p in ['global_narrow','local']]
        lines.append(f"| {n:,} | {counts[0]}/3 | {counts[1]}/3 | {sum(r['training']['final_fixed']['fit_gate'] for r in g)}/6 | {mean(r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'] for r in g):.2%} |")
    smallest=[r for r in records if r['training']['density']==4096 and r['metadata']['step']==30000]
    largest=[r for r in records if r['training']['density']==65536 and r['metadata']['step']==30000]
    train_means=[mean(r['training']['final_fixed']['gap']['rmse'] for r in g) for g in [smallest,largest]]
    test_means=[mean(r['independent_fit']['gap']['rmse'] for r in g) for g in [smallest,largest]]
    lines+=['',f"Mean gap RMSE (training / independent) changes from {train_means[0]:.6g} / {test_means[0]:.6g} at 4,096 points to {train_means[1]:.6g} / {test_means[1]:.6g} at 65,536. Training error rises across densities while independent error falls and the train–independent gap narrows. This supports sparse-data overfitting as a contributor. Structural recovery is bounded by the existing gate: dense training fit and perfect independent signs are not achieved.",'',
        '## Controlled intervention','',
        'Each density starts from its corresponding exact 20,000-update decayed model and Adam state. The 4,096-point dataset is byte-identical to the preceding diagnostic. Larger arms are nested prefixes of balanced blocks, with 50% original-sampler, 25% uniform full-gap and 25% facing-interior points. All arms receive 10,000 updates on the same endpoint-inclusive cosine schedule (1e-4 to 1e-6), with no optimiser reset between evaluation checkpoints.','',
        'Gradients accumulate chunk sums divided by the total number of target elements, then Adam steps once. Chunk size is 16,384; the two smaller arms use one chunk and the largest uses four. This is exact full-dataset ordinary MSE in the mathematical sense; floating-point accumulation can differ from a monolithic reduction.','',
        'Independent matched-distribution data use 65,536 points per sampler, common across seeds/densities, with zero exact-coordinate overlap with training data. The existing 131,072 requested full-gap probes and the original structural evaluator are also reused. Errors are stratified by absolute true SDF, for each subset and central/outer footprint; empty bins are missing, not zero.','',
        '## Checkpoint outcomes','',
        '| Sampler | Fixed points | Updates | Training fit | Gap train RMSE | Gap independent RMSE | Independent gap signs | Chamfer 192 / 384 | Structural gate |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for policy in ['global_narrow','local']:
        for n in SIZES:
            for step in STEPS:
                g=[r for r in records if r['metadata']['config']['policy']==policy and r['training']['density']==n and r['metadata']['step']==step]
                train_rmse=mean(r['training']['final_fixed']['gap']['rmse'] for r in g)
                test_rmse=mean(r['independent_fit']['gap']['rmse'] for r in g)
                accuracy=mean(r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'] for r in g)
                cds=[mean(r['evaluations'][res]['symmetric_chamfer'] for r in g) for res in ['192','384']]
                distance=' / '.join('missing' if v is None else f'{v:.6f}' for v in cds)
                lines.append(f"| {policy} | {n:,} | {step:,} | {sum(r['training']['final_fixed']['fit_gate'] for r in g)}/3 | {train_rmse:.6g} | {test_rmse:.6g} | {accuracy:.2%} | {distance} | {sum(r['primary_gate'] for r in g)}/3 |")
    lines+=['','Training-fit gate is unchanged: every original/gap/interior subset RMSE ≤0.001 and no wrong training signs with |true SDF|>0.001. Larger sets give a zero-error gate more opportunities to fail, so compare RMSE and error counts/denominators as well as binary fit. Means are descriptive seed means, not confidence intervals.','',
        '## Persistence and individual decisions','',
        'The structural gate retains all original probe signs and requires two components, Euler characteristic four, closed edges and a resolved analytic reference at BOTH 192 and 384. A persistent observed pass requires 26k and 30k; observations say nothing definitive between checkpoints. Full-gap signs remain a separate diagnostic.','',
        '| Sampler | Seed | Points | Gate 23k / 26k / 30k | Fit 23k / 26k / 30k | Interpretation |',
        '|---|---:|---:|---|---|---|']
    fmt=lambda values:' / '.join('pass' if v else 'fail' for v in values)
    for t in trajectories:
        lines.append(f"| {t['policy']} | {t['seed']} | {t['density']:,} | {fmt(t['primary'])} | {fmt(t['training_fit'])} | {t['final_classification']} |")
    lines+=['',f'Observed passing-to-failing transitions: {len(regressions)}.','']
    for r in regressions:
        meshes='; '.join(f"r{res}: components={v['components']}, Euler={v['euler']}" for res,v in r['failing_meshes'].items())
        lines.append(f"- {r['policy']}, seed {r['seed']}, {r['density']:,} points: pass at {r['passing_step']:,}, fail at {r['failing_step']:,}; {meshes}.")
    lines+=['','A failed fit gate alone is not proof that optimisation stalled. The per-subset errors and the training history show whether error decreased; failure under this schedule/budget leaves optimisation and capacity unresolved. A passing finite-data fit alongside failed structure motivates bridge localisation between samples, but does not by itself prove where bridges form.','',
        '## Training progress despite strict fit failures','',
        '| Points | Subset | Mean initial RMSE | Mean final RMSE | RMSE improvement pairs | Final strong-margin errors / tested |',
        '|---:|---|---:|---:|---:|---:|']
    for n in SIZES:
        initial=[r for r in records if r['training']['density']==n and r['metadata']['step']==23000]
        final=[r for r in records if r['training']['density']==n and r['metadata']['step']==30000]
        for group in GROUPS:
            before=[r['training']['initial_fixed'][group]['rmse'] for r in initial]
            after=[r['training']['final_fixed'][group]['rmse'] for r in final]
            errors=sum(r['training']['final_fixed'][group]['margin_gt_001_errors'] for r in final)
            count=sum(r['training']['final_fixed'][group]['margin_gt_001_samples'] for r in final)
            lines.append(f'| {n:,} | {group} | {mean(before):.6g} | {mean(after):.6g} | {sum(a<b for a,b in zip(after,before))}/6 | {errors} / {count} |')
    lines+=['',
        '## Matched density differences at 30k','',
        '| Dense arm vs 4,096 | Gap signs improvement pairs | Mean sign change (pp) | Mean independent gap RMSE change | Mean Chamfer 192 / 384 change |',
        '|---|---:|---:|---:|---:|']
    for n in SIZES[1:]:
        g=[r for r in comparisons if r['density']==n]
        cd=[mean(r[f'chamfer{res}_delta'] for r in g) for res in ['192','384']]
        cds=' / '.join('missing' if v is None else f'{v:+.6f}' for v in cd)
        lines.append(f"| {n:,} | {sum(r['gap_sign_accuracy_delta']>0 for r in g)}/6 | {100*mean(r['gap_sign_accuracy_delta'] for r in g):+.3f} | {mean(r['independent_gap_rmse_delta'] for r in g):+.6g} | {cds} |")
    lines+=['','A density increase changes empirical coverage while keeping proportions, starting state, objective and update count fixed. Compute is deliberately not matched. Better sampled signs/distances without persistent structure are partial coverage benefits, not a geometric repair.','',
        '## Extra compute','',
        '| Points | Updates per arm | Presentations per arm | Work ratio | Six-arm training seconds | Measured time ratio | Evaluation seconds |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    base=sum(r['train_seconds'] for r in compute if r['density']==4096)
    for n in SIZES:
        g=[r for r in compute if r['density']==n]
        seconds=sum(r['train_seconds'] for r in g)
        lines.append(f"| {n:,} | 10,000 | {n*10000:,} | {n/4096:g}x | {seconds:.1f} | {seconds/base:.2f}x | {sum(r['evaluation_seconds'] for r in g):.1f} |")
    lines+=['',f"Completed-trajectory training presentations: {sum(r['point_presentations'] for r in compute):,}. Measured completed training: {sum(r['train_seconds'] for r in compute)/60:.2f} minutes; evaluation: {sum(r['evaluation_seconds'] for r in compute)/60:.2f} minutes. Training timers include full-loss logging/synchronisation and exclude dataset generation, checkpoint I/O and evaluation; evaluation timers include metric I/O. These are workload costs on one RTX 2080 Super, not a speed benchmark.",'',
        f"Interrupted attempts retained: {len(verification['interrupted_attempts'])}. Their logs record at least {sum(a['logged_discarded_point_presentations'] for a in verification['interrupted_attempts']):,} additional discarded point presentations, excluded from the matched completed-trajectory totals above. Each may have an unlogged tail of fewer than 100 completed updates plus a partial update. Their exact elapsed compute is unavailable; see [interruption audit](interrupted-attempts.json). Verified complete segments were reused and only interrupted segments restarted from the previous saved Adam checkpoint.",'',
        '## Evidence and limits','',
        '- [Checkpoint measurements](checkpoints.csv), [all surface-distance bins](surface-distance-errors.csv), [20k baseline bins](baseline-errors.csv), [starting geometry](starting-geometry.json), [paired differences](paired-density-differences.csv), [compute](compute.csv), [persistence](persistence.json), [audit](audit.json), [frozen manifest](manifest.json).',
        '- [Density curves](density-curves.png), [training versus independent surface-distance errors](surface-distance-errors.png).',
        f"- Missing meshes: {sum(not e['mesh_present'] for r in records for e in r['evaluations'].values())} of 108 grid evaluations. Every absent/incorrect mesh fails the structural gate; missing distances are excluded from means.",
        '- Sampled surface distance is unsquared symmetric Chamfer, half the directed means, using 32,768 surface points. Reference sampling floors remain recorded. This is not an exact surface distance or a topology certificate.',
        '- Three paired seeds per sampler, one development geometry, one model capacity and one update budget cannot establish a universal cause. A fitted sparse arm and an unfitted dense arm do not cleanly distinguish interpolation from optimisation.',
        f"- Legacy 4,096-point control reproduction: maximum absolute parameter difference {max(x['max_parameter_abs_difference'] for x in verification['prior_4096_reproduction']):.6g}; exact Adam state match at {sum(x['optimizer_state_equal'] for x in verification['prior_4096_reproduction'])}/18 checkpoints.",
        '- Full model/Adam/RNG states, nested and independent datasets, baseline errors, attempt folders and frozen sources are retained under runs/gap-density-sweep in the original checkout. This export is compact; the audit records their hashes.',
        '- The audit verifies the fresh-case reservation hash and exact legacy 4,096-point prefix. No fresh cases or training seeds were evaluated. No new objective is promoted.','',
        '## Reproduction','',
        'From the original neural-geometry checkout, with the six verified decayed checkpoints present. Resume uses verified complete segments and retains interrupted attempts. Use a new output folder for a fresh execution.','',
        '```powershell',"$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'",
        '.venv/Scripts/python.exe -m scripts.gap_density_controls run --manifest docs/gap-controls/density-sweep.json --out runs/gap-density-sweep',
        '.venv/Scripts/python.exe -m scripts.report_gap_density_controls --out runs/gap-density-sweep --dest docs/gap-density-sweep',
        '```','']
    (dest/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(dest/'export-audit.json',dict(files={p.relative_to(dest).as_posix():digest(p) for p in dest.rglob('*') if p.is_file() and p.name!='export-audit.json'},report_source_sha256=digest(__file__)))
    print(decision,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();report(a.out,a.dest)
