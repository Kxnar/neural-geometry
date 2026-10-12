"""Hash-only evidence audit and paper-style report. Does not evaluate models."""
import argparse
import csv
from pathlib import Path
import statistics
import math
import torch

from neural_geometry_scale.study import ROOT, atomic_json, check_files, digest, identity, read
from scripts.gap_confirmation import DEST, INPUTS, ARMS, ENDS, qualify, verify_manifest,stream_hash,same
from scripts.gap_lr_controls import tensor_hash
from scripts.gap_density_controls import overlap


def csv_write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def weighted_mse(metrics):
    # The frozen independent/training pools have 50/25/25 composition.
    return .5*metrics['original']['mse']+.25*metrics['gap']['mse']+.25*metrics['interior']['mse']


def audit(out):
    torch.set_num_threads(4)
    m=read(out/'manifest.json'); verify_manifest(m)
    if digest(DEST/'batch-1-manifest.json')!=digest(out/'manifest.json'):
        # atomic_json normalizes JSON; equality is the semantic frozen contract.
        assert read(DEST/'batch-1-manifest.json')==m
    done=read(out/'run-complete.json'); records=read(out/'summary.json')['runs']
    assert done['checkpoints']==len(records)==48 and digest(out/'summary.json')==done['summary_sha256']
    for name,sha in m['source_hashes'].items(): assert digest(out/'source'/name)==sha
    for r in records:
        folder=ROOT/r['folder']; check_files(folder,read(folder/'complete.json'))
        assert read(folder/'metrics.json')=={k:v for k,v in r.items() if k!='folder'}
        assert digest(folder/'checkpoint.pt')==r['metadata']['checkpoint_sha256']
        assert r['metadata']['source_hashes']==m['source_hashes']
        assert r['training']==read(folder/'training.json')
        expected=dict(components=2,euler=4,closed_edges=True)
        probes=all(r['field'][name]['sign_accuracy']==1.0 for name in ['component_0','component_1','gap'])
        assert probes==r['field']['all_probe_signs_correct']
        assert set(r['evaluations'])=={'192','384'}
        for res,e in r['evaluations'].items():
            er=read(folder/f'evaluation-r{res}.json')
            assert er['metrics']==e and er['metadata']==r['metadata'] and e['resolution']==int(res)
            assert e['reference_resolved']==(e['reference_mesh']==expected)
            assert e['structure_match']==(e['mesh']==expected and e['reference_mesh']==expected)
        assert r['primary_gate']==(probes and all(e['mesh_present'] and e['mesh']==expected and e['reference_mesh']==expected for e in r['evaluations'].values()))
        def finite(value):
            if isinstance(value,float): return math.isfinite(value)
            if isinstance(value,dict): return all(finite(v) for v in value.values())
            if isinstance(value,list): return all(finite(v) for v in value)
            return True
        assert finite(r)
    for pair in m['pairs']:
        c=pair['config']; root=out/identity(c)
        assert digest(root/'orders.pt')==read(root/'orders.json')['sha256']
        orders=torch.load(root/'orders.pt',map_location='cpu',weights_only=True)
        data=torch.load(ROOT/pair['pool'],map_location='cpu',weights_only=True)
        independent=torch.load(ROOT/pair['independent'],map_location='cpu',weights_only=True)
        da=read(root/'data-audit.json')
        assert tensor_hash(data['points'],data['target'])==da['pool_sha256'] and tensor_hash(independent['points'],independent['target'])==da['independent_sha256']
        assert overlap(data['points'],independent['points'])==0
        for arm in ARMS:
            rows=sorted([r for r in records if r['metadata']['config']==c and r['training']['arm']==arm],key=lambda r:r['metadata']['step'])
            assert [r['metadata']['step'] for r in rows]==ENDS
            previous=pair['sha256']; start=20000
            for r in rows:
                t=r['training']; assert t['parent_sha256']==previous and t['start']==start
                assert t['point_presentations']==(t['end']-start)*arm['points']
                assert t['sample_rng_sha256']==t['initial_sample_rng_sha256']
                assert t['order_sha256']==read(root/'orders.json')['tensors'][arm['name']]
                assert tensor_hash(orders[arm['name']])==t['order_sha256'] and stream_hash(data,arm,orders[arm['name']],start,t['end'])==t['stream_sha256']
                assert t['pool_sha256']==da['pool_sha256'] and r['independent_points_sha256']==da['independent_sha256']
                saved=torch.load(ROOT/r['folder']/'checkpoint.pt',map_location='cpu',weights_only=True)
                assert saved['config']==c and saved['step']==t['end'] and all(int(s['step'])==t['end'] for s in saved['optimizer']['state'].values())
                assert tensor_hash(saved['sample_rng'])==t['sample_rng_sha256']
                if arm['name'] in ['dense','canonical']:
                    old=torch.load(ROOT/pair['historical'][arm['name']][str(t['end'])],map_location='cpu',weights_only=True)
                    assert all(same(saved[k],old[k]) for k in ['model','optimizer','sample_rng'])
                previous=r['metadata']['checkpoint_sha256']; start=t['end']
    exact=read(out/'reproduction.json')['checkpoints']
    expected_exact={(p,s,a,end) for p in ['global_narrow','local'] for s in [31,32,33] for a in ['dense','canonical'] for end in ENDS}
    assert len(exact)==24 and {(e['policy'],e['seed'],e['arm'],e['step']) for e in exact}==expected_exact
    decision=qualify(records,exact)
    budget=read(DEST/'budget.json')
    assert all(a['finished'] for a in budget['attempts'])
    charged=sum(a['charged_seconds'] for a in budget['attempts']); batches=[a['batch'] for a in budget['attempts'] if a.get('batch') is not None]
    assert charged<=7200 and set(batches)<={1,2,3} and 1 in batches
    if batches.count(1)>1:
        assert batches.count(1)==2 and budget['attempts'][4].get('recovery_of_attempt')==4
        assert read(DEST/'retained-io-failure/budget-before-recovery.json')['attempts']==budget['attempts'][:4]
    assert digest(ROOT/'results/gap-sampling/fresh-cases-reserved.json')==m['fresh_cases_sha256']
    return dict(checkpoints_verified=48,frozen_sources_verified=True,parent_chains_verified=True,
        input_files_verified=len(read(INPUTS/'inventory.json')['files']),exact_reproduction=exact,
        charged_seconds=charged,experiment_batches=sorted(set(batches)),charged_segments=len(batches),decision=decision,
        manifest_annotation='The embedded new protocol and hashed numerical code govern. Inherited protocol_sha256 and selection strings are legacy provenance fields, not the new executable protocol hash.',
        reserved_cases=8,reserved_generation_calls=0,reserved_evaluation_calls=0,reserved_access_scope='order gate batch 1 only; later confirmation access is recorded separately',reservation_sha256=m['fresh_cases_sha256'])


def report(out):
    verification=audit(out); decision=verification['decision']; records=read(out/'summary.json')['runs']
    if decision['gate_pass']: raise RuntimeError('Positive gate requires conditional confirmation before final report')
    atomic_json(DEST/'audit.json',verification); atomic_json(DEST/'decision.json',decision)
    atomic_json(DEST/'reservation-consumption.json',dict(archived_reservation='inputs/gap-confirmation/fresh-cases-reserved.json',
        sha256=verification['reservation_sha256'],status='all_reserved_untouched',generation_calls=[],evaluation_calls=[],cases_consumed=0,cases_remaining=8))
    compute=[]; trajectories=[]; checkpoints=[]; paired=[]
    for pair in read(out/'manifest.json')['pairs']:
        c=pair['config']
        for arm in ARMS:
            rr=sorted([r for r in records if r['metadata']['config']==c and r['training']['arm']==arm],key=lambda r:r['metadata']['step'])
            trajectories.append(dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],gate_26000=rr[0]['primary_gate'],gate_30000=rr[1]['primary_gate'],
                persistent=all(r['primary_gate'] for r in rr),regression=rr[0]['primary_gate'] and not rr[1]['primary_gate']))
            compute.append(dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],presentations=sum(r['training']['point_presentations'] for r in rr),
                training_seconds=sum(r['training']['seconds'] for r in rr),evaluation_seconds=sum(r['evaluation_seconds'] for r in rr),
                stage_time_complete=all(r['stage_seconds'] is not None for r in rr),
                stage_seconds=sum(r['stage_seconds'] for r in rr if r['stage_seconds'] is not None),
                stage_lower_bound_seconds=sum(r['stage_seconds'] if r['stage_seconds'] is not None else r['training']['seconds']+r['evaluation_seconds'] for r in rr),
                prep_and_io_seconds=sum(r['stage_seconds']-r['training']['seconds']-r['evaluation_seconds'] for r in rr if r['stage_seconds'] is not None)))
            for r in rr:
                row=dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],step=r['metadata']['step'],primary_gate=r['primary_gate'],
                    pool_fit=r['training']['final_fixed']['fit_gate'],training_pool_mse=weighted_mse(r['training']['final_fixed']),
                    independent_mse=weighted_mse(r['independent_fit']),independent_gap_mse=r['independent_fit']['gap']['mse'],
                    independent_gap_rmse=r['independent_fit']['gap']['rmse'],full_gap_sign_accuracy=r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'])
                for res,e in r['evaluations'].items():
                    row.update({f'chamfer_{res}':e['symmetric_chamfer'],f'components_{res}':e['mesh']['components'],f'euler_{res}':e['mesh']['euler'],
                                f'closed_{res}':e['mesh']['closed_edges'],f'reference_resolved_{res}':e['reference_resolved']})
                checkpoints.append(row)
                if arm['name']!='dense':
                    dense=next(d for d in records if d['metadata']['config']==c and d['metadata']['step']==r['metadata']['step'] and d['training']['arm']['name']=='dense')
                    paired.append(dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],step=r['metadata']['step'],candidate_gate=r['primary_gate'],dense_gate=dense['primary_gate'],
                        independent_gap_mse_delta=r['independent_fit']['gap']['mse']-dense['independent_fit']['gap']['mse'],
                        independent_gap_rmse_delta=r['independent_fit']['gap']['rmse']-dense['independent_fit']['gap']['rmse']))
    for name,rows in [('compute.csv',compute),('persistence.csv',trajectories),('checkpoints.csv',checkpoints),('paired-differences.csv',paired)]: csv_write(DEST/name,rows)
    dense=sum(r['training_seconds'] for r in compute if r['arm']=='dense')
    totals={}
    for arm in ARMS:
        rows=[r for r in compute if r['arm']==arm['name']]
        totals[arm['name']]=dict(persistent=sum(t['persistent'] for t in trajectories if t['arm']==arm['name']),
            stage_time_complete=all(r['stage_time_complete'] for r in rows),
            **{k:sum(r[k] for r in rows) for k in ['presentations','training_seconds','evaluation_seconds','stage_seconds','stage_lower_bound_seconds','prep_and_io_seconds']})
    failed=[t for t in trajectories if t['arm']!='dense' and not t['persistent']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    figure,axes=plt.subplots(1,3,figsize=(12.5,4.3),gridspec_kw={'width_ratios':[1,1,1.3]})
    pair_labels=[f'{p} / {s}' for p in ['global_narrow','local'] for s in [31,32,33]]
    for index,step in enumerate(ENDS):
        matrix=np.array([[int(t[f'gate_{step}']) for t in trajectories if f"{t['policy']} / {t['seed']}"==label] for label in pair_labels])
        axes[index].imshow(matrix,vmin=0,vmax=1,cmap=ListedColormap(['#bd3d3a','#2c8260']),aspect='auto')
        axes[index].set_xticks(range(4),[a['name'] for a in ARMS],rotation=35,ha='right')
        axes[index].set_yticks(range(6),pair_labels if index==0 else ['']*6)
        axes[index].set_title(f'Structural gate at {step:,} updates')
        for y in range(6):
            for x in range(4): axes[index].text(x,y,'pass' if matrix[y,x] else 'fail',ha='center',va='center',color='white',fontsize=9)
    names=[a['name'] for a in ARMS]; bottoms=np.zeros(4)
    for key,label,color in [('training_seconds','Training','#335f91'),('evaluation_seconds','Evaluation','#dd9d33'),('prep_and_io_seconds','Prep / I/O','#9c9c9c')]:
        values=np.array([totals[n][key]/60 for n in names]); axes[2].bar(range(4),values,bottom=bottoms,label=label,color=color); bottoms+=values
    axes[2].set_xticks(range(4),names,rotation=35,ha='right'); axes[2].set_ylabel('Measured recovery stage minutes / six pairs')
    axes[2].set_title('Stage cost (shuffle is a lower bound)'); axes[2].legend(fontsize=8)
    figure.suptitle('Frozen order-robustness gate: every pair retained',fontsize=12)
    figure.tight_layout(); figure.savefig(DEST/'order-robustness.png',dpi=180); plt.close(figure)
    if not decision['baseline_reproduction']:
        claim='The order-robustness campaign is inconclusive because its dense or canonical baseline did not reproduce the historical model/Adam/RNG tensors exactly.'
    elif failed:
        descriptions='; '.join(f"{t['arm']} {t['policy']} seed{t['seed']} ({'pass' if t['gate_26000'] else 'fail'} at26k, {'pass' if t['gate_30000'] else 'fail'} at30k)" for t in failed)
        claim=f"On the axis-aligned parallel-plate development geometry (gap0.02), changing only the 16,384-point balanced presentation order failed persistent structural recovery in {len(failed)} of the18 candidate trajectories: {descriptions}. Thus order-robust consistent preservation is unsupported; unseen reserved geometries were not tested."
    else:
        guards=[]
        for d in decision['arms']:
            if d['training_time_ratio']>.90: guards.append(f"{d['arm']} training/dense ratio{d['training_time_ratio']:.4f}>0.90")
            for c in d['comparisons']:
                for key,limit in [('gap_rmse_ratio',1.10),('chamfer_ratio_192',1.05),('chamfer_ratio_384',1.05)]:
                    if c[key]>limit: guards.append(f"{d['arm']} at{c['step']} {key}={c[key]:.5f}>{limit}")
                if c['gap_sign_delta']<-.005: guards.append(f"{d['arm']} at{c['step']} gap_sign_delta={c['gap_sign_delta']:.6f}<-0.005")
        claim='All three balanced orders retained observed persistent structure in all six axis-aligned gap0.02 development pairs, but the frozen criterion failed: '+ '; '.join(guards)+'. No reserved geometry was tested.'
    atomic_json(DEST/'claim.json',dict(claim=claim,scope='Development axis-aligned parallel plates gap0.02, policiesglobal_narrow/local seeds31-33;26k/30k and grids192/384',failed_trajectories=failed,totals=totals))
    lines=['# Balanced sampling: order robustness and the confirmation stopping gate','',
        '## Abstract','',claim,'',
        f"One new frozen experiment batch completed48 checkpoints. Whole experiment subprocess wall time and all three test attempts were charged: **{verification['charged_seconds']/60:.2f} minutes** of the120-minute allowance. Twelve targeted tests passed; two earlier fixture setup failures are retained. The maximum was three batches. The conditional unseen-geometry stage was not entered; all eight reserved cases and seeds71–73 remain unevaluated.",'',
        '## Hypothesis and protocol','',
        'The hypothesis was that the leading16,384-point balanced cycling method preserves dense observed structural recovery with substantially less compute despite changing its sampling order. The gate required success under canonical order, a one-block phase shift, and independently shuffled four-block cycles in all six development pairs. The executable protocol, code hashes, thresholds and input hashes were frozen before experiment launch in [batch-1-manifest.json](batch-1-manifest.json). No scientific parameters were changed after launch. A failed order, error guardrail, missing/nonfinite result, unresolved reference, or runtime criterion prohibits confirmation and tuning on reserved cases.','',
        '## Methods','',
        'The geometry is two axis-aligned parallel plates with gap0.02. Each pair uses a width128, depth3 SIREN and retained Adam state at20,000 updates, with policiesglobal_narrow/local and seeds31,32,33. All four arms share the same parent model, optimizer, sampling RNG and saved65,536-point dense pool. The pool has50% original sampler,25% full-footprint gap and25% facing interior points. Recovery uses10,000 ordinary equal-point MSE updates and the same cosine schedule from1e-4 to1e-6. Chunk accumulation divides squared-error sums by the total number of targets, with one Adam step per update.','',
        'Dense uses all65,536 points each update. Canonical uses balanced16,384-point blocks0,1,2,3; phase1 uses1,2,3,0. Shuffle precomputes2,500 independent permutations of the four blocks using a separate CPU Generator seeded700000+pair index, without touching model/global or sampling RNG. Every cycling update is balanced and every four-update cycle presents the complete pool exactly once. Arm execution order rotates by pair index. GPU execution is serial, deterministic and TF32-disabled. Schedules, stream hashes and checkpoints are retained.','',
        'The unchanged structural criterion combines all original gap/interior probe signs with two closed mesh components and Euler characteristic4 at both192 and384, requiring a resolved analytic reference. Persistent success means passing at26k and30k; it does not certify intervening updates or continuous topology. Mean independent gap RMSE must be at most1.10 times dense, mean sampled symmetric Chamfer at each grid at most1.05 times dense, and full-gap sampled sign accuracy no more than0.005 below dense at each checkpoint. Each candidate must use25% of dense recovery presentations and at most90% of aggregate dense recovery training time. Independent data contain65,536 disjoint points; full-gap signs use131,072 sampled probes.','',
        '## Results','',
        '![Every paired structural outcome and measured recovery-stage cost](order-robustness.png)','',
        '| Arm | Persistent pairs | Presentations, six pairs | Train seconds | Eval seconds | Prep/I/O seconds | Stage seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        t=totals[arm['name']]; prefix='' if t['stage_time_complete'] else '>='
        lines.append(f"| {arm['name']} | {t['persistent']}/6 | {t['presentations']:,} | {t['training_seconds']:.2f} | {t['evaluation_seconds']:.2f} | {prefix}{t['prep_and_io_seconds']:.2f} | {prefix}{t['stage_lower_bound_seconds']:.2f} |")
    lines += ['','| Policy | Seed | Arm | Gate26k | Gate30k | Persistent |','|---|---:|---|---|---|---|']
    for t in trajectories: lines.append(f"| {t['policy']} | {t['seed']} | {t['arm']} | {t['gate_26000']} | {t['gate_30000']} | {t['persistent']} |")
    lines += ['','All48 per-checkpoint ordinary training-pool/independent MSE values, RMSE, mesh diagnostics, fit failures and full-gap signs appear in [checkpoints.csv](checkpoints.csv). [paired-differences.csv](paired-differences.csv) retains matched seed differences. [decision.json](decision.json) provides every frozen threshold comparison. Dense/canonical historical reproduction is audited at all24 corresponding checkpoints by exact model,Adam and sampling-RNG tensor equality in [audit.json](audit.json).','',
        '## Compute accounting','',
        'Each candidate recovery trajectory presents163,840,000 points versus655,360,000 for dense:75% fewer. All dense labels and pool storage remain required. Training seconds include synchronized update work and logging. Evaluation seconds include independent field/gap data and both mesh resolutions. Stage time also includes initialization, pool metrics, transfer, checkpoint writes and hashes; other process overhead is charged in [budget.json](budget.json). The timing estimates are descriptive serial measurements on the same GPU; they are not hardware-independent guarantees. The common20k prefix was historical and was not rerun or newly timed, so these results support recovery-stage comparisons only; no measured full-pipeline runtime savings are claimed.','',
        'OneDrive denied an atomic completion-marker replacement after the local seed33 shuffle30k checkpoint and full measurements had been saved. The original failed attempt and1588.4591-second charge are retained. A separately hashed I/O-only continuation validated all eight temp-marker payload hashes and recovered those measurements without repeating training or evaluation, then ran only the two absent dense33 stages. Both segments share logical batch1 and both full wall charges remain in the ledger. The enclosing shuffle30k stage timer had not been persisted; it remains null. Its training/evaluation timers are present and counted exactly once; prep/I/O and stage totals for shuffle are lower bounds. The dense33 stages ran after a pause, weakening the intended execution-order timing comparison. Frozen numerical sources, inputs, presentation schedules and decision criteria remained unchanged. The new embedded protocol governs; inherited legacy protocol_sha256 and selection strings were left intact as historical provenance fields.','',
        '## Failure analysis and stopping decision','',decision['reason']+'.','']
    for d in decision['arms']:
        lines.append(f"- {d['arm']}: persistent{d['persistent_passes']}/6; training/dense ratio{d['training_time_ratio']:.4f}; qualifies={d['qualifies']}.")
        for c in d['comparisons']:
            lines.append(f"  At{c['step']}: gapRMSE ratio{c['gap_rmse_ratio']:.5f}; Chamfer192 ratio{c['chamfer_ratio_192']:.5f}; Chamfer384 ratio{c['chamfer_ratio_384']:.5f}; gap sign delta{c['gap_sign_delta']:.6f}; guards pass={c['passes']}.")
    lines += ['',
        'All failures are retained. The first two reserved cases would have been the17-degree gap0.012 and37-degree gap0.03 parallel plates with seeds71–73, under a separate complete freeze; that conditional design was never executed. The archived reservation and [reservation-consumption.json](reservation-consumption.json) record zero consumed cases. Simple static/random sampling comparisons on unseen cases were consequently not run. No balancing superiority, unseen-family generalization, novelty, confidence interval or continuous topology claim follows from this gate.','',
        '## Reproducibility','',
        'The local [input inventory](../../inputs/gap-confirmation/inventory.json) contains all six20k parent checkpoints, saved training/independent pools and24 historical dense/canonical reproduction checkpoints, with source paths, sizes and SHA-256 hashes. Runtime inputs are self-contained in this workspace; the original source checkout was read-only. Exact training code is under runs/gap-confirmation/batch-1/source. The complete artifact inventory covers code, inputs, models, schedules, metrics, logs and exports.','',
        'Use the Python/CUDA dependencies in requirements-windows-cu126.txt and a fresh output directory. The following reproduction is a separate future training run; it was not executed as an extra campaign batch. Set $python to the environment recorded in runs/gap-confirmation/batch-1/environment.json or an equivalent environment.','',
        '```powershell',"$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'",'$env:PYTHONDONTWRITEBYTECODE=\'1\'',
        '& $python -m scripts.gap_confirmation run --manifest results/gap-confirmation/batch-1-manifest.json --out runs/reproduce-order-gate',
        '& $python -m scripts.report_gap_confirmation --out runs/gap-confirmation/batch-1',
        '```','',
        'The run command checks frozen code and all input hashes and rejects an existing output directory. budget-run enforces the original campaign batch/time ledger and rejects a second batch1; use run only for a separately authorized reproduction. The reporter hashes artifacts and reads existing measurements; it performs no model evaluation.','']
    # Restore normal scientific prose spacing after mechanical numeric formatting.
    import re
    clean=[]; fenced=False
    for line in lines:
        if line.startswith('```'): fenced=not fenced
        if not fenced and not line.startswith('```'):
            # Format prose only; links, filenames and exact shell commands stay literal.
            chunks=re.split(r'(\[[^\]]*\]\([^)]*\)|requirements-windows-cu126\.txt)',line)
            for j in range(0,len(chunks),2):
                chunks[j]=re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', chunks[j])
                chunks[j]=re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', chunks[j])
            line=''.join(chunks)
        clean.append(line)
    text='\n'.join(clean)
    (DEST/'report.md').write_text(text,encoding='utf-8')
    artifacts={}
    for base in [INPUTS,out,DEST]:
        for path in base.rglob('*'):
            if path.is_file() and path.name not in ['run-artifacts.json','export-audit.json']:
                artifacts[path.relative_to(ROOT).as_posix()]=dict(sha256=digest(path),bytes=path.stat().st_size)
    for name in ['scripts/gap_confirmation.py','scripts/report_gap_confirmation.py','tests/test_gap_confirmation.py']:
        artifacts[name]=dict(sha256=digest(ROOT/name),bytes=(ROOT/name).stat().st_size)
    atomic_json(DEST/'run-artifacts.json',dict(files=artifacts,total_files=len(artifacts),total_bytes=sum(f['bytes'] for f in artifacts.values())))
    atomic_json(DEST/'export-audit.json',dict(files={p.name:digest(p) for p in DEST.iterdir() if p.is_file() and p.name!='export-audit.json'}))
    print(claim); print('Charged seconds:',verification['charged_seconds']); print('Artifacts:',len(artifacts))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,default=ROOT/'runs/gap-confirmation/batch-1')
    p.add_argument('--audit-only',action='store_true'); a=p.parse_args()
    if a.audit_only:
        verification=audit(a.out); atomic_json(DEST/'audit.json',verification); atomic_json(DEST/'decision.json',verification['decision'])
        import json
        print(json.dumps(verification['decision'],indent=2)); print('Charged seconds:',verification['charged_seconds'])
    else: report(a.out)
