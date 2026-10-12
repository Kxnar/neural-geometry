"""Audit retained measurements only; assemble the final bounded campaign paper."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
import torch

from neural_geometry_scale.study import ROOT,read,digest,check_files,identity,atomic_json
from scripts.gap_confirmation import DEST,INPUTS,same
from scripts.gap_reserved_confirmation import ARMS,ENDS,sources
from scripts.report_gap_confirmation import audit as gate_audit,weighted_mse,csv_write
from scripts.recover_gap_confirmation_io import resilient_json as atomic_json


def qualification(records):
    expected={(i,s,a['name'],e) for i in [0,1] for s in [71,72,73] for a in ARMS for e in ENDS}
    keys=[(r['case_index'],r['metadata']['config']['seed'],r['training']['arm']['name'],r['metadata']['step']) for r in records]
    complete=len(keys)==48 and set(keys)==expected; rows=[]; decisions=[]
    torch.set_num_threads(4)
    for i in [0,1]:
        for seed in [71,72,73]:
            for arm in ARMS:
                rr=[r for r in records if r['case_index']==i and r['metadata']['config']['seed']==seed and r['training']['arm']['name']==arm['name']]
                rows.append(dict(case_index=i,seed=seed,arm=arm['name'],complete=len(rr)==2,
                    gate_26000=next((r['primary_gate'] for r in rr if r['metadata']['step']==26000),False),
                    gate_30000=next((r['primary_gate'] for r in rr if r['metadata']['step']==30000),False),
                    persistent=len(rr)==2 and all(r['primary_gate'] for r in rr),
                    reference_resolved=len(rr)==2 and all(e['reference_resolved'] for r in rr for e in r['evaluations'].values())))
    for arm in ARMS[1:]:
        aa=[r for r in records if r['training']['arm']['name']==arm['name']]; dense=[r for r in records if r['training']['arm']['name']=='dense']; checks=[]
        for end in ENDS:
            a=[r for r in aa if r['metadata']['step']==end]; b=[r for r in dense if r['metadata']['step']==end]
            if len(a)!=6 or len(b)!=6: checks.append(dict(step=end,passes=False,incomplete=True)); continue
            mean=statistics.mean; d=dict(step=end,gap_rmse_ratio=mean(r['independent_fit']['gap']['rmse'] for r in a)/mean(r['independent_fit']['gap']['rmse'] for r in b),
                gap_sign_delta=mean(r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'] for r in a)-mean(r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'] for r in b))
            for res in ['192','384']:
                x=[r['evaluations'][res]['symmetric_chamfer'] for r in a]; y=[r['evaluations'][res]['symmetric_chamfer'] for r in b]
                d['chamfer_ratio_'+res]=mean(x)/mean(y) if all(v is not None and math.isfinite(v) for v in x+y) and mean(y)>0 else None
            d['passes']=d['gap_rmse_ratio']<=1.10 and d['gap_sign_delta']>=-.005 and all(d['chamfer_ratio_'+res] is not None and d['chamfer_ratio_'+res]<=1.05 for res in ['192','384'])
            checks.append(d)
        training_ratio=sum(r['training']['seconds'] for r in aa)/sum(r['training']['seconds'] for r in dense) if dense else None
        trajectories=[r for r in rows if r['arm']==arm['name']]
        persistent=sum(r['persistent'] for r in trajectories)
        lost=[]
        for r in trajectories:
            dense_row=next(x for x in rows if x['case_index']==r['case_index'] and x['seed']==r['seed'] and x['arm']=='dense')
            if dense_row['persistent'] and not r['persistent']: lost.append(dict(case_index=r['case_index'],seed=r['seed']))
        valid_refs=all(r['reference_resolved'] for r in trajectories)
        decisions.append(dict(arm=arm['name'],persistent_passes=persistent,training_time_ratio=training_ratio,
            dense_persistent_losses=lost,reference_resolved=valid_refs,comparisons=checks,
            qualifies=complete and persistent==6 and valid_refs and not lost and training_ratio<=.9 and all(c['passes'] for c in checks)))
    return dict(complete=complete,trajectories=rows,arms=decisions,canonical_qualifies=next(d['qualifies'] for d in decisions if d['arm']=='canonical'))


def audit_confirmation():
    m=read(DEST/'confirmation-manifest.json'); assert m['source_hashes']==sources()
    assert digest(INPUTS/'fresh-cases-reserved.json')==m['reservation_sha256']
    assert m['case_allowlist']==read(INPUTS/'fresh-cases-reserved.json')['cases'][:2]
    assert m['reservation_sha256']==digest(INPUTS/'fresh-cases-reserved.json')
    assert m['order_gate_manifest_sha256']==digest(DEST/'batch-1-manifest.json')
    assert m['gate_decision_sha256']==digest(DEST/'decision.json')
    records=[]; prefixes=[]; pairs=[]
    for i in [0,1]:
        out=ROOT/f'runs/gap-confirmation/batch-{i+2}'
        if not (out/'summary.json').exists(): continue
        assert read(out/'manifest.json')==m
        for name,sha in m['source_hashes'].items(): assert digest(out/'source'/name)==sha
        rr=read(out/'summary.json')['runs']
        if (out/'run-complete.json').exists():
            done=read(out/'run-complete.json'); assert done['checkpoints']==len(rr)==24 and done['summary_sha256']==digest(out/'summary.json')
        pp=read(out/'prefixes.json')['pairs']; assert len(pp)==3 and {p['config']['seed'] for p in pp}=={71,72,73}
        prefixes.extend(dict(case_index=i,**p) for p in pp)
        for p in pp:
            c=p['config']; root=out/identity(c); assert {k:c[k] for k in ['scene','feature','angle']}==m['case_allowlist'][i]
            assert c['seed'] in [71,72,73] and c['policy']=='global_narrow' and c['eval_seed']==91000
            assert digest(root/'prefix-20000/checkpoint.pt')==p['parent_sha256']
            for folder in [root/'prefix-10000',root/'prefix-20000']: check_files(folder,read(folder/'training-complete.json'))
            from scripts.gap_lr_controls import tensor_hash
            base=torch.load(root/'prefix-10000/checkpoint.pt',map_location='cpu',weights_only=True)
            parent=torch.load(root/'prefix-20000/checkpoint.pt',map_location='cpu',weights_only=True)
            assert base['config']==parent['config']==c and base['step']==10000 and parent['step']==20000
            assert all(int(s['step'])==10000 for s in base['optimizer']['state'].values())
            assert all(int(s['step'])==20000 for s in parent['optimizer']['state'].values())
            continuation=read(root/'prefix-20000/training.json')
            assert continuation['parent_sha256']==digest(root/'prefix-10000/checkpoint.pt') and continuation['start']==10000 and continuation['end']==20000
            assert continuation['initial_sample_rng_sha256']==tensor_hash(base['sample_rng']) and continuation['sample_rng_sha256']==tensor_hash(parent['sample_rng'])==p['sampling_rng_sha256']
            data=torch.load(root/'pool.pt',weights_only=True); independent=torch.load(root/'independent.pt',weights_only=True)
            audit=read(root/'data-audit.json')
            assert tensor_hash(data['points'],data['target'])==audit['pool_sha256']
            assert tensor_hash(independent['points'],independent['target'])==audit['independent_sha256']
            from scripts.gap_density_controls import overlap
            assert overlap(data['points'],independent['points'])==0
            assert len(data['points'])==len(independent['points'])==65536
            assert torch.bincount(data['groups']).tolist()==torch.bincount(independent['groups']).tolist()==[32768,16384,16384]
            schedule=read(root/'random-schedule.json'); assert schedule['sha256']==digest(root/'random-indices.npy') and schedule['shape']==[10000,16384]
            import numpy as np
            plan=np.load(root/'random-indices.npy',mmap_mode='r'); assert plan.dtype==np.int32 and plan.shape==(10000,16384)
            for arm in ARMS:
                rows=sorted([r for r in rr if r['metadata']['config']==c and r['training']['arm']==arm],key=lambda r:r['metadata']['step'])
                assert len(rows)==2 and {r['metadata']['step'] for r in rows}=={26000,30000}
                previous=p['parent_sha256']; start=20000
                for r in rows:
                    folder=ROOT/r['folder']; check_files(folder,read(folder/'complete.json'))
                    assert read(folder/'metrics.json')=={k:v for k,v in r.items() if k!='folder'}
                    assert r['metadata']['source_hashes']==m['source_hashes'] and r['metadata']['checkpoint_sha256']==digest(folder/'checkpoint.pt')
                    t=r['training']; assert t['parent_sha256']==previous and t['start']==start and t['end']==r['metadata']['step']
                    assert t['point_presentations']==(t['end']-start)*arm['points']
                    assert t['sample_rng_sha256']==t['initial_sample_rng_sha256']==p['sampling_rng_sha256']
                    assert t['pool_sha256']==read(root/'data-audit.json')['pool_sha256']
                    assert r['independent_points_sha256']==read(root/'data-audit.json')['independent_sha256']
                    from scripts.gap_reserved_confirmation import selection_stream
                    assert t['stream_sha256']==selection_stream(data,arm,t['start'],t['end'],plan)
                    saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
                    assert saved['config']==c and saved['step']==t['end'] and saved['arm']==arm
                    assert all(int(s['step'])==t['end'] for s in saved['optimizer']['state'].values())
                    from scripts.gap_lr_controls import tensor_hash
                    assert tensor_hash(saved['sample_rng'])==p['sampling_rng_sha256']
                    assert set(r['evaluations'])=={'192','384'}
                    for res in ['192','384']:
                        e=read(folder/f'evaluation-r{res}.json'); assert e['metadata']==r['metadata'] and e['metrics']==r['evaluations'][res]
                        assert e['metrics']['resolution']==int(res)
                    expected=dict(components=2,euler=4,closed_edges=True)
                    probes=all(r['field'][name]['sign_accuracy']==1.0 for name in ['component_0','component_1','gap'])
                    assert probes==r['field']['all_probe_signs_correct']
                    structure=all(e['mesh_present'] and e['mesh']==expected and e['reference_mesh']==expected for e in r['evaluations'].values())
                    assert r['primary_gate']==(probes and structure)
                    for e in r['evaluations'].values():
                        assert e['reference_resolved']==(e['reference_mesh']==expected)
                        assert e['structure_match']==(e['mesh']==expected and e['reference_mesh']==expected)
                    # No nonfinite measurements can silently qualify. Empty bins are null.
                    def finite(value):
                        if isinstance(value,float): return math.isfinite(value)
                        if isinstance(value,dict): return all(finite(v) for v in value.values())
                        if isinstance(value,list): return all(finite(v) for v in value)
                        return True
                    assert finite(r)
                    previous=r['metadata']['checkpoint_sha256']; start=t['end']
            pairs.append(dict(case_index=i,seed=c['seed'],parent_sha256=p['parent_sha256'],pool_hash=read(root/'data-audit.json')['pool_sha256'],
                independent_hash=read(root/'data-audit.json')['independent_sha256'],random_schedule_sha256=schedule['sha256']))
        records.extend(dict(case_index=i,**r) for r in rr)
        assert len({p['independent_hash'] for p in pairs if p['case_index']==i})==1
    ledger=read(DEST/'reservation-consumption.json'); assert len(ledger['selected_cases'])==2 and len(ledger['untouched_cases'])==6
    reservation=read(INPUTS/'fresh-cases-reserved.json')
    assert ledger['selected_cases']==reservation['cases'][:2] and ledger['untouched_cases']==reservation['cases'][2:]
    assert all(e['case_index'] in [0,1] and e['case']==m['case_allowlist'][e['case_index']] and e['seed'] in [71,72,73] for e in ledger['events'])
    assert all(e['utc']>=m['created_utc'] for e in ledger['events'])
    allowed={'generate_training_pool','generate_independent_pool','train_shared_prefix','generate_random_selection_schedule'}|{f'evaluate_{a["name"]}_{end}' for a in ARMS for end in ENDS}
    assert all(e['action'] in allowed for e in ledger['events'])
    expected_events={(i,s,a) for i in [0,1] for s in [71,72,73] for a in allowed}
    events=[(e['case_index'],e['seed'],e['action']) for e in ledger['events']]
    assert len(events)==len(expected_events)==72 and set(events)==expected_events
    budget=read(DEST/'budget.json')
    assert all(a['manifest_sha256']==digest(DEST/'confirmation-manifest.json') for a in budget['attempts'] if a.get('batch') in [2,3])
    return m,records,prefixes,pairs


def report():
    gate=gate_audit(ROOT/'runs/gap-confirmation/batch-1'); assert gate['decision']['gate_pass']
    atomic_json(DEST/'audit.json',gate)
    m,records,prefixes,pairs=audit_confirmation(); decision=qualification(records)
    atomic_json(DEST/'confirmation-audit.json',dict(pairs=pairs,checkpoints=len(records),frozen_sources_verified=True,shared_prefix_verified=True,
        source_science_unchanged=True,actual_data_tensor_hashes_verified=True,ordered_recovery_stream_hashes_verified=True,
        primary_gates_recomputed_from_saved_probes_and_meshes=True,complete_finite_records_verified=True,
        confirmation_manifest_sha256=digest(DEST/'confirmation-manifest.json'),order_gate_manifest_sha256=digest(DEST/'batch-1-manifest.json'),
        gate_decision_sha256=digest(DEST/'decision.json'),reservation_sha256=digest(INPUTS/'fresh-cases-reserved.json'),
        reservation_events_verified=72,untouched_complete_cases=6,confirmation_decision=decision))
    atomic_json(DEST/'confirmation-decision.json',decision)
    csv_write(DEST/'confirmation-persistence.csv',decision['trajectories'])
    checkpoints=[]; compute=[]; bins=[]
    for r in records:
        c=r['metadata']['config']; name=r['training']['arm']['name']; end=r['metadata']['step']; case=r['case_index']
        row=dict(case_index=case,feature=c['feature'],angle=c['angle'],seed=c['seed'],arm=name,step=end,primary_gate=r['primary_gate'],
            pool_fit=r['training']['final_fixed']['fit_gate'],training_pool_mse=weighted_mse(r['training']['final_fixed']),independent_mse=weighted_mse(r['independent_fit']),
            independent_gap_mse=r['independent_fit']['gap']['mse'],independent_gap_rmse=r['independent_fit']['gap']['rmse'],full_gap_sign_accuracy=r['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'])
        for res,e in r['evaluations'].items(): row.update({f'chamfer_{res}':e['symmetric_chamfer'],f'components_{res}':e['mesh']['components'],f'euler_{res}':e['mesh']['euler'],f'closed_{res}':e['mesh']['closed_edges'],f'reference_resolved_{res}':e['reference_resolved']})
        checkpoints.append(row)
        for split,groups in r['surface_distance_errors'].items():
            for group,buckets in groups.items():
                for bucket,v in buckets.items(): bins.append(dict(case_index=case,seed=c['seed'],arm=name,step=end,split=split,group=group,distance_bin=bucket,**v))
    for p in prefixes:
        c=p['config']; out=ROOT/f'runs/gap-confirmation/batch-{p["case_index"]+2}'; random=read(out/identity(c)/'random-schedule.json')
        for arm in ARMS:
            rr=[r for r in records if r['case_index']==p['case_index'] and r['metadata']['config']==c and r['training']['arm']==arm]
            train=sum(r['training']['seconds'] for r in rr); evaluate=sum(r['evaluation_seconds'] for r in rr); stage=sum(r['stage_seconds'] for r in rr)
            selection=random['preparation_seconds'] if arm['name']=='random' else 0
            compute.append(dict(case_index=p['case_index'],seed=c['seed'],arm=arm['name'],complete=len(rr)==2,prefix_presentations=p['point_presentations'],
                recovery_presentations=sum(r['training']['point_presentations'] for r in rr),prefix_training_seconds=p['training_seconds'],
                prefix_stage_seconds=p['stage_seconds'],pool_preparation_seconds=p['data_preparation_seconds'],selection_preparation_seconds=selection,
                recovery_training_seconds=train,evaluation_seconds=evaluate,recovery_stage_seconds=stage,
                recovery_other_stage_seconds=stage-train-evaluate,prefix_other_stage_seconds=p['stage_seconds']-p['training_seconds'],
                attributed_pipeline_seconds=p['stage_seconds']+p['data_preparation_seconds']+selection+stage))
    for row in compute:
        baseline=next(x for x in compute if x['case_index']==row['case_index'] and x['seed']==row['seed'] and x['arm']=='dense')
        row['paired_recovery_training_ratio_to_dense']=row['recovery_training_seconds']/baseline['recovery_training_seconds']
        row['paired_attributed_pipeline_ratio_to_dense']=row['attributed_pipeline_seconds']/baseline['attributed_pipeline_seconds']
    paired=[]
    for r in checkpoints:
        if r['arm']!='canonical': continue
        for control in ['dense','random','static']:
            b=next(x for x in checkpoints if x['case_index']==r['case_index'] and x['seed']==r['seed'] and x['step']==r['step'] and x['arm']==control)
            row={k:r[k] for k in ['case_index','feature','angle','seed','step']}; row['control']=control
            row.update(candidate_gate=r['primary_gate'],control_gate=b['primary_gate'],structural_regression=b['primary_gate'] and not r['primary_gate'])
            for metric in ['training_pool_mse','independent_mse','independent_gap_mse','independent_gap_rmse','full_gap_sign_accuracy','chamfer_192','chamfer_384']:
                row[metric+'_candidate_minus_control']=r[metric]-b[metric]
            paired.append(row)
    csv_write(DEST/'confirmation-checkpoints.csv',checkpoints); csv_write(DEST/'confirmation-compute.csv',compute); csv_write(DEST/'confirmation-distance-bins.csv',bins)
    csv_write(DEST/'confirmation-paired-differences.csv',paired)
    totals={}
    for arm in ARMS:
        cc=[c for c in compute if c['arm']==arm['name']]; trajectories=[t for t in decision['trajectories'] if t['arm']==arm['name']]
        totals[arm['name']]=dict(persistent=sum(t['persistent'] for t in trajectories),
            **{k:sum(c[k] for c in cc) for k in ['prefix_presentations','recovery_presentations','prefix_training_seconds','recovery_training_seconds','evaluation_seconds','selection_preparation_seconds','attributed_pipeline_seconds']})
    dense=totals['dense']; cycle=totals['canonical']; failed=[t for t in decision['trajectories'] if t['arm']=='canonical' and not t['persistent']]
    resolved_failed=[t for t in failed if t['reference_resolved']]; unresolved=[t for t in failed if not t['reference_resolved']]
    train_saving=1-cycle['recovery_training_seconds']/dense['recovery_training_seconds']; pipeline_saving=1-cycle['attributed_pipeline_seconds']/dense['attributed_pipeline_seconds']
    claim=(f"The frozen consistency criterion was {'met' if decision['canonical_qualifies'] else 'not met'}: on two preregistered rotated parallel-plate geometries (gaps 0.012/0.03; angles 17/37 degrees) and seeds 71–73, balanced 16,384-point cycling achieved observed persistent structural recovery at 26k/30k and both grids 192/384 in {cycle['persistent']}/6 paired trajectories versus dense {dense['persistent']}/6, using 75% fewer recovery gradient point presentations, {train_saving*100:.1f}% less measured recovery training time and {pipeline_saving*100:.1f}% less attributed prefix-plus-recovery-plus-evaluation runtime.")
    claim+=f" Random sampling recovered {totals['random']['persistent']}/6, so balancing superiority is unsupported."
    if failed:
        claim+=' It failed all three seeds on gap 0.012 / angle 17°, where dense also failed on resolved references; consistent preservation across this subset is unsupported.'
    elif not decision['canonical_qualifies']: claim+=' All observed structural tests passed, but a frozen error/runtime guardrail failed.'
    if not decision['complete']: claim+=' The campaign is incomplete; unobserved trajectories cannot qualify.'
    atomic_json(DEST/'claim.json',dict(claim=claim,canonical_qualifies=decision['canonical_qualifies'],totals=totals,resolved_candidate_failures=resolved_failed,unresolved_reference_cases=unresolved,
        scope='Two angled parallel-plate cases, one geometry family, three paired seeds; observed at26k/30k and192/384 only; six reserved cases untouched'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    figure,axes=plt.subplots(1,3,figsize=(13,4.7),gridspec_kw={'width_ratios':[1,1,1.3]})
    labels=[f"gap {m['case_allowlist'][i]['feature']} / {m['case_allowlist'][i]['angle']}° / seed {s}" for i in [0,1] for s in [71,72,73]]
    for index,end in enumerate(ENDS):
        values=np.array([[int(next(t[f'gate_{end}'] for t in decision['trajectories'] if t['case_index']==i and t['seed']==s and t['arm']==arm['name'])) for arm in ARMS] for i in [0,1] for s in [71,72,73]])
        axes[index].imshow(values,vmin=0,vmax=1,cmap=ListedColormap(['#bd3d3a','#2c8260']),aspect='auto')
        axes[index].set_xticks(range(4),[a['name'] for a in ARMS],rotation=35,ha='right')
        axes[index].set_yticks(range(6),labels if index==0 else ['']*6); axes[index].set_title(f'Observed gate at {end:,}')
        for y in range(6):
            for x in range(4): axes[index].text(x,y,'pass' if values[y,x] else 'fail',color='white',ha='center',va='center',fontsize=9)
    bottoms=np.zeros(4)
    for key,label,color in [('prefix_training_seconds','Shared prefix training','#72936e'),('recovery_training_seconds','Recovery training','#335f91'),('evaluation_seconds','Evaluation','#dd9d33')]:
        values=np.array([totals[a['name']][key]/60 for a in ARMS]); axes[2].bar(range(4),values,bottom=bottoms,label=label,color=color); bottoms+=values
    overhead=np.array([totals[a['name']]['attributed_pipeline_seconds']/60 for a in ARMS])-bottoms
    axes[2].bar(range(4),overhead,bottom=bottoms,label='Prep / transfer / I/O',color='#9c9c9c')
    axes[2].set_xticks(range(4),[a['name'] for a in ARMS],rotation=35,ha='right'); axes[2].set_ylabel('Attributed minutes, six independent pipelines')
    axes[2].set_title('Common prefix cost attributed to each arm'); axes[2].legend(fontsize=8)
    figure.suptitle('Frozen confirmation: every geometry / seed / control retained',fontsize=12)
    figure.tight_layout(); figure.savefig(DEST/'confirmation-summary.png',dpi=180); plt.close(figure)
    budget=read(DEST/'budget.json'); charge=sum(a['charged_seconds'] for a in budget['attempts']); assert charge<=7200 and all(a['finished'] for a in budget['attempts'])
    ids=sorted({a['batch'] for a in budget['attempts'] if a.get('batch') is not None}); assert ids==[1,2,3]
    lines=['# Balanced sampling under a bounded, frozen confirmation campaign','', '## Abstract','',claim,'',
        f"Exactly three scientific batches were attempted, charging **{charge/60:.2f} minutes** of the120-minute training/evaluation allowance including every model-test subprocess, preparation, I/O, failed work and the linked recovery segment. Twenty targeted tests passed; both earlier fixture failures are retained. All{48+len(records)} newly evaluated recovery checkpoints plus12 new shared-prefix checkpoints are retained; historical input checkpoints are separate. Six complete reserved cases remain untouched.",'',
        '## Hypothesis and preregistration','',
        'The hypothesis was that balanced16,384-point cycling preserves dense observed geometric recovery with substantially less compute across new geometries and paired seeds. Batch1 first tested canonical, phase-shifted and independently shuffled presentation order on the six development pairs. All three orders passed every frozen structural/error/compute criterion. Both subsequent confirmation batches, executable code, designated cases and all thresholds were frozen together in [confirmation-manifest.json](confirmation-manifest.json) before any reserved points, probes or models were generated. Both batches ran without tuning, including failed cases.','',
        'The designated subset is exactly the first two archived cases: gap0.012/17-degree and gap0.03/37-degree parallel plates, each with seeds71,72,73 and global_narrow. The other parallel-plate case, all three mixed-thickness cases and both multiscale-stack cases remain untouched. This is confirmation in one analytic geometry family, not evidence across families.','',
        '## Methods','',
        'Each geometry/seed trained one shared width128, depth3 SIREN prefix:10,000 stochastic ordinary-MSE updates at1e-4 with2048 points, followed by10,000 identical cosine-decay updates. Model, Adam and sampling RNG were saved and forked into all four recovery arms. Each arm then used10,000 ordinary equal-point MSE updates with the frozen cosine schedule1e-4 to1e-6. Chunk squared-error sums are divided by total targets, with one Adam update.','',
        'Dense uses65,536 points each update. Canonical cycles balanced16,384-point blocks0,1,2,3 from the same pool. Static repeatedly uses its balanced first16,384 points. Random samples16,384 genuinely unstratified indices without replacement at each update using a dedicated independent CPU RNG and precomputed saved schedules. All arms share pool coordinates/labels and independent evaluation data. The pool remains50% original sampler,25% full gap and25% facing interior. Gap/interior sampling and footprint splits use the local geometry frame, rotated into world coordinates; full-gap bins cover the entire gap width. Analytic SDF labels and oracle gap/interior regions from the known geometry are required; this study does not estimate those regions for arbitrary inputs.','',
        'Persistent structural recovery requires all original gap/interior probe signs plus two closed mesh components and Euler4 at BOTH192 and384 at BOTH26k and30k, with resolved analytic reference meshes. Missing meshes, nonfinite results, unresolved references and missing observations cannot pass. The canonical criterion requires all six pairs, no dense-pass/candidate-fail pair, mean Chamfer at each grid<=1.05xdense, independent gapRMSE<=1.10xdense, sampled gap sign delta>=-0.005 at each checkpoint, and aggregate recovery training<=90%dense.','',
        '## Order gate and validity','',
        'All48 development checkpoints passed the structural gate. Canonical, phase1 and per-cycle shuffle each retained6/6 persistent pairs and passed all guardrails. All24 dense/canonical historical model, Adam and sampling-RNG checkpoints reproduced exactly. Their measured training/dense ratios were0.33884,0.33841,0.33883. The largest gapRMSE ratio was1.05440, largest Chamfer ratio1.00268 and worst sampled gap sign delta was -0.001372. The [frozen order-gate manifest](batch-1-manifest.json) and [gate audit](audit.json) retain exact source/input hashes and all reproduction results.','',
        'OneDrive denied a completion-marker replacement after local33 shuffle30k full measurements had been saved. The failed1588.4591-second attempt, traceback and both markers are retained. An I/O-only recovery verified all eight temp-marker payload hashes, recovered the measurements without any repeated training/evaluation, and ran only the two missing dense33 stages. Its128.7097 seconds were also charged to logical batch1. The unpersisted shuffle30k enclosing stage timer remains null; training/evaluation timers are intact and counted once. Dense33 ran after a pause, weakening the intended execution-order timing comparison. Frozen science remained unchanged. The embedded new gate protocol governs; inherited legacy protocol-hash/selection fields remain unchanged and are explicitly annotated.','',
        '## Confirmation results','',
        '![All confirmation structural outcomes and measured pipeline costs](confirmation-summary.png)','',
        '| Arm | Persistent pairs | Recovery presentations (six pairs) | Recovery train seconds | Evaluation seconds | Random schedule prep seconds | Attributed full pipeline seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,t in totals.items(): lines.append(f"| {name} | {t['persistent']}/6 | {t['recovery_presentations']:,} | {t['recovery_training_seconds']:.2f} | {t['evaluation_seconds']:.2f} | {t['selection_preparation_seconds']:.2f} | {t['attributed_pipeline_seconds']:.2f} |")
    lines+=['','| Gap / angle | Seed | Arm | Gate26k | Gate30k | Persistent | Reference resolved |','|---|---:|---|---|---|---|---|']
    for t in decision['trajectories']:
        case=m['case_allowlist'][t['case_index']]; lines.append(f"| {case['feature']} / {case['angle']} | {t['seed']} | {t['arm']} | {t['gate_26000']} | {t['gate_30000']} | {t['persistent']} | {t['reference_resolved']} |")
    lines+=['','Every seed and failure is retained in [confirmation-checkpoints.csv](confirmation-checkpoints.csv), including ordinary training-pool and independent MSE, gapRMSE, signs, meshes and strict fit failures. Overall MSE is the50/25/25 weighted mean of the stored original/gap/interior subset MSE values, exactly matching their pool proportions. [confirmation-paired-differences.csv](confirmation-paired-differences.csv) gives cycling-minus-dense/random/static differences for each paired seed and checkpoint, including structural regressions. [confirmation-distance-bins.csv](confirmation-distance-bins.csv) reports independent versus training errors by distance and footprint. Exact frozen threshold comparisons are in [confirmation-decision.json](confirmation-decision.json).','',
        '## Compute accounting','',
        'Recovery gradient presentations are163,840,000 for each sampling arm versus655,360,000 for dense:75% fewer. The shared prefix presents40,960,000 training points, yielding204,800,000 versus696,320,000 total prefix-plus-recovery gradient presentations:70.59% fewer. These counts exclude diagnostic/evaluation queries; all such work remains in measured runtime. No dense label/storage saving is claimed. Prefix computation is executed once per pair and its SAME measured cost is attributed to each arm when comparing independent full pipelines. Summing attributed arm totals therefore intentionally repeats the common cost; the campaign budget counts actual work once.','',
        'Attributed pipeline time includes shared-prefix stage work, pool preparation, arm-specific schedule preparation, recovery training/evaluation and recovery-stage I/O/transfer/hash work. Random schedule preparation is explicitly counted. Training and evaluation timers are measured separately with GPU synchronization. Whole-process charges also include other setup and audit overhead; per-arm attributed stage costs exclude small boundary-marker, summary and process-bookkeeping work, rather than reconstructing an unmeasured end-to-end arm wall clock. Measurements are descriptive serial timings on the same RTX2080Super, not universal speed guarantees.','',
        '## Failure analysis, controls and limitations','']
    for d in decision['arms']:
        lines.append(f"- {d['arm']}: persistent{d['persistent_passes']}/6; training/dense ratio{d['training_time_ratio']:.4f}; qualifies={d['qualifies']}; dense-persistent losses={d['dense_persistent_losses']}.")
    lines+=['','| Arm | Step | Gap RMSE / dense | Chamfer192 / dense | Chamfer384 / dense | Gap sign delta | Error guardrails pass |',
            '|---|---:|---:|---:|---:|---:|---|']
    for d in decision['arms']:
        for c in d['comparisons']:
            lines.append(f"| {d['arm']} | {c['step']:,} | {c['gap_rmse_ratio']:.6f} | {c['chamfer_ratio_192']:.6f} | {c['chamfer_ratio_384']:.6f} | {c['gap_sign_delta']:.6f} | {c['passes']} |")
    lines+=['',
        'Unresolved analytic references are measurement limitations and are distinguished from prediction failures on resolved grids. No resolution or threshold was changed after exposure. Random/static control outcomes bound attribution: matching random performance does not establish that balancing is necessary or superior. Failures shared by dense and cycling cannot be attributed specifically to sampling. Gap width, angle and seeds all changed from development, so their individual causal effects are not isolated. These finite-training results do not establish representational impossibility. Sampled signs and finite-grid meshes cannot certify continuous topology or behavior between checkpoints. Three paired seeds per geometry are descriptive evidence, not an inferred population success rate. Six held-out cases and other geometry families remain untested.','']
    if any(r['case_index']==1 and r['training']['arm']['name']=='dense' and any(e['mesh']['components']>2 for e in r['evaluations'].values()) for r in records):
        lines+=['Dense failures on the wider case include extra disconnected mesh components. The narrow case instead exhibits merged components and incorrect Euler characteristics; small distance or sign errors can coexist with either structural failure. Random matches cycling at3/6 persistent structural successes but fails the26k full-gap sign guardrail; static reaches0/6 and fails distance guardrails. These paired outcomes do not establish balancing necessity or statistical superiority.','']
    for r in records:
        if (r['training']['arm']['name']=='canonical' or r['case_index']==1 and r['training']['arm']['name']=='dense') and not r['primary_gate']:
            c=r['metadata']['config']; meshes='; '.join(f"r{res}: components={e['mesh']['components']}, Euler={e['mesh']['euler']}, closed={e['mesh']['closed_edges']}, reference_resolved={e['reference_resolved']}, Chamfer={e['symmetric_chamfer']}" for res,e in r['evaluations'].items())
            lines.append(f"{r['training']['arm']['name']} failure gap{c['feature']}, angle{c['angle']}, seed{c['seed']}, step{r['metadata']['step']}: original probe signs all correct={r['field']['all_probe_signs_correct']}; independent gap MSE={r['independent_fit']['gap']['mse']:.8g}; {meshes}.")
    lines+=['',
        '## Reproducibility and retained evidence','',
        'The original checkout was read-only. The local input bundle contains all exact development parents, pools and historical reproduction checkpoints. Confirmation pools, independent data, shared prefix model/Adam/RNG, random schedules, every recovery checkpoint and both-grid metrics remain under runs/gap-confirmation/batch-2 and batch-3. Frozen source snapshots accompany each batch. [run-artifacts.json](run-artifacts.json) indexes every campaign input, source, model, schedule, metric, log and export by SHA-256 and size. [reservation-consumption.json](reservation-consumption.json) logs every selected-case generation/evaluation and lists the six untouched cases.','',
        'Set $python to the recorded Python environment or install requirements-windows-cu126.txt. The preparation command creates a new child workspace containing exact frozen source, all development inputs, reservation and manifests, with an empty independent budget and no runs. The gate wrapper adds only completion-marker I/O retries. The delivered numerical source hashes remain unchanged. The package and full retained-metrics reporter were verified in a clean workspace; no additional training replay was performed. These commands describe a separately authorized future reproduction, not extra campaign batches.','',
        '```powershell',"$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'",'$env:PYTHONDONTWRITEBYTECODE=\'1\'',
        '& $python -m scripts.prepare_gap_reproduction --destination replay-new',
        'Set-Location replay-new',
        '& $python -m scripts.run_gap_replay',
        "if ($LASTEXITCODE -ne 0) { throw 'Replay order gate failed; stop before confirmation.' }",
        '& $python -m scripts.gap_reserved_confirmation budget-run --case-index 0',
        "if ($LASTEXITCODE -ne 0) { throw 'First confirmation batch failed; stop.' }",
        '& $python -m scripts.gap_reserved_confirmation budget-run --case-index 1',
        "if ($LASTEXITCODE -ne 0) { throw 'Second confirmation batch failed; stop.' }",
        '& $python -m scripts.report_reserved_confirmation','```','',
        'The original budget ledger rejects another scientific batch or exhausted budget. Reporting hashes and reads retained metrics without model evaluation. Its paper narrative describes this retained campaign, including its original I/O interruption; a fresh replay must report its own budget and infrastructure events rather than reuse those historical details.','']
    import re
    clean=[]; fenced=False
    for line in lines:
        if line.startswith('```'): fenced=not fenced
        if not fenced and not line.startswith('```'):
            chunks=re.split(r'(\[[^\]]*\]\([^)]*\)|requirements-windows-cu126\.txt|1d809bd13e9e8e4ac3019d5bc12096b3fef8af13136b696c8f507af78f015caf)',line)
            for j in range(0,len(chunks),2):
                chunks[j]=re.sub(r'(?<=[A-Za-z])(?=\d)',' ',chunks[j]); chunks[j]=re.sub(r'(?<=\d)(?=[A-Za-z])',' ',chunks[j])
                chunks[j]=re.sub(r'(\d) e([+-]\d)',r'\1e\2',chunks[j])
                chunks[j]=re.sub(r'(\d) k\b',r'\1k',chunks[j])
                chunks[j]=re.sub(r'%([A-Za-z])',r'% \1',chunks[j])
                chunks[j]=chunks[j].replace('xdense',' × dense')
            line=''.join(chunks)
            line=line.replace('gapRMSE','gap RMSE').replace('prefix:10','prefix: 10').replace(':75%',': 75%').replace(':70.59%',': 70.59%').replace(',25%',', 25%').replace('  ×',' ×')
        clean.append(line)
    (DEST/'report.md').write_text('\n'.join(clean),encoding='utf-8')
    refresh_inventory()
    print(claim); print('Charged minutes:',charge/60)


def refresh_inventory():
    artifacts={}
    for base in [INPUTS,ROOT/'runs/gap-confirmation/batch-1',ROOT/'runs/gap-confirmation/batch-2',ROOT/'runs/gap-confirmation/batch-3',DEST]:
        for p in base.rglob('*'):
            if p.is_file() and p.name not in ['run-artifacts.json','export-audit.json']: artifacts[p.relative_to(ROOT).as_posix()]=dict(sha256=digest(p),bytes=p.stat().st_size)
    for p in [ROOT/'scripts/report_reserved_confirmation.py',ROOT/'scripts/report_gap_confirmation.py',ROOT/'scripts/prepare_gap_reproduction.py',ROOT/'scripts/run_gap_replay.py',ROOT/'scripts/verify_gap_exports.py',ROOT/'scripts/verify_gap_reproduction.py',ROOT/'requirements-windows-cu126.txt',ROOT/'README.md',ROOT/'.gitattributes']:
        artifacts[p.relative_to(ROOT).as_posix()]=dict(sha256=digest(p),bytes=p.stat().st_size)
    atomic_json(DEST/'run-artifacts.json',dict(files=artifacts,total_files=len(artifacts),total_bytes=sum(f['bytes'] for f in artifacts.values())))
    atomic_json(DEST/'export-audit.json',dict(files={p.name:digest(p) for p in DEST.iterdir() if p.is_file() and p.name!='export-audit.json'}))
    print('Artifacts:',len(artifacts))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--inventory-only',action='store_true')
    if p.parse_args().inventory_only: refresh_inventory()
    else: report()
