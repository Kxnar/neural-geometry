"""Compile the audited, capped sampling campaign without more model evaluation."""
import csv
import shutil
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neural_geometry_scale.study import ROOT, read, atomic_json, digest


def main():
    dest=ROOT/'docs/gap-sampling'; runroot=ROOT/'runs/gap-sampling'
    ledger=read(runroot/'budget.json')
    assert all(a['finished'] and a['exit_code']==0 for a in ledger['attempts'])
    assert len(ledger['attempts'])<=3
    charged=sum(a['charged_seconds'] for a in ledger['attempts'])+read(dest/'validation.json')['pytest_seconds']
    assert charged<7200
    all_records=[]; decisions=[]; methods=[]; source_artifacts={}
    for attempt in ledger['attempts']:
        batch=attempt['batch']; folder=dest/f'batch-{batch}'
        export=read(folder/'export-audit.json')
        for name,sha in export['files'].items(): assert digest(folder/name)==sha
        audit=read(folder/'audit.json'); m=read(folder/'manifest.json')
        assert audit['reservation_unchanged'] and audit['checkpoints_verified']==18*len(m['protocol']['arms'])
        rr=read(folder/'summary.json')['runs']; all_records.extend(rr)
        dd=read(folder/'decision.json')['arms']; decisions.extend(dict(batch=batch,**d) for d in dd)
        tt=read(folder/'persistence.json')['trajectories']
        source_artifacts[str(folder.relative_to(ROOT))]=digest(folder/'export-audit.json')
        for arm in m['protocol']['arms']:
            r=[x for x in rr if x['training']['arm']==arm]
            t=[x for x in tt if x['arm']==arm['name']]
            last=[x for x in r if x['metadata']['step']==30000]
            methods.append(dict(batch=batch,name=arm['name'],batch_points=arm['points'],phase=arm['phase'],
                persistent=sum(x['persistent'] for x in t),regressions=sum(x['regression'] for x in t),
                passing_checkpoints=sum(x['primary_gate'] for x in r),checkpoints=len(r),
                presentations_per_trajectory=arm['points']*10000,
                training_seconds=sum(x['training']['seconds'] for x in r),evaluation_seconds=sum(x['evaluation_seconds'] for x in r),
                final_gap_rmse=statistics.mean(x['independent_fit']['gap']['rmse'] for x in last),
                final_gap_signs=statistics.mean(x['full_gap']['regions']['whole_gap']['all_margins']['sign_accuracy'] for x in last),
                final_pool_fit=sum(x['training']['final_fixed']['fit_gate'] for x in last),
                final_chamfer_192=statistics.mean(x['evaluations']['192']['symmetric_chamfer'] for x in last),
                final_chamfer_384=statistics.mean(x['evaluations']['384']['symmetric_chamfer'] for x in last)))
    dense=next(x for x in methods if x['name']=='dense')
    for x in methods:
        x['presentation_ratio']=x['presentations_per_trajectory']/dense['presentations_per_trajectory']
        x['training_time_ratio']=x['training_seconds']/dense['training_seconds']
        x['training_and_evaluation_ratio']=(x['training_seconds']+x['evaluation_seconds'])/(dense['training_seconds']+dense['evaluation_seconds'])
    qualified=[x for x in methods if any(d['arm']==x['name'] and d['qualifies'] for d in decisions)]
    best=min(qualified,key=lambda x:(x['batch_points'],x['phase'])) if qualified else dense
    result=dict(completed_batches=len(ledger['attempts']),charged_seconds_including_tests=charged,
        training_seconds=sum(x['training_seconds'] for x in methods),evaluation_seconds=sum(x['evaluation_seconds'] for x in methods),
        training_presentations=sum(x['presentations_per_trajectory']*6 for x in methods),
        methods=methods,decisions=decisions,best_qualified_method=best['name'],
        total_checkpoints=len(all_records),missing_meshes=sum(not e['mesh_present'] for r in all_records for e in r['evaluations'].values()),
        source_export_hashes=source_artifacts,fresh_cases_evaluated=False,additional_batches_permitted=0)
    atomic_json(dest/'campaign-summary.json',result);shutil.copyfile(runroot/'budget.json',dest/'budget.json')
    atomic_json(dest/'run-artifacts.json',dict(root=str(runroot),
        files={p.relative_to(runroot).as_posix():digest(p) for p in runroot.rglob('*') if p.is_file()}))
    shutil.copyfile(ROOT/'docs/gap-controls/fresh-cases.json',dest/'fresh-cases-reserved.json')
    with (dest/'methods.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(methods[0]));writer.writeheader();writer.writerows(methods)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    labels=[x['name'].replace('cycle-','Cycle ').replace('dense','Dense') for x in methods]
    colors=['#64748b']+['#167d8d']*(len(methods)-1)
    axes[0].bar(labels,[x['persistent'] for x in methods],color=colors)
    axes[0].set_ylim(0,6.6);axes[0].set_ylabel('Persistent passes / 6 paired trajectories')
    axes[0].set_title('Gate at both 26k and 30k, at 192 and 384')
    axes[1].bar(labels,[x['training_time_ratio'] for x in methods],color=colors)
    axes[1].set_ylabel('Measured training time / dense control')
    axes[1].set_title('10,000 additional ordinary-MSE updates')
    for ax in axes:
        ax.tick_params(axis='x',rotation=20);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(dest/'sampling-summary.png',dpi=180);plt.close(fig)
    lines=['# Can sampling retain dense-data structural recovery with less compute?','',
        f"The best qualifying tested method is **{best['name']}**. It retained observed persistent structure in **{best['persistent']}/6** paired development trajectories, using **{100*(1-best['presentation_ratio']):.2f}% fewer point presentations** and **{100*(1-best['training_time_ratio']):.1f}% less measured training time** than the rerun dense control. This compares the additional 10,000-update recovery stage from shared 20,000-update checkpoints.",'',
        f"Completed **{len(ledger['attempts'])} new batches**, **{len(all_records)} checkpoints** and **{2*len(all_records)} grid evaluations**. Conservative run wall time plus the verification suite was **{charged/60:.2f} minutes**, below the two-hour allowance. No additional batch will be run under this goal. Fresh cases remain reserved.",'',
        '![Sampling results](sampling-summary.png)','',
        '| Method | Batch | Persistent | Presentations per trajectory | Training seconds, six trajectories | Evaluation seconds | Final independent gap RMSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for x in methods:lines.append(f"| {x['name']} | {x['batch']} | {x['persistent']}/6 | {x['presentations_per_trajectory']:,} | {x['training_seconds']:.2f} | {x['evaluation_seconds']:.2f} | {x['final_gap_rmse']:.6g} |")
    lines+=['','## Protocol and controls','',
        'One existing development geometry: axis-aligned parallel plates with a 0.02 gap, width-128 depth-3 SIREN, original samplers global_narrow and local, paired seeds 31–33. Every trajectory restores its exact 20k model and Adam state. The objective is ordinary equal-point MSE, with the same endpoint-inclusive cosine schedule from 1e-4 to 1e-6 for 10k more updates. Adam is not restarted at evaluation checkpoints.','',
        'The saved 65,536-point pool is identical to the dense baseline. A cycling update presents one contiguous balanced block: 50% original-sampler points, 25% full-gap points and 25% facing-interior points. Blocks are disjoint and exhaustive over a cycle; no residuals or evaluation probes select training points. The original sampling RNG is preserved. Cycling changes both exposure order and gradient noise, so the experiment does not isolate them. All dense labels and storage remain necessary.','',
        'The unchanged structural gate requires all original probe signs, two mesh components, Euler characteristic four, closed edges and a resolved analytic reference at both 192 and 384. Persistent success requires passes at both 26k and 30k. Guardrails, frozen before batch 1 and kept throughout, require mean Chamfer at each grid within 5% of dense, independent gap RMSE within 10%, and full-gap sign accuracy no more than 0.5 percentage points worse, at both later checkpoints. Qualification also requires all six persistent passes, fewer presentations and at least 10% lower aggregate training time. These are descriptive tolerances, not statistical noninferiority tests.','',
        'The independent 65,536-point evaluation sets and 131,072 requested full-gap probes are unchanged. Exact-coordinate training/evaluation overlap is zero. Pool fit uses all dense points and retains the earlier strict subset-RMSE and strong-margin-sign criterion.','',
        '## Sequential decisions and retained negatives','']
    for attempt in ledger['attempts']:
        batch=attempt['batch'];m=read(dest/f'batch-{batch}/manifest.json')
        lines += [f"### Batch {batch}",'','**Preregistered hypothesis:** '+m['protocol']['hypothesis'],'',f"[Preregistered controls and criteria](batch-{batch}/manifest.json) · [All results](batch-{batch}/report.md) · [Audit](batch-{batch}/audit.json)",'']
        for d in [d for d in decisions if d['batch']==batch]:
            lines.append(f"{d['arm']}: {d['persistent_passes']}/6 persistent; qualifies={d['qualifies']}. Presentation ratio {d['presentation_ratio']:.4f}; measured training-time ratio {d['training_time_ratio']:.4f}. Both later-checkpoint guardrail measurements are retained in [decision.json](batch-{batch}/decision.json).")
            lines.append('')
            failed=[c for c in d['comparisons'] if not c['guards_pass']]
            if failed:
                lines.append('Guardrail failures: '+'; '.join(f"at {c['step']}, gap RMSE ratio {c['gap_rmse_ratio']:.4f}, gap-sign change {100*c['gap_sign_delta']:+.3f} percentage points, Chamfer ratios {c['chamfer_ratio_192']:.4f}/{c['chamfer_ratio_384']:.4f}" for c in failed)+'.')
                lines.append('')
        if batch==2:
            lines.append('The 4096-point negative result is retained in full. Local seed 33 has one mesh component at resolution 192 at 26k and recovers at 30k. All six final checkpoints pass, but only five trajectories satisfy persistence; independent gap RMSE also exceeds the frozen 10% tolerance at both later checkpoints. Early structural passes are 1/6 at 23k. No fourth batch or phase-shift run was substituted for this failure.')
        lines+=['']
    lines+=['| Method | Passing checkpoints | Passing-to-failing trajectories | Final pool-fit passes | Final full-gap signs | Final Chamfer 192 / 384 |','|---|---:|---:|---:|---:|---:|']
    for x in methods:lines.append(f"| {x['name']} | {x['passing_checkpoints']}/{x['checkpoints']} | {x['regressions']} | {x['final_pool_fit']}/6 | {x['final_gap_signs']:.3%} | {x['final_chamfer_192']:.6f} / {x['final_chamfer_384']:.6f} |")
    lines+=['',
        'The earlier fixed-data controls remain negative context: fixed 4,096 points yielded 0/6 persistent passes and fixed 16,384 points yielded 2/6; dense full-data training yielded 6/6. Cycling retains dense coverage over time while reducing points per update. Those historical comparisons change the pool and presentation schedule together; they cannot identify coverage as the sole cause.','',
        '## Compute and evidence audit','',
        f"Measured training: {result['training_seconds']/60:.2f} minutes; measured evaluation: {result['evaluation_seconds']/60:.2f} minutes; total training presentations: {result['training_presentations']:,}. The conservative {charged/60:.2f}-minute charge additionally includes subprocess startup, data preparation, hashing, checkpoint I/O and the 10.23-second verification suite. All three batch processes completed normally; no interrupted or discarded research attempts are hidden. Pure report generation and hash audits perform no further model training or field/mesh evaluation.",'',
        f"For {best['name']}, combined measured training plus evaluation is {best['training_and_evaluation_ratio']:.3f} times the dense control. Point-presentation savings are not wall-clock speedups. Timings are one serial workload on an RTX 2080 Super; later batches reuse batch-1 timings, and execution order, thermal state and load can affect them. Batch 1 alternates arm order across pairs. The common 20k prefix is excluded from all savings, and there are no label-count or storage savings.",'',
        'All 66 tests passed before the first run. The tests independently check ordinary-MSE updates, complete balanced cycles and exact Adam/model continuation across a cycle boundary. The batch-1 dense rerun reproduces historical model and Adam tensors and geometry metrics exactly at all 18 checkpoints. Audits regenerate pool coordinates, verify independent-set disjointness, stream hashes, parent chains, Adam steps, source snapshots and the fresh-case reservation hash.','',
        f"Missing meshes: {result['missing_meshes']}/{2*len(all_records)}. Every missing mesh is a structural failure. Full model/Adam/RNG checkpoints, pool and independent tensors, logs and frozen sources are retained under `{runroot}`. Compact exports and SHA-256 inventories are in this report directory.",'',
        '## Limits and next research decision','',
        'This is adaptively selected development evidence on one analytic geometry, one architecture and three seeds per original sampler. Reusing the same seeds across batches does not provide independent confirmation. The gate is a finite-resolution diagnostic; it does not certify continuous topology, all surface signs, or correctness between checkpoints. Pool-fit failures and independent near-surface errors remain. The unsquared sampled symmetric Chamfer is half the two directed means over 32,768 surface samples, not exact surface distance.','',
        f"Freeze {best['name']} as the leading tested candidate. The next research decision is an order-robustness check on the existing development geometry: predeclare shifted and shuffled balanced-block schedules against the same dense control, with the same gates and compute accounting. Block order and phase were not varied in these three batches. If that check supports the candidate, freeze a separate held-out confirmation study before consuming any reserved cases; do not tune after viewing them. These are proposed follow-ups, not work performed in this capped goal.",'',
        '## Reproduction','',
        'Use the original neural-geometry checkout and its existing `.venv`. Each batch report gives exact commands using a new output directory. Source snapshots, environment versions, parent checkpoint hashes, dataset hashes, protocols and every checkpoint measurement are retained. Reproduction requires the parent checkpoints and historical dense artifacts in that checkout; the compact export alone does not contain their binary tensors.','',
        '```powershell',f"Set-Location '{ROOT}'",'.venv/Scripts/python.exe -m scripts.report_gap_sampling_campaign','```','',
        'The command above rebuilds this synthesis from audited exports without training. See [campaign-summary.json](campaign-summary.json), [methods.csv](methods.csv), [budget.json](budget.json), the [complete binary and log artifact inventory](run-artifacts.json), and each batch report for the complete machine-readable evidence.','']
    (dest/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    source=dest/'report-source';source.mkdir(exist_ok=True)
    shutil.copyfile(__file__,source/Path(__file__).name)
    atomic_json(dest/'campaign-export-audit.json',dict(files={p.relative_to(dest).as_posix():digest(p) for p in dest.rglob('*') if p.is_file() and p.name!='campaign-export-audit.json'}))
    print(result)


if __name__=='__main__':main()
