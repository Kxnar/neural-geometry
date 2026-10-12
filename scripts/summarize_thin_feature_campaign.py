"""Output analysis only: read retained artifacts, audit hashes, and plot results.

No geometry/model/training module is imported or executed.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/thin-feature-campaign'
RUNS=ROOT/'runs/thin-feature-campaign'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    analysis=read(DEST/'analysis.json');budget=read(DEST/'budget.json');selection=read(DEST/'selection.json')
    confirmation=read(DEST/'confirmation-manifest.json');development=read(DEST/'development-manifest.json')
    tasks={t['id']:t for t in development['tasks']}
    stages={};inventory=[];source_checks=[];references={};completed={};random_plans=[];dataset_pairs={}
    for stage in ['development','confirmation']:
        stagepath=RUNS/stage;manifest=development if stage=='development' else confirmation
        completed[stage]=[]
        for folder in sorted(stagepath.iterdir()) if stagepath.exists() else []:
            if not folder.is_dir():continue
            if (folder/'complete.json').exists():
                result=read(folder/'complete.json');completed[stage].append(result)
                if stage=='confirmation':
                    t=result['task'];data_meta=read(folder/'data.json')
                    dataset_pairs.setdefault((t['case_index'],t['seed']),[]).append(dict(arm=t['arm'],hashes=data_meta['hashes']))
            for name,expected in manifest['sources'].items():
                path=folder/'source'/name
                if path.exists():
                    if sha(path)!=expected:raise RuntimeError(f'Source snapshot drift: {path}')
                    source_checks.append(path.relative_to(ROOT).as_posix())
            for path in sorted(folder.rglob('*')):
                if path.is_file():inventory.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
            plan_path=folder/'random-indices.npy'
            if stage=='confirmation' and plan_path.exists():
                plan=np.load(plan_path,mmap_mode='r');meta=read(folder/'random-schedule.json')
                task=next(t for t in confirmation['tasks'] if t['id']==folder.name)
                if plan.shape!=(10000,16384) or plan.dtype!=np.int32:raise RuntimeError('Saved random schedule shape/dtype mismatch')
                sampled=[0,4999,9999]
                if any(len(np.unique(plan[j]))!=16384 or plan[j].min()<0 or plan[j].max()>=65536 for j in sampled):raise RuntimeError('Saved random selection invalid')
                expected_seed=800000+task['case_index']*100+task['seed']
                saved_hash=next(v['sha256'] for v in inventory if v['path']==plan_path.relative_to(ROOT).as_posix())
                if meta['seed']!=expected_seed or meta['sha256']!=saved_hash:raise RuntimeError('Random schedule seed/hash mismatch')
                random_plans.append(dict(task=folder.name,shape=list(plan.shape),dtype=str(plan.dtype),seed=expected_seed,
                                         sha256=saved_hash,sampled_rows_unique_and_in_range=sampled))
            for path in sorted(folder.glob('metrics-*.json')):
                checkpoint=read(path)
                for res,metric in checkpoint['evaluations'].items():
                    key=f'{stage}:r{res}';references.setdefault(key,dict(total=0,resolved=0,unresolved=0))
                    references[key]['total']+=1;references[key]['resolved']+=int(metric['reference_resolved']);references[key]['unresolved']+=int(not metric['reference_resolved'])
        own=[a for a in budget['attempts'] if a['stage']==stage]
        stages[stage]=dict(attempts=len(own),completed=len(completed[stage]),failed_attempts=sum(a['exit_code']!=0 for a in own),
                           timed_out=sum(a.get('timed_out',False) for a in own),charged_seconds=sum(a['charged_seconds'] for a in own))
    qa=[a for a in budget['attempts'] if a['stage']=='qa']
    stages['qa']=dict(attempts=len(qa),failed_attempts=sum(a['exit_code']!=0 for a in qa),charged_seconds=sum(a['charged_seconds'] for a in qa))
    if any(not a['finished'] for a in budget['attempts']):raise RuntimeError('Unclosed external attempt')
    for stage in ['development','confirmation','qa']:
        if stages[stage]['charged_seconds']>budget['limits'][stage]:raise RuntimeError(f'Stage budget exceeded: {stage}')
    if sum(a['charged_seconds'] for a in budget['attempts'])>budget['limits']['global_seconds']:raise RuntimeError('Global budget exceeded')
    narrow=[r for r in completed['development'] if r['task']['case']['feature']==.012 and r['task']['seed']==71]
    fixed_narrow=dict(completed=len(narrow),expected=8,persistent=sum(r['persistence']['persistent'] for r in narrow),
                      scope='Two fixed models; two angles; dense/cycling; seed71. No architectural impossibility inference.')
    retry_events=[]
    for attempt in budget['attempts']:
        log=ROOT/attempt['log']
        if log.exists():
            for line in log.read_text(encoding='utf-8').splitlines():
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                if isinstance(event,dict) and 'io_retry' in event:retry_events.append(dict(external_attempt=attempt['number'],**event))
    supplemental=dict(stages=stages,development_selection_inconclusive=selection['selection_inconclusive'],
                      fixed_narrow_development=fixed_narrow,reference_outcomes=references,io_retries=retry_events,
                      source_snapshots_verified=len(source_checks),random_schedule_audits=random_plans,all_attempt_artifact_count=len(inventory),
                      total_charged_seconds=sum(a['charged_seconds'] for a in budget['attempts']))
    pair_audits=[dict(case_index=i,seed=s,arms=[v['arm'] for v in values],exact_dataset_bytes_match=all(v['hashes']==values[0]['hashes'] for v in values)) for (i,s),values in sorted(dataset_pairs.items())]
    if any(not value['exact_dataset_bytes_match'] for value in pair_audits):raise RuntimeError('Independent arm dataset pairing mismatch')
    supplemental['dataset_pairing_audits']=pair_audits
    supplemental['all_18_dataset_triples_paired']=len(pair_audits)==18 and all(len(value['arms'])==3 for value in pair_audits)
    supplemental['evidence_hashes']={name:sha(DEST/name) for name in ['protocol.json','development-manifest.json',
        'development-manifest.json.sha256','confirmation-manifest.json','confirmation-manifest.json.sha256',
        'selection.json','budget.json','reservation-events.jsonl','audit.json','analysis.json']}
    supplemental['analysis_script_sha256']=sha(Path(__file__))
    write(DEST/'supplement.json',supplemental)
    write(DEST/'artifact-inventory.json',dict(files=inventory,source_snapshots_checked=source_checks))
    lines=['# Campaign evidence and execution audit','',
           f'Development completed {stages["development"]["completed"]}/16 screen trajectories. Selection is {"inconclusive; predeclared SIREN fallback" if selection["selection_inconclusive"] else "complete under the frozen ranking"}. ',
           f'The fixed narrow screen independently completed {fixed_narrow["completed"]}/8 trajectories with {fixed_narrow["persistent"]} persistent successes. This observed narrow failure is distinct from an incomplete screen, a file-access failure, or an architectural impossibility claim.','',
           '| Stage | External attempts | Failed attempts | Timeouts | Charged minutes |','|---|---:|---:|---:|---:|']
    for stage,value in stages.items():lines.append(f'| {stage} | {value["attempts"]} | {value["failed_attempts"]} | {value.get("timed_out",0)} | {value["charged_seconds"]/60:.3f} |')
    lines+=['', 'The development Windows file-access failure is retained with consumed time and partial files; it is not classified as a measured reconstruction gate failure or replaced. Confirmation uses the separately frozen I/O-only retry wrapper. Earlier failed QA attempts remain charged, followed by passing repairs.',
            '', '| Retained reference evaluation | Total | Resolved | Unresolved |','|---|---:|---:|---:|']
    for key,value in references.items():lines.append(f'| {key} | {value["total"]} | {value["resolved"]} | {value["unresolved"]} |')
    lines+=['',f'All {len(inventory)} retained run files, including partial attempts and source snapshots, are hashed in artifact-inventory.json. {len(source_checks)} source snapshots match their exact stage inventories. Retry events are exported in supplement.json. Stage and global hard budgets were checked from every durable finished attempt.',
            '', 'The plot reads retained results only. No model is evaluated or trained to make it.']
    lines+=['', '## Geometric dimension conventions', '',
            '| Reserved family/cases | Reported feature parameter | Physical dimensions |',
            '|---|---|---|',
            '| Parallel plates, case 2 | 0.06 | Full gap 0.06 |',
            '| Mixed thickness, cases 3/4/5 | 0.015 / 0.03 / 0.06 half-thickness | Full thin-component thickness 0.03 / 0.06 / 0.12; these values are not gaps |',
            '| Multiscale stacks, cases 6/7 | Full gaps 0.012 and 0.05 | Component half-thicknesses 0.025 / 0.07 / 0.12; full thicknesses 0.05 / 0.14 / 0.24 |',
            '', 'Development narrow/wide parameters 0.012/0.03 are full parallel-plate gaps. All dimensions remain the immutable reservation values.']
    (DEST/'supplement.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    arms=['dense','random','cycling'];colors=['#3D5A80','#E09F3E','#2A9D8F']
    families=['parallel_plates','mixed_thickness','multiscale_plate_stack']
    denominators={family:sum(t['arm']=='dense' and t['case']['scene']==family for t in confirmation['tasks']) for family in families}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.4));x=np.arange(3);width=.24
    for index,(arm,color) in enumerate(zip(arms,colors)):
        values=[analysis['family_successes'][family][arm] for family in families]
        bars=axes[0].bar(x+(index-1)*width,values,width,label=arm,color=color)
        axes[0].bar_label(bars,padding=3)
    axes[0].set_xticks(x,['Parallel plates\n3 pairs','Mixed thickness\n9 pairs','Narrow stacks\n6 pairs'])
    axes[0].set_ylabel('Persistent successes');axes[0].set_ylim(0,10);axes[0].set_yticks(range(0,10))
    axes[0].set_title('Reserved reconstruction reliability');axes[0].legend(frameon=False)
    times=[analysis['full_pipeline_seconds'][a]/60 for a in arms]
    bars=axes[1].bar(arms,times,color=colors);axes[1].bar_label(bars,labels=[f'{v:.1f} min' for v in times],padding=4)
    axes[1].set_ylabel('Full pipeline minutes, all attempts');axes[1].set_ylim(0,max(times)*1.23 if max(times)>0 else 1)
    axes[1].set_title('Measured full pipeline\nIndependent prefixes, preparation, training, evaluation, I/O')
    target='achieved' if analysis['target_achieved'] else 'not achieved'
    fig.suptitle(f'Thin-feature campaign — target {target}',fontsize=16,fontweight='bold',y=.98)
    fig.text(.5,.88,f'Fixed narrow development: {fixed_narrow["persistent"]}/{fixed_narrow["expected"]} trajectories persistent; confirmation {stages["confirmation"]["completed"]}/54 complete; selection {"inconclusive fallback" if selection["selection_inconclusive"] else "complete"}',ha='center',fontsize=11,color='#8B2635')
    fig.text(.5,.015,'Persistent: 26k + 30k, grids 192 + 384, probes and topology. Stack gaps 0.012/0.05 are full widths; mixed feature is half-thickness.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.83]);fig.savefig(DEST/'summary.png',dpi=180,facecolor='white');plt.close(fig)
    base_path=DEST/'sealed-generator-report.md'
    report_path=DEST/'report.md'
    if not base_path.exists():base_path.write_bytes(report_path.read_bytes())
    cycle_joint=analysis['conditional_joint_success_time']['cycling']
    random_joint=analysis['conditional_joint_success_time']['random']
    editorial=['## Measured comparison and limitations','',
        f'Cycling saves {(1-analysis["cycling_dense_runtime_ratio"])*100:.2f}% of the full fixed-trajectory pipeline. On {cycle_joint["pairs"]} jointly successful pairs, mean first observed persistent correctness is {cycle_joint["mean_candidate_seconds"]:.2f}s vs dense {cycle_joint["mean_dense_seconds"]:.2f}s ({(1-cycle_joint["mean_candidate_seconds"]/cycle_joint["mean_dense_seconds"])*100:.2f}% reduction). Random conditional correctness is {random_joint["mean_candidate_seconds"]:.2f}s ({(1-random_joint["mean_candidate_seconds"]/random_joint["mean_dense_seconds"])*100:.2f}% reduction). These conditional retrospective timings do not establish online stopping or resolve the failed narrow-stack pairs.', '',
        'Full pipeline is external supervisor launch through durable process exit, including interpreter startup, independently repeated initialization/prefixes, labels/pools, random schedules, recovery, transfers, evaluations, checkpoints and hashes. Failed attempts count. These are new actual wall clocks, distinct from earlier shared-prefix attribution.', '',
        '| Arm | Actual pipeline minutes | Saving against dense | Persistent successes |','|---|---:|---:|---:|']
    for arm in arms:
        seconds=analysis['full_pipeline_seconds'][arm]
        editorial.append(f'| {arm} | {seconds/60:.3f} | {(1-seconds/analysis["full_pipeline_seconds"]["dense"])*100:.2f}% | {analysis["persistent_successes"][arm]}/18 |')
    editorial+=['', 'Family denominators per arm: parallel plates3, mixed thickness9, multiscale stacks6. Mixed feature values0.015/0.03/0.06 are half-thickness (full thin thickness0.03/0.06/0.12); stack gaps0.012/0.05 are full widths.', '',
        f'Repair charged {stages["development"]["charged_seconds"]/60:.3f}/60 minutes:13/16 screen trajectories completed,1 retained file-access failure and2 unstarted. All8 narrow development trajectories failed persistent gates. Model comparison is inconclusive; the frozen SIREN fallback was used, without validation-seed expansion. This does not prove either architecture incapable of narrow reconstruction.', '',
        f'Target checks: complete evidence={analysis["complete_valid_evidence"]}; lost dense successes={analysis["lost_dense_successes"]}; cycling aggregate error guards={analysis["aggregate_guards"]["cycling"]["passes"]}; all6 narrow stack pairs persistent with individual guards={analysis["narrow_stack_all6_with_guards"]}. Speed saving and dense-success preservation do not establish narrow reliability.', '',
        '| Arm | Endpoint | Gap RMSE ratio (limit1.10) | Full-gap sign delta (minimum−0.005) | Chamfer192 ratio (limit1.05) | Chamfer384 ratio (limit1.05) | Pass |',
        '|---|---:|---:|---:|---:|---:|---|']
    for arm,value in analysis['aggregate_guards'].items():
        for row in value['endpoints']:
            number=lambda key:f'{row[key]:.5f}' if row.get(key) is not None else 'unavailable'
            editorial.append(f'| {arm} | {row["step"]} | {number("gap_rmse_ratio")} | {number("sign_delta")} | {number("chamfer_ratio_192")} | {number("chamfer_ratio_384")} | {"pass" if row["passes"] else "fail"} |')
    editorial+=['', 'Conditional checkpoint-limited observed time to correctness applies only to jointly persistent successful pairs. It excludes paired failures from this conditional statistic; their consumed time and null onset remain in trajectories.csv and full pipeline totals. It is retrospective, not an online stopping policy.']
    editorial+=['', '| Candidate | Jointly successful pairs | Candidate mean seconds | Dense mean seconds | Reduction |', '|---|---:|---:|---:|---:|']
    for arm,value in analysis['conditional_joint_success_time'].items():
        editorial.append(f'| {arm} | {value["pairs"]} | {value["mean_candidate_seconds"]:.2f} | {value["mean_dense_seconds"]:.2f} | {(1-value["mean_candidate_seconds"]/value["mean_dense_seconds"])*100:.2f}% |')
    editorial+=['', 'Representative narrow failure, case6 seed71 dense23k: analytic references resolved three components/Euler6 at both192/384; predicted closed meshes have two components, Euler−56/−8 respectively. Original component interior signs passed at that checkpoint. This identifies a topology mismatch without identifying which boxes merged. All reference-resolution totals and every checkpoint failure reason are exported.', '',
        '![Retained-result summary](summary.png)', '', '[Execution, dimension conventions and artifact audit](supplement.md). The sealed reporter remains unchanged; this editorial section is generated by the separately hashed read-only analysis script.']
    editorial+=['', 'The [NeurIPS2020 Fourier-features paper](https://proceedings.nips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html) motivates testing high-frequency regression through tunable bandwidth. It does not establish topology repair. The present fixed Fourier configuration failed measured narrow development gates; incomplete screening prevents a general architectural conclusion.']
    lead=(f'**Measured saving with unresolved narrow reliability:** cycling reduces full fixed-trajectory pipeline time by {(1-analysis["cycling_dense_runtime_ratio"])*100:.2f}%, preserves all dense successes, and passes aggregate error guards. Every arm has 12/18 persistent successes and 0/6 narrow-stack successes. On the 12 jointly successful pairs, first observed persistent correctness is {cycle_joint["mean_candidate_seconds"]:.2f}s vs dense {cycle_joint["mean_dense_seconds"]:.2f}s ({(1-cycle_joint["mean_candidate_seconds"]/cycle_joint["mean_dense_seconds"])*100:.2f}% reduction). These are separate retrospective measurements; the complete reliability target is not achieved.\n\n')
    base=base_path.read_text(encoding='utf-8').replace('# Thin-feature campaign\n\n','# Thin-feature campaign\n\n'+lead,1)
    base=base.replace("`{'model': 'siren', 'width': 128, 'depth': 3}`",'SIREN, width 128, depth 3')
    for arm in arms:
        old=str(analysis['total_time_per_persistent'][arm])
        base=base.replace(old,f'{analysis["total_time_per_persistent"][arm]:.1f}')
    for family,denominator in [('parallel_plates',3),('mixed_thickness',9),('multiscale_plate_stack',6)]:
        values=analysis['family_successes'][family]
        old=f'| {family} | {values["dense"]} | {values["random"]} | {values["cycling"]} |'
        new=f'| {family.replace("_"," ")} | {values["dense"]}/{denominator} | {values["random"]}/{denominator} | {values["cycling"]}/{denominator} |'
        base=base.replace(old,new)
    base=re.sub(r'Cycling/dense actual pipeline ratio: .*?Narrow-stack six-pair gate with error guards: False\.',f'Cycling/dense actual pipeline ratio: {analysis["cycling_dense_runtime_ratio"]:.4f}. Lost dense successes: {analysis["lost_dense_successes"]}. Narrow-stack six-pair gate with error guards: fail.',base)
    editorial_text='\n'.join(editorial).replace('=True','=pass').replace('=False','=fail')
    for word in ['plates','thickness','stacks','values','thicknesses','case','seed','dense','Euler','limit','minimum','Repair','charged','trajectories','completed','with','NeurIPS','gaps','All','all','both','and']:
        editorial_text=re.sub(r'('+word+r')(?=\d)',r'\1 ',editorial_text)
    editorial_text=editorial_text.replace('minutes:13','minutes: 13').replace('completed,1','completed, 1')
    editorial_text=re.sub(r'(?<=\d)(s|k|minutes|screen|narrow|persistent)(?=[ ,;.])',r' \1',editorial_text)
    report_path.write_text(base+'\n'+editorial_text+'\n',encoding='utf-8')
    output_paths=[p for p in DEST.iterdir() if p.is_file() and p.name!='final-output-inventory.json']+[Path(__file__),ROOT/'README.md']
    write(DEST/'final-output-inventory.json',dict(scope='Final review outputs and output-only analysis source; self excluded',
        files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output_paths)]))
    print(json.dumps(supplemental),flush=True)


if __name__=='__main__':main()
