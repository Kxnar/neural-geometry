"""Read-only scientific reporting; never constructs a scene or evaluates a model."""
import csv
import math
from scripts.thin_feature_campaign import DEST,RUNS,read,write,load_completed,guard,aggregate_guards


def ratio(a,b):
    return a/b if b else None


def report():
    rows=load_completed('confirmation'); ledger=read(DEST/'budget.json'); selection=read(DEST/'selection.json')
    seal=read(DEST/'confirmation-manifest.json'); audit=read(DEST/'audit.json')
    lookup={(r['task']['case_index'],r['task']['seed'],r['task']['arm']):r for r in rows}
    attempts=[a for a in ledger['attempts'] if a['stage']=='confirmation']; arm_times={a:0. for a in ['dense','random','cycling']}
    for attempt in attempts:
        if attempt['task']: arm_times[attempt['task'].rsplit('-',1)[-1]]+=attempt['charged_seconds']
    counts={a:sum(r['persistence']['persistent'] for r in rows if r['task']['arm']==a) for a in arm_times}
    comparisons=[]; complete=len(rows)==54 and audit['prefix_pairing'] and all(a['finished'] for a in attempts)
    for case_index in range(2,8):
        for seed in [71,72,73]:
            dense=lookup.get((case_index,seed,'dense'))
            for arm in ['random','cycling']:
                candidate=lookup.get((case_index,seed,arm)); dp=bool(dense and dense['persistence']['persistent']); cp=bool(candidate and candidate['persistence']['persistent'])
                comparisons.append(dict(case_index=case_index,seed=seed,arm=arm,dense_persistent=dp,candidate_persistent=cp,
                                        lost_dense=dp and not cp,candidate_only=cp and not dp,paired_failure=not dp and not cp,
                                        guards=bool(dense and candidate and guard(candidate['records'],dense['records']))))
    lost=sum(c['lost_dense'] for c in comparisons if c['arm']=='cycling')
    narrow=all(c['candidate_persistent'] and c['guards'] for c in comparisons if c['case_index'] in [6,7] and c['arm']=='cycling')
    cycle_ratio=ratio(arm_times['cycling'],arm_times['dense'])
    aggregate={}
    for arm in ['random','cycling']:
        pairs=[(lookup[(i,s,arm)]['records'],lookup[(i,s,'dense')]['records']) for i in range(2,8) for s in [71,72,73] if (i,s,arm) in lookup and (i,s,'dense') in lookup]
        aggregate[arm]=aggregate_guards(pairs)
    target=complete and not lost and narrow and aggregate['cycling']['passes'] and cycle_ratio is not None and cycle_ratio<=.75
    metrics=dict(complete_valid_evidence=complete,persistent_successes=counts,expected_pairs_per_arm=18,
                 full_pipeline_seconds=arm_times,cycling_dense_runtime_ratio=cycle_ratio,lost_dense_successes=lost,
                 narrow_stack_all6_with_guards=narrow,target_achieved=target,comparisons=comparisons,
                 aggregate_guards=aggregate,total_time_per_persistent={a:(arm_times[a]/counts[a] if counts[a] else 'infinity') for a in counts})
    write(DEST/'analysis.json',metrics)
    with (DEST/'paired-results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    flat=[]
    for task in seal['tasks']:
        r=lookup.get((task['case_index'],task['seed'],task['arm']))
        charged=sum(a['charged_seconds'] for a in attempts if a['task']==task['id'])
        onset=r['persistence']['onset'] if r else None
        through=next((x['cumulative_pipeline_seconds'] for x in r['records'] if x['step']==onset),None) if r else None
        flat.append(dict(task=task['id'],family=task['case']['scene'],case_index=task['case_index'],seed=task['seed'],arm=task['arm'],
                         status='complete' if r else ('failed_or_capped' if charged else 'not_started_budget_exhausted'),
                         persistent=bool(r and r['persistence']['persistent']),onset=onset,observed_time_to_correct=through,
                         charged_pipeline_seconds=charged,censor_horizon_seconds=charged if onset is None else None,
                         later_verification_seconds=charged-through if through is not None else None))
    with (DEST/'trajectories.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    checkpoint_rows=[];regions=[]
    for r in rows:
        t=r['task']
        for x in r['records']:
            for res,e in x['evaluations'].items():
                checkpoint_rows.append(dict(task=t['id'],family=t['case']['scene'],step=x['step'],resolution=res,
                                            primary=x['primary_gate'],reference_resolved=e['reference_resolved'],mesh_present=e['mesh_present'],
                                            components=e['mesh']['components'],euler=e['mesh']['euler'],closed_edges=e['mesh']['closed_edges'],
                                            chamfer=e['symmetric_chamfer'],failure_reasons=';'.join(x['failure_reasons'])))
            for dataset in ['independent','full_gap']:
                for name,metric in x[dataset].items(): regions.append(dict(task=t['id'],family=t['case']['scene'],step=x['step'],dataset=dataset,region=name,**metric))
    for filename,records in [('checkpoints.csv',checkpoint_rows),('regions.csv',regions)]:
        if records:
            with (DEST/filename).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    families={}
    for family in ['parallel_plates','mixed_thickness','multiscale_plate_stack']:
        families[family]={a:sum(r['persistence']['persistent'] for r in rows if r['task']['case']['scene']==family and r['task']['arm']==a) for a in counts}
    joint={}
    for arm in ['random','cycling']:
        own=[c for c in comparisons if c['arm']==arm and c['dense_persistent'] and c['candidate_persistent']]
        times=[]
        for x in own:
            def observed(a):
                rr=lookup[(x['case_index'],x['seed'],a)]
                return next(y['cumulative_pipeline_seconds'] for y in rr['records'] if y['step']==rr['persistence']['onset'])
            times.append(dict(case_index=x['case_index'],seed=x['seed'],candidate=observed(arm),dense=observed('dense')))
        joint[arm]=dict(pairs=len(times),mean_candidate_seconds=sum(x['candidate'] for x in times)/len(times) if times else None,
                       mean_dense_seconds=sum(x['dense'] for x in times)/len(times) if times else None,observations=times)
    metrics.update(family_successes=families,conditional_joint_success_time=joint)
    write(DEST/'analysis.json',metrics)
    development=load_completed('development');devflat=[]
    for r in development:
        t=r['task'];devflat.append(dict(model=t['config']['model'],feature=t['case']['feature'],angle=t['case']['angle'],seed=t['seed'],arm=t['arm'],
                                       persistent=r['persistence']['persistent'],worker_seconds=r['worker_seconds']))
    if devflat:
        with (DEST/'development-results.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(devflat[0]));writer.writeheader();writer.writerows(devflat)
    factors=[]
    for model in ['siren','fourier']:
        for arm in ['dense','cycling']:
            for angle in [17,37]:
                narrowrow=next((r for r in devflat if r['model']==model and r['arm']==arm and r['feature']==.012 and r['angle']==angle and r['seed']==71),None)
                widerow=next((r for r in devflat if r['model']==model and r['arm']==arm and r['feature']==.03 and r['angle']==angle and r['seed']==71),None)
                factors.append(dict(model=model,arm=arm,factor='narrow_minus_wide',fixed=str(angle),seed=71,
                                    persistent_difference=int(narrowrow['persistent'])-int(widerow['persistent']) if narrowrow and widerow else None))
            for width in [.012,.03]:
                hi=next((r for r in devflat if r['model']==model and r['arm']==arm and r['feature']==width and r['angle']==37 and r['seed']==71),None)
                lo=next((r for r in devflat if r['model']==model and r['arm']==arm and r['feature']==width and r['angle']==17 and r['seed']==71),None)
                factors.append(dict(model=model,arm=arm,factor='37_minus_17',fixed=str(width),seed=71,
                                    persistent_difference=int(hi['persistent'])-int(lo['persistent']) if hi and lo else None))
    with (DEST/'development-factors.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(factors[0]));writer.writeheader();writer.writerows(factors)
    lines=['# Thin-feature campaign','',
           f'**Target {"achieved" if target else "not achieved"}.** Completed {len(rows)}/54 independent confirmation trajectories. ',
           f'Selected configuration: `{selection["selected_config"]}`; development selection {"complete" if selection["comparison_complete"] else "inconclusive (frozen SIREN fallback)"}.','',
           '| Arm | Persistent successes /18 | Full pipeline seconds, all attempts | Seconds per persistent success |',
           '|---|---:|---:|---:|']
    for arm in counts: lines.append(f'| {arm} | {counts[arm]} | {arm_times[arm]:.1f} | {metrics["total_time_per_persistent"][arm]} |')
    lines+=['','| Family | Dense successes | Random successes | Cycling successes |','|---|---:|---:|---:|']
    for family,v in families.items(): lines.append(f'| {family} | {v["dense"]} | {v["random"]} | {v["cycling"]} |')
    lines+=['',f'Cycling/dense actual pipeline ratio: {cycle_ratio}. Lost dense successes: {lost}. Narrow-stack six-pair gate with error guards: {narrow}.',
            '', 'Missing, failed, unresolved and nonfinite evaluations fail closed. Failed attempts remain in pipeline totals. Unstarted trajectories are explicitly censored in trajectories.csv and prevent a complete-evidence claim.',
            '', 'Persistent success requires both 26k and 30k, both grids 192 and 384, every original probe sign, closed mesh edges, expected components/Euler and resolved analytic references. Onset at23k requires all three checkpoints; onset26k requires26k+30k. These are retrospective checkpoint-limited observations, not online stopping.',
            '', 'Runtime includes independently repeated prefixes, pool/label generation, random schedules, evaluation, transfers and checkpoint/hash I/O. Worker cumulative onset time includes supervisor launch and interpreter startup; final closure overhead appears in full pipeline totals. Historical attributed savings are separate and do not establish this target. Historical error guards compare endpoint means across the full paired set; per-pair guards are separately reported, with strict narrow-stack preservation.',
            '', 'Scope: oracle analytic box unions in three families. Known regions and exact SDF labels are used. This is not arbitrary unseen-shape evidence or a topology guarantee. Fourier features test bandwidth; they do not promise topology repair.',
            '', 'Random is evaluated identically. If random matches cycling, these results do not establish balancing necessity or superiority.',
            '', 'See protocol.json, audit.json, budget.json, paired-results.csv and trajectories.csv for immutable setup, failures and measurements.']
    (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': report()
