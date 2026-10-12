"""I/O-only continuation of the frozen order gate after a retained marker failure.

Scientific train/evaluation functions, input data, schedules and criteria remain
the original hashed implementation. Existing measurements are never recomputed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

import neural_geometry_scale.study as study
import scripts.gap_confirmation as gate
import scripts.gap_density_controls as density
import scripts.gap_lr_controls as lr


ORIGINAL_ATOMIC=study.atomic_json


def resilient_json(path,data):
    for attempt in range(40):
        try:
            ORIGINAL_ATOMIC(path,data)
            return
        except PermissionError:
            if attempt==39: raise
            print(json.dumps(dict(io_retry=str(path),attempt=attempt+1)),flush=True)
            time.sleep(.1)


def install_retry():
    # Only atomic artifact publication is adapted; numerical functions unchanged.
    for module in [study,gate,density,lr]: module.atomic_json=resilient_json


def validate_saved(folder,c,pair,arm,order,previous,end,data,m):
    study.check_files(folder,study.read(folder/'training-complete.json'))
    r=study.read(folder/'metrics.json'); t=r['training']
    assert gate.valid_row(r) and r['metadata']['config']==c and r['metadata']['step']==end
    assert r['metadata']['source_hashes']==m['source_hashes']
    assert r['metadata']['checkpoint_sha256']==study.digest(folder/'checkpoint.pt')
    assert t==study.read(folder/'training.json') and t['parent_sha256']==study.digest(previous)
    assert t['arm']==arm and t['end']==end and t['point_presentations']==(end-t['start'])*arm['points']
    assert t['stream_sha256']==gate.stream_hash(data,arm,order,t['start'],end)
    assert t['order_sha256']==lr.tensor_hash(order)
    assert t['sample_rng_sha256']==t['initial_sample_rng_sha256']
    saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    parent=torch.load(previous,map_location='cpu',weights_only=True)
    assert saved['config']==c and saved['step']==end and parent['step']==t['start']
    assert all(int(s['step'])==end for s in saved['optimizer']['state'].values())
    assert torch.equal(saved['sample_rng'],parent['sample_rng'])
    assert lr.tensor_hash(saved['sample_rng'])==t['sample_rng_sha256']
    for res in ['192','384']:
        e=study.read(folder/f'evaluation-r{res}.json')
        assert e['metadata']==r['metadata'] and e['metrics']==r['evaluations'][res]
    return r


def run():
    install_retry(); m=study.read(gate.DEST/'batch-1-manifest.json'); gate.verify_manifest(m)
    freeze=study.read(gate.DEST/'io-recovery-freeze.json')
    assert freeze['source_sha256']==study.digest(__file__) and freeze['original_manifest_sha256']==study.digest(gate.DEST/'batch-1-manifest.json')
    out=study.ROOT/'runs/gap-confirmation/batch-1'; records=study.read(out/'summary.json')['runs']
    exact=study.read(out/'reproduction.json')['checkpoints']
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    existing_keys={(r['metadata']['config']['policy'],r['metadata']['config']['seed'],r['training']['arm']['name'],r['metadata']['step']) for r in records}
    assert len(existing_keys)==len(records)==45
    recovered=[]; trained=[]
    for index,pair in enumerate(m['pairs']):
        c=pair['config']; root=out/study.identity(c)
        data=torch.load(study.ROOT/pair['pool'],weights_only=True)
        independent=torch.load(study.ROOT/pair['independent'],weights_only=True)
        orders=torch.load(root/'orders.pt',weights_only=True)
        assert study.digest(root/'orders.pt')==study.read(root/'orders.json')['sha256']
        for arm in gate.ARMS[index%4:]+gate.ARMS[:index%4]:
            assert torch.equal(orders[arm['name']],gate.schedule(arm['name'],index))
            previous=study.ROOT/pair['checkpoint']
            for end in gate.ENDS:
                folder=root/arm['name']/f'step-{end}'/'attempt-0001'
                key=(c['policy'],c['seed'],arm['name'],end)
                if key in existing_keys:
                    study.check_files(folder,study.read(folder/'complete.json'))
                    previous=folder/'checkpoint.pt'; continue
                if (folder/'metrics.json').exists():
                    # Promote only a temp marker whose every listed payload matches.
                    marker=study.read(folder/'complete.json.tmp'); study.check_files(folder,marker)
                    archive=gate.DEST/'retained-io-failure'; archive.mkdir(exist_ok=True)
                    for name in ['complete.json','complete.json.tmp','metrics.json']:
                        shutil.copyfile(folder/name,archive/name.replace('.tmp','-tmp'))
                    r=validate_saved(folder,c,pair,arm,orders[arm['name']],previous,end,data,m)
                    r['stage_seconds']=None
                    r['io_recovery']=dict(training_repeated=False,evaluation_repeated=False,original_temp_marker_payloads_verified=True,
                        stage_timing='Unavailable: interrupted before enclosing stage timer was persisted')
                    recovered.append(dict(key=list(key),folder=folder.relative_to(study.ROOT).as_posix(),checkpoint_sha256=study.digest(folder/'checkpoint.pt')))
                else:
                    # Only the final local33 dense stages are absent.
                    assert key in [('local',33,'dense',26000),('local',33,'dense',30000)] and not folder.exists()
                    stage=time.perf_counter()
                    gate.train(c,previous,end,folder,data,arm,orders[arm['name']])
                    r=density.measurements(c,folder,m,data,independent)
                    r['stage_seconds']=time.perf_counter()-stage
                    saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
                    old=torch.load(study.ROOT/pair['historical'][arm['name']][str(end)],map_location='cpu',weights_only=True)
                    exact.append(dict(policy=c['policy'],seed=c['seed'],arm=arm['name'],step=end,
                        **{key+'_exact':gate.same(saved[key],old[key]) for key in ['model','optimizer','sample_rng']}))
                    resilient_json(out/'reproduction.json',dict(checkpoints=exact)); trained.append(list(key))
                resilient_json(folder/'metrics.json',r)
                resilient_json(folder/'complete.json',{p.name:study.digest(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json' and not p.name.endswith('.tmp')})
                records.append(dict(folder=folder.relative_to(study.ROOT).as_posix(),**r)); resilient_json(out/'summary.json',dict(runs=records))
                previous=folder/'checkpoint.pt'
                print(json.dumps(dict(pair=f"{c['policy']}:{c['seed']}",arm=arm['name'],step=end,primary=r['primary_gate'],
                    recovered_without_model_work=bool(r.get('io_recovery')),completed=len(records),expected=48)),flush=True)
    assert len(records)==48 and len(trained)==2 and len(recovered)==1
    gate.verify_manifest(m)
    resilient_json(out/'io-recovery-audit.json',dict(recovered_metrics=recovered,new_model_stages=trained,
        source_sha256=study.digest(__file__),numerical_sources_unchanged=True))
    resilient_json(out/'run-complete.json',dict(checkpoints=48,summary_sha256=study.digest(out/'summary.json'),reservation_unchanged=True))


def budget_continue():
    ledger=gate.DEST/'budget.json'; state=study.read(ledger); remaining=gate.allowance(state)
    assert remaining>120 and state['attempts'][-1]['batch']==1 and state['attempts'][-1]['exit_code']==1
    assert not (gate.DEST/'io-recovery-freeze.json').exists()
    m=study.read(gate.DEST/'batch-1-manifest.json'); gate.verify_manifest(m)
    out=study.ROOT/'runs/gap-confirmation/batch-1'
    archive=gate.DEST/'retained-io-failure'; archive.mkdir(exist_ok=True)
    failed=out/'c88efd8a5355dcbf/shuffle/step-30000/attempt-0001'
    snapshot={}
    for src,name in [(out/'summary.json','summary-before-recovery.json'),(out/'reproduction.json','reproduction-before-recovery.json'),
                     (gate.DEST/'batch-1-stdout.log','failed-stdout.log'),(ledger,'budget-before-recovery.json'),
                     (failed/'complete.json','complete.json'),(failed/'complete.json.tmp','complete.json-tmp'),(failed/'metrics.json','metrics.json')]:
        shutil.copyfile(src,archive/name); snapshot[name]=study.digest(archive/name)
    study.check_files(failed,study.read(failed/'complete.json.tmp'))
    resilient_json(archive/'snapshot.json',snapshot)
    resilient_json(gate.DEST/'io-recovery-freeze.json',dict(created_utc=gate.utc(),source_sha256=study.digest(__file__),
        original_manifest_sha256=study.digest(gate.DEST/'batch-1-manifest.json'),
        adaptation='I/O-only atomic-write retry and same-batch continuation. Promote hash-verified saved metrics. No repeated numerical work, criteria/input/order changes.',
        allowed_new_stages=[['local',33,'dense',26000],['local',33,'dense',30000]],missing_stage_timing='Retain null; do not infer timing.'))
    segment=dict(batch=1,recovery_of_attempt=len(state['attempts']),segment=2,finished=False,started_utc=gate.utc(),
        source_sha256=study.digest(__file__),manifest_sha256=study.digest(gate.DEST/'batch-1-manifest.json'),
        command=['-m','scripts.recover_gap_confirmation_io','run'],log='results/gap-confirmation/batch-1-recovery-stdout.log')
    # The failed attempt is immutable. A linked segment shares its logical batch ID.
    state['attempts'].append(segment)
    resilient_json(ledger,state); began=time.perf_counter(); code=-1
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(study.ROOT),PYTHONDONTWRITEBYTECODE='1')
    try:
        with (study.ROOT/segment['log']).open('x',encoding='utf-8') as f:
            code=subprocess.run([sys.executable]+segment['command'],cwd=study.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining-5).returncode
    except subprocess.TimeoutExpired: segment['timed_out']=True
    finally:
        segment.update(finished=True,finished_utc=gate.utc(),charged_seconds=time.perf_counter()-began,exit_code=code)
        resilient_json(ledger,state)
    print(json.dumps(segment),flush=True)
    if code: raise SystemExit(code)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['run','budget-continue']); a=p.parse_args()
    if a.command=='run': run()
    else: budget_continue()
