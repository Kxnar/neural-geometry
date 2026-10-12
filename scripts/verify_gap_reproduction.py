"""Verify the whole reporter in a clean bundle using retained data, without training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from neural_geometry_scale.study import ROOT,digest,atomic_json
from scripts.gap_confirmation import DEST
from scripts.prepare_gap_reproduction import prepare

def verify(destination):
    destination=destination.resolve(); prepare(destination)
    copied=linked=0
    for base in [ROOT/f'runs/gap-confirmation/batch-{i}' for i in [1,2,3]]+[DEST]:
        for src in base.rglob('*'):
            if not src.is_file(): continue
            target=destination/src.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True)
            if src.suffix in ['.pt','.npy']:
                try: os.link(src,target); linked+=1
                except OSError: shutil.copyfile(src,target); copied+=1
            else:
                shutil.copyfile(src,target); copied+=1
    env=dict(os.environ,PYTHONPATH=str(destination),PYTHONDONTWRITEBYTECODE='1')
    commands=[['-m','scripts.report_reserved_confirmation'],['-m','scripts.verify_gap_exports']]
    log=[]
    for command in commands:
        result=subprocess.run([sys.executable]+command,cwd=destination,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log.append(result.stdout)
        if result.returncode:
            (DEST/'retained-replay-verification.log').write_text('\n'.join(log),encoding='utf-8')
            raise RuntimeError('Clean retained-data reporter verification failed; see retained-replay-verification.log')
    assert digest(DEST/'report.md')==digest(destination/'results/gap-confirmation/report.md')
    assert json.loads((DEST/'claim.json').read_text())==json.loads((destination/'results/gap-confirmation/claim.json').read_text())
    # The portable reporter's gate audit explicitly scopes zero reserved calls to batch 1.
    shutil.copyfile(destination/'results/gap-confirmation/audit.json',DEST/'audit.json')
    result=dict(destination=str(destination.relative_to(ROOT)),report_sha256=digest(DEST/'report.md'),report_and_claim_match=True,
                complete_reporter_exercised=True,additional_training_updates=0,additional_model_evaluations=0,
                retained_binary_readonly_hardlinks=linked,copied_evidence_files=copied,
                scope='Package and retained-data reporter verification; a full training replay was not executed')
    (DEST/'retained-replay-verification.log').write_text('\n'.join(log),encoding='utf-8')
    atomic_json(DEST/'replay-verification.json',result); print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--destination',type=Path,required=True)
    verify(p.parse_args().destination)
