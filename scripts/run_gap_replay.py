"""Future clean replay of the frozen gate, adding only atomic I/O retries."""
from pathlib import Path
import argparse
from neural_geometry_scale.study import ROOT,read
from scripts.gap_confirmation import run,charged,DEST,qualify
from scripts.recover_gap_confirmation_io import install_retry
from scripts.recover_gap_confirmation_io import resilient_json as atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',action='store_true'); a=p.parse_args()
    install_retry()
    if a.run:
        out=ROOT/'runs/gap-confirmation/batch-1'
        run(DEST/'batch-1-manifest.json',out)
        decision=qualify(read(out/'summary.json')['runs'],read(out/'reproduction.json')['checkpoints'])
        atomic_json(DEST/'replay-order-gate-decision.json',decision)
        if not decision['gate_pass']: raise RuntimeError('Fresh replay failed the frozen order gate; stop before reserved confirmation')
    else: charged(['-m','scripts.run_gap_replay','--run'],batch=1)
