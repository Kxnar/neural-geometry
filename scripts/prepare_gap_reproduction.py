"""Create an empty-output, self-contained replay workspace. No model work."""
import argparse
import shutil
from pathlib import Path
from neural_geometry_scale.study import ROOT,read,digest,atomic_json
from scripts.gap_confirmation import DEST,INPUTS


def prepare(destination):
    destination=destination.resolve(); destination.relative_to(ROOT)
    if destination==ROOT or destination.exists(): raise RuntimeError('Choose a new child directory within this workspace')
    destination.mkdir(parents=True); files={}
    manifest=read(DEST/'confirmation-manifest.json')
    paths=set(manifest['source_hashes'])
    paths.update(f['path'] for f in read(INPUTS/'inventory.json')['files'])
    paths.add('inputs/gap-confirmation/inventory.json')
    paths.update('results/gap-confirmation/'+n for n in ['batch-1-manifest.json','confirmation-manifest.json','decision.json'])
    paths.add('results/gap-sampling/fresh-cases-reserved.json')
    paths.update(['scripts/report_gap_confirmation.py','scripts/report_reserved_confirmation.py','scripts/prepare_gap_reproduction.py','scripts/run_gap_replay.py','scripts/verify_gap_exports.py','scripts/verify_gap_reproduction.py','README.md','.gitattributes'])
    for relative in sorted(paths):
        src=ROOT/relative; target=destination/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,target)
        assert digest(src)==digest(target); files[relative]=digest(target)
    result=destination/'results/gap-confirmation'
    atomic_json(result/'budget.json',dict(limit_seconds=7200,maximum_batches=3,attempts=[]))
    reservation=read(INPUTS/'fresh-cases-reserved.json')
    atomic_json(result/'reservation-consumption.json',dict(sha256=manifest['reservation_sha256'],selected_cases=reservation['cases'][:2],untouched_cases=reservation['cases'][2:],events=[]))
    atomic_json(destination/'replay-preparation.json',dict(files=files,model_evaluations=0,training_updates=0,
        destination=str(destination),purpose='Separate future reproduction; not another experiment in the delivered campaign'))
    print('Prepared',len(files),'hash-verified files at',destination)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--destination',type=Path,required=True); a=p.parse_args(); prepare(a.destination)
