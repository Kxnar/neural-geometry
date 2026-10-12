"""Independent SHA-256/local-link audit; no scientific code or model evaluation."""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/gap-confirmation'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def verify():
    inventory=json.loads((DEST/'run-artifacts.json').read_text())
    for name,item in inventory['files'].items():
        p=ROOT/name
        assert p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],name
    export=json.loads((DEST/'export-audit.json').read_text())
    for name,digest in export['files'].items(): assert sha(DEST/name)==digest,name
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',(DEST/'report.md').read_text(encoding='utf-8')):
        if '://' in target or target.startswith('#'): continue
        p=(DEST/target.split('#')[0]).resolve(); p.relative_to(ROOT)
        assert p.is_file(),target
        links.append(target)
    def rows(name):
        with (DEST/name).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
    checkpoint=rows('confirmation-checkpoints.csv'); persistence=rows('confirmation-persistence.csv'); compute=rows('confirmation-compute.csv'); paired=rows('confirmation-paired-differences.csv')
    assert len(checkpoint)==48 and len(persistence)==len(compute)==24 and len(paired)==36
    assert len({(r['case_index'],r['seed'],r['arm'],r['step']) for r in checkpoint})==48
    claim=json.loads((DEST/'claim.json').read_text()); decision=json.loads((DEST/'confirmation-decision.json').read_text())
    assert claim['canonical_qualifies']==decision['canonical_qualifies'] is False
    for arm,total in claim['totals'].items():
        assert sum(r['persistent']=='True' for r in persistence if r['arm']==arm)==total['persistent']
        assert sum(int(r['recovery_presentations']) for r in compute if r['arm']==arm)==total['recovery_presentations']
    budget=json.loads((DEST/'budget.json').read_text()); assert all(a['finished'] for a in budget['attempts'])
    charge=sum(a['charged_seconds'] for a in budget['attempts']); assert charge<=7200
    assert sorted({a['batch'] for a in budget['attempts'] if a.get('batch') is not None})==[1,2,3]
    result=dict(inventory_files=len(inventory['files']),local_links_verified=len(links),confirmation_checkpoints=48,paired_comparisons=36,
                charged_seconds=charge,model_evaluations=0,training_updates=0,report_sha256=sha(DEST/'report.md'))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__': verify()
