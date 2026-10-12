"""Verify public evidence and explicitly mapped historical byte references."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def hash_bytes(data):return hashlib.sha256(data).hexdigest()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def apply_recipe(original,edits):
    output=[];offset=0
    for edit in edits:
        start,end=edit['start'],edit['end']
        if not offset<=start<=end<=len(original):raise ValueError('Invalid edit offsets')
        output.extend([original[offset:start],base64.b64decode(edit['replacement_base64'],validate=True)])
        offset=end
    return b''.join(output)+original[offset:]
def check_bytes(data,record):
    if len(data)!=record['bytes'] or hash_bytes(data)!=record['sha256']:raise ValueError('Evidence byte mismatch')
def historical_reference(name,expected,records):
    record=records[name]
    if record.get('status')=='omitted':
        if expected!=record['original']['sha256']:raise ValueError('Omission reference mismatch')
        return
    allowed={record['public']['sha256']}
    if record.get('original'):allowed.add(record['original']['sha256'])
    if expected not in allowed:raise ValueError(f'Unknown historical reference: {name}')
def verify(git_ref=None,local=None,original_archive=None):
    manifest=read(ROOT/'publication/thin-feature-manifest.json')
    normalization=read(ROOT/'publication/public-normalization.json')
    records={entry['path']:entry for entry in normalization['files']}
    public={name:r['public'] for name,r in records.items() if r['status']!='omitted'}
    mapping_path='publication/public-normalization.json'
    if git_ref:
        checks={**public,mapping_path:dict(bytes=(ROOT/mapping_path).stat().st_size,sha256=digest(ROOT/mapping_path))}
        specs=[(':'+name) if git_ref=='index' else git_ref+':'+name for name in checks]
        payload=subprocess.check_output(['git','cat-file','--batch'],input=('\n'.join(specs)+'\n').encode(),cwd=ROOT)
        offset=0
        for spec,record in zip(specs,checks.values()):
            end=payload.index(b'\n',offset);header=payload[offset:end].split();offset=end+1
            if len(header)!=3 or header[1]!=b'blob':raise ValueError(spec)
            size=int(header[2]);data=payload[offset:offset+size];offset+=size+1
            try:check_bytes(data,record)
            except ValueError as error:raise ValueError(f'Git blob mismatch: {spec}') from error
        if offset!=len(payload):raise ValueError('Unexpected Git batch output')
    for name,record in records.items():
        if record['status']=='omitted':
            if (ROOT/name).exists():raise ValueError(f'Deliberately omitted file present: {name}')
        else:check_bytes((ROOT/name).read_bytes(),record['public'])
    for entry in manifest['included']+manifest.get('publication_infrastructure',[]):historical_reference(entry['path'],entry['sha256'],records)
    for entry in manifest['omitted']:
        if entry['path'] in records:historical_reference(entry['path'],entry['sha256'],records)
        elif not entry['path'].endswith(('.pt','.npy')):raise ValueError('Unmapped omission')
        if (ROOT/entry['path']).exists():raise ValueError('Raw omission unexpectedly present')
        if local:check_bytes((Path(local)/entry['path']).read_bytes(),entry)
    for seal in ['development-manifest.json','confirmation-manifest.json']:
        name='results/thin-feature-campaign/'+seal;path=ROOT/name
        historical_reference(name,Path(str(path)+'.sha256').read_text().strip(),records)
        for source,expected in read(path)['sources'].items():historical_reference(source,expected,records)
    outputs=read(ROOT/'results/thin-feature-campaign/final-output-inventory.json')['files']
    for entry in outputs:historical_reference(entry['path'],entry['sha256'],records)
    raw={entry['path']:entry for entry in manifest['omitted']}
    run_files=read(ROOT/'results/thin-feature-campaign/artifact-inventory.json')['files']
    for entry in run_files:
        if entry['path'] in records:historical_reference(entry['path'],entry['sha256'],records)
        elif raw.get(entry['path'],{}).get('sha256')!=entry['sha256']:raise ValueError('Run artifact reference mismatch')
    archive_checks=0
    if original_archive:
        with zipfile.ZipFile(original_archive) as archive:
            for record in records.values():
                if not record.get('original'):continue
                data=archive.read(record['path']);check_bytes(data,record['original']);archive_checks+=1
                if record['status']=='transformed':check_bytes(apply_recipe(data,normalization['recipes'][record['recipe']]['edits']),record['public'])
                elif record['status']=='unchanged':check_bytes(data,record['public'])
    print(json.dumps(dict(public_files_verified=len(public),transformed=sum(r['status']=='transformed' for r in records.values()),
        deliberate_document_omissions=sum(r['status']=='omitted' for r in records.values()),raw_omissions=len(raw),
        historical_output_references=len(outputs),run_artifact_references=len(run_files),original_archive_files_checked=archive_checks,
        git_ref=git_ref,local_raw_verified=bool(local))))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-ref',help='index or commit/ref: verify exact public Git blobs')
    parser.add_argument('--local-artifacts',help='Optional retained raw artifact root')
    parser.add_argument('--original-archive',help='Optional private original-public-tree ZIP for checking transformation recipes')
    args=parser.parse_args();verify(args.git_ref,args.local_artifacts,args.original_archive)
