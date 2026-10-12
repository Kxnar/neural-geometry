import base64
import hashlib
import pytest
from scripts.verify_thin_publication import apply_recipe,check_bytes,historical_reference

def record(data):return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def test_transformation_recipe_and_reference():
    original=b'header\nlegacy credit\nmeasured=12\n';public=b'header\nmeasured=12\n'
    edits=[dict(start=7,end=21,replacement_base64=base64.b64encode(b'').decode())]
    assert apply_recipe(original,edits)==public
    entries={'result':dict(status='transformed',original=record(original),public=record(public))}
    historical_reference('result',record(original)['sha256'],entries)
    historical_reference('result',record(public)['sha256'],entries)
    check_bytes(public,entries['result']['public'])
def test_unchanged_reference():
    data=b'metric=26.00\n';entries={'result':dict(status='unchanged',original=record(data),public=record(data))}
    historical_reference('result',record(data)['sha256'],entries);check_bytes(data,record(data))
def test_explicit_nonbinary_document_omission():
    data=b'obsolete project note';entries={'note.md':dict(status='omitted',original=record(data))}
    historical_reference('note.md',record(data)['sha256'],entries)
    with pytest.raises(ValueError):historical_reference('note.md','0'*64,entries)
def test_public_tamper_rejected():
    with pytest.raises(ValueError):check_bytes(b'metric=13',record(b'metric=12'))
    with pytest.raises(ValueError):apply_recipe(b'abc',[dict(start=2,end=1,replacement_base64='')])
def test_unknown_historical_reference_rejected():
    entries={'result':dict(status='transformed',original=record(b'original'),public=record(b'public'))}
    with pytest.raises(ValueError):historical_reference('result','0'*64,entries)
