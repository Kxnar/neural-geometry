import os
import pytest
from scripts import thin_feature_io_retry as io


def windows_error(code):
    error=PermissionError('synthetic Windows access lock');error.winerror=code;return error


def test_only_replace_is_retried_for_observed_windows_lock(monkeypatch,tmp_path):
    attempts=[];original=os.replace
    def flaky(source,dest,**kwargs):
        attempts.append(1)
        if len(attempts)<3: raise windows_error(5)
        return original(source,dest,**kwargs)
    monkeypatch.setattr(io,'_original_replace',flaky);monkeypatch.setattr(io.time,'sleep',lambda seconds:None)
    source=tmp_path/'source';dest=tmp_path/'dest';source.write_bytes(b'exact frozen checkpoint bytes')
    io.retry_replace(source,dest)
    assert len(attempts)==3 and dest.read_bytes()==b'exact frozen checkpoint bytes' and not source.exists()


def test_non_windows_permission_and_other_io_errors_are_not_swallowed(monkeypatch):
    def fail(*args,**kwargs): raise PermissionError('not a Windows share lock')
    monkeypatch.setattr(io,'_original_replace',fail)
    with pytest.raises(PermissionError):io.retry_replace('a','b')
    def unavailable(*args,**kwargs): raise FileNotFoundError('missing source')
    monkeypatch.setattr(io,'_original_replace',unavailable)
    with pytest.raises(FileNotFoundError):io.retry_replace('a','b')


def test_retry_is_bounded_and_does_not_touch_model_sampling_or_evaluation(monkeypatch):
    ticks=iter([0,0,0,6]);monkeypatch.setattr(io.time,'perf_counter',lambda:next(ticks))
    monkeypatch.setattr(io.time,'sleep',lambda seconds:None)
    def fail(*args,**kwargs):raise windows_error(5)
    monkeypatch.setattr(io,'_original_replace',fail)
    with pytest.raises(PermissionError):io.retry_replace('a','b')
    raw=(io.campaign.ROOT/io.WRAPPER_PATH).read_text()
    assert 'factory=' not in raw and 'geometry=' not in raw and 'batch=' not in raw and 'loss=' not in raw


def test_wrapper_hashes_and_worker_dispatch_are_explicit(monkeypatch):
    monkeypatch.setattr(io,'_installed',False);hashes=io.campaign.scientific_hashes
    scientific_names=['factory','worker','pool','sample_regions','geometry','persistent','guard','aggregate_guards']
    scientific_before={name:getattr(io.campaign,name) for name in scientific_names}
    supervise=io.campaign.supervise;seen=[]
    monkeypatch.setattr(io.campaign,'supervise',lambda command,*args,**kwargs:seen.append(command))
    monkeypatch.setattr(io.campaign,'scientific_hashes',hashes)
    monkeypatch.setattr(io.campaign,'check_manifest',io.campaign.check_manifest)
    original_replace=os.replace
    io.install()
    assert io.WRAPPER_PATH in io.campaign.scientific_hashes() and io.TEST_PATH in io.campaign.scientific_hashes()
    io.campaign.supervise(['-m','scripts.thin_feature_campaign','worker','--task','fixture'],'qa')
    assert seen[0][:3]==['-m','scripts.thin_feature_io_retry','worker']
    assert io.campaign.factory.__module__=='scripts.thin_feature_campaign'
    assert all(getattr(io.campaign,name) is original for name,original in scientific_before.items())
    # Restore global OS behavior before pytest fixture cleanup.
    os.replace=original_replace


def test_original_development_seal_and_new_confirmation_inventory(monkeypatch):
    monkeypatch.setattr(io,'_installed',False)
    monkeypatch.setattr(io.campaign,'scientific_hashes',lambda:{'original-scientific-source':'unchanged'})
    monkeypatch.setattr(io.campaign,'supervise',lambda *a,**k:None)
    seen=[]
    monkeypatch.setattr(io.campaign,'check_manifest',lambda path:seen.append(io.campaign.scientific_hashes()))
    monkeypatch.setattr(io.campaign,'read',lambda path:dict(stage='development',sources={'original-scientific-source':'unchanged'}))
    original_replace=os.replace;io.install()
    io.campaign.check_manifest('preserved-development-seal')
    assert seen[-1]=={'original-scientific-source':'unchanged'}
    assert io.WRAPPER_PATH in io.campaign.scientific_hashes()
    monkeypatch.setattr(io.campaign,'read',lambda path:dict(stage='confirmation',sources={}))
    with pytest.raises(RuntimeError,match='does not include'):io.campaign.check_manifest('invalid-new-confirmation')
    os.replace=original_replace
