"""Versioned I/O-only recovery for Windows/OneDrive atomic-replace locks.

The frozen scientific module is unchanged. Confirmation records this wrapper's
and its tests' hashes, checks them in each worker, and snapshots both. All retry
waits occur inside the externally timed subprocess and its unchanged deadline.
"""
import json
import os
import time
from pathlib import Path

from scripts import thin_feature_campaign as campaign

WRAPPER_PATH='scripts/thin_feature_io_retry.py'
TEST_PATH='tests/test_thin_feature_io_retry.py'
_installed=False
_original_replace=os.replace


def retry_replace(source,destination,*,src_dir_fd=None,dst_dir_fd=None):
    began=time.perf_counter();attempt=0
    while True:
        try:
            kwargs={}
            if src_dir_fd is not None: kwargs['src_dir_fd']=src_dir_fd
            if dst_dir_fd is not None: kwargs['dst_dir_fd']=dst_dir_fd
            result=_original_replace(source,destination,**kwargs)
            if attempt:
                print(json.dumps(dict(io_retry='recovered',operation='atomic_replace',source=str(source),destination=str(destination),
                                      retries=attempt,elapsed_seconds=time.perf_counter()-began)),flush=True)
            return result
        except PermissionError as error:
            # Only observed Windows access/share locks are retryable, not arbitrary I/O errors.
            elapsed=time.perf_counter()-began
            if getattr(error,'winerror',None) not in [5,32,33] or elapsed>=5: raise
            delay=min(.025*2**min(attempt,5),.5,5-elapsed)
            print(json.dumps(dict(io_retry='waiting',operation='atomic_replace',winerror=error.winerror,
                                  destination=str(destination),delay_seconds=delay,attempt=attempt+1)),flush=True)
            time.sleep(delay);attempt+=1


def installed_hashes():
    return {path:campaign.digest(campaign.ROOT/path) for path in [WRAPPER_PATH,TEST_PATH]}


def install():
    global _installed
    if _installed: return
    original_hashes=campaign.scientific_hashes
    original_supervise=campaign.supervise
    original_check=campaign.check_manifest
    campaign.scientific_hashes=lambda:{**original_hashes(),**installed_hashes()}
    wrapped_hashes=campaign.scientific_hashes
    def check_manifest(path):
        manifest=campaign.read(path)
        if manifest['stage']=='confirmation' and WRAPPER_PATH not in manifest['sources']:
            raise RuntimeError('Confirmation seal does not include I/O wrapper')
        if manifest['stage']=='development' and WRAPPER_PATH not in manifest['sources']:
            # Audit the preserved original seal against its exact original inventory.
            campaign.scientific_hashes=original_hashes
            try: return original_check(path)
            finally: campaign.scientific_hashes=wrapped_hashes
        return original_check(path)
    campaign.check_manifest=check_manifest
    def supervise(command,*args,**kwargs):
        if len(command)>=3 and command[:3]==['-m','scripts.thin_feature_campaign','worker']:
            command=['-m','scripts.thin_feature_io_retry','worker']+command[3:]
        return original_supervise(command,*args,**kwargs)
    campaign.supervise=supervise
    os.replace=retry_replace
    _installed=True


if __name__=='__main__':
    install();campaign.main()
