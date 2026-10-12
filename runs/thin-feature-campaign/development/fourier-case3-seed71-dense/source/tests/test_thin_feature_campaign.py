import copy
import subprocess
import sys
import time
import pytest
import torch
from scripts import thin_feature_campaign as c


def plate(): return dict(scene='parallel_plates',feature=.04,angle=11)
def stack(): return dict(scene='multiscale_plate_stack',angle=11,centers=[[0,0,-.3],[0,0,-.1],[0,0,.2]],half_extents=[[.4,.4,.04],[.4,.4,.07],[.4,.4,.1]],full_gaps=[.09,.13])


def test_legacy_pool_exact_and_rotation():
    data=c.pool(plate(),13); old=c.legacy_pool(dict(plate(),seed=13,policy='global_narrow'))
    for key in old: assert torch.equal(data[key],old[key])
    scene=c.FamilyScene(plate()); local=data['points']@scene.rotation(data['points'])
    assert local[data['groups']==1,2].abs().max()<.020001


@pytest.mark.parametrize('case',[dict(scene='mixed_thickness',feature=.04,angle=11),stack()])
def test_gap_and_component_coverage(case):
    scene=c.FamilyScene(case); gap,gi,inside,ci=c.sample_regions(scene,4096,torch.Generator().manual_seed(9))
    assert (scene.sdf(gap@scene.rotation(gap).T)>0).all()
    assert (scene.sdf(inside@scene.rotation(inside).T)<0).all()
    assert set(gi.tolist())==set(range(len(scene.gaps())))
    centers,half=scene.boxes()
    for i in range(len(centers)):
        assert (torch.abs(inside[ci==i]-centers[i])<=half[i]+1e-7).all()
    assert set(ci.tolist())==set(range(len(centers)))


def test_fourier_wiring_hash_checkpoint(tmp_path):
    model=c.factory(c.CONFIGS[1],13,'cpu'); again=c.factory(c.CONFIGS[1],13,'cpu')
    assert model.basis.shape==(3,64) and torch.equal(model.basis,again.basis)
    assert model.basis.std()>5
    opt=torch.optim.Adam(model.parameters());g=torch.Generator().manual_seed(14)
    hashes=c.save_checkpoint(tmp_path/'state.pt',model,opt,g,0,c.CONFIGS[1])
    saved=torch.load(tmp_path/'state.pt',weights_only=True)
    assert c.state_hash(saved['model'])==hashes['model'] and c.tensor_hash(saved['sample_rng'])==hashes['sample_rng']


def row(step,passed=True):
    return dict(step=step,primary_gate=passed,field={'all_probe_signs_correct':passed},
                independent={'gap':{'rmse':.001}},full_gap={'gap':{'sign_accuracy':1.}},
                evaluations={str(r):dict(mesh_present=True,reference_resolved=True,symmetric_chamfer=.01) for r in [192,384]})


def test_persistence_and_nonfinite_failclosed():
    assert c.persistent([row(23000),row(26000),row(30000)])==dict(persistent=True,onset=23000)
    assert c.persistent([row(23000,False),row(26000),row(30000)])['onset']==26000
    assert not c.persistent([row(30000)])['persistent']
    invalid=row(26000);invalid['evaluations']['384']['symmetric_chamfer']=float('nan')
    assert not c.persistent([invalid,row(30000)])['persistent']
    invalid=row(26000);invalid['evaluations']['384']['reference_resolved']=False
    assert not c.persistent([invalid,row(30000)])['persistent']


def test_topology_expectations():
    assert c.FamilyScene(plate()).expected==dict(components=2,euler=4,closed_edges=True)
    assert c.FamilyScene(stack()).expected==dict(components=3,euler=6,closed_edges=True)


def test_budget_failures_charged_and_cap(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'DEST',tmp_path);monkeypatch.setattr(c,'LIMITS',dict(development=.9,confirmation=10,qa=10,global_seconds=20))
    c.write(tmp_path/'budget.json',dict(attempts=[]))
    # allowance reserves five seconds for durable closure; use a larger cap.
    c.LIMITS['development']=6
    assert not c.supervise(['-c','raise RuntimeError("expected failure")'],'development')
    ledger=c.read(tmp_path/'budget.json');assert ledger['attempts'][0]['charged_seconds']>0 and ledger['attempts'][0]['exit_code']!=0
    assert not c.supervise(['-c','import time; time.sleep(30)'],'development')
    capped=c.read(tmp_path/'budget.json')['attempts'][-1]
    assert capped['timed_out'] and capped['charged_seconds']>0
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=sys.platform!='win32')
    c.kill_tree(process); assert process.poll() is not None


def test_source_tamper_before_exposure(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'scientific_hashes',lambda:{'source':'new'})
    c.write(tmp_path/'seal.json',dict(sources={'source':'old'}))
    (tmp_path/'seal.json.sha256').write_text(c.digest(tmp_path/'seal.json'))
    with pytest.raises(RuntimeError,match='source drift'): c.check_manifest(tmp_path/'seal.json')


def test_seal_tamper_before_exposure(tmp_path):
    c.write(tmp_path/'seal.json',dict(tasks=[]));(tmp_path/'seal.json.sha256').write_text(c.digest(tmp_path/'seal.json'))
    c.write(tmp_path/'seal.json',dict(tasks=[dict(case='changed')]))
    with pytest.raises(RuntimeError,match='seal drift'): c.check_manifest(tmp_path/'seal.json')


@pytest.mark.parametrize('case',[dict(scene='mixed_thickness',feature=.04,angle=11),stack()])
def test_oracle_topology_real_mesh(case):
    torch.set_num_threads(2); scene=c.FamilyScene(case)
    class Oracle:
        def __call__(self,p): return scene.sdf(p)
    metrics=c.geometry(Oracle(),scene,'cpu',64,count=1024)
    assert metrics['reference_resolved'] and metrics['structure_match'] and metrics['mesh_present']


def test_cuda_unreserved_train_checkpoint_eval(tmp_path):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    c.deterministic();scene=c.FamilyScene(plate()); model=c.factory(c.CONFIGS[1],13,'cuda')
    opt=torch.optim.Adam(model.parameters(),lr=1e-4);g=torch.Generator(device='cuda').manual_seed(14)
    p,t,_=c.batch(scene,2048,'global_narrow',g,'cuda');opt.zero_grad();loss=(model(p)-t).square().mean();loss.backward();opt.step()
    saved=c.save_checkpoint(tmp_path/'cuda.pt',model,opt,g,1,c.CONFIGS[1]);assert saved['optimizer']
    r=c.field_metrics(model,scene,'cuda',9901,128);assert math_isfinite(r['uniform_sdf_mae'])
    geometry=c.geometry(model,scene,'cuda',32,count=128);assert 'mesh_present' in geometry


def test_independent_arm_prefix_pairing_small_cpu():
    torch.set_num_threads(2);cfg=dict(model='siren',width=8,depth=2);scene=c.FamilyScene(plate());hashes=[]
    for arm in c.ARMS:
        model=c.factory(cfg,13,'cpu');opt=torch.optim.Adam(model.parameters(),lr=1e-4);g=torch.Generator().manual_seed(1013)
        # Random preparation uses its own namespace and cannot perturb prefix.
        if arm=='random': torch.randperm(128,generator=torch.Generator().manual_seed(9001))
        for _ in range(5):
            p,t,_=c.batch(scene,32,'global_narrow',g);opt.zero_grad();loss=(model(p)-t).square().mean();loss.backward();opt.step()
        hashes.append((c.state_hash(model.state_dict()),c.state_hash(opt.state_dict()),c.tensor_hash(g.get_state())))
    assert len(set(hashes))==1


def math_isfinite(value):
    import math
    return math.isfinite(value)


def test_report_source_has_no_evaluation_calls():
    raw=(c.ROOT/'scripts/report_thin_feature_campaign.py').read_text()
    assert 'FamilyScene(' not in raw and 'factory(' not in raw and 'geometry(' not in raw
