import copy
import math
import pytest
import torch

from scripts.gap_confirmation import (ARMS, allowance, same, schedule, stream_hash, train, qualify, valid_row, verify_manifest)
from scripts.gap_density_controls import nested_data, full_gradient
from scripts.gap_lr_controls import lr_at
from neural_geometry_scale.study import model_for


def config():
    return dict(device='cpu',scene='parallel_plates',feature=.02,angle=0,model='siren',width=8,depth=2,seed=31,lr=1e-4,batch_size=32,policy='local')


@pytest.mark.parametrize('name',['canonical','phase-1','shuffle'])
def test_balance_and_coverage_and_rng(name):
    torch.manual_seed(999); before=torch.get_rng_state().clone()
    order=schedule(name,2); assert torch.equal(before,torch.get_rng_state())
    assert torch.equal(order,schedule(name,2))
    assert all(torch.equal(x.sort().values,torch.arange(4)) for x in order.reshape(-1,4))
    assert torch.bincount(order).tolist()==[2500]*4
    d=nested_data(config())
    for j in range(4): assert torch.bincount(d['groups'][j*16384:(j+1)*16384]).tolist()==[8192,4096,4096]


def test_gradient_is_ordinary_mse():
    torch.set_num_threads(2); c=config(); a=model_for(c,torch.device('cpu')); b=copy.deepcopy(a)
    gen=torch.Generator().manual_seed(777); x=torch.rand(73,3,generator=gen); y=torch.rand(73,1,generator=gen)
    direct=(a(x)-y).square().mean(); direct.backward(); chunked=full_gradient(b,x,y,17)
    assert torch.allclose(direct,chunked,atol=1e-7,rtol=1e-6)
    assert all(torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5) for p,q in zip(a.parameters(),b.parameters()))


@pytest.mark.parametrize('name',['canonical','phase-1','shuffle'])
def test_noncycle_resume_exact_model_adam_rng(tmp_path,name):
    torch.set_num_threads(2); c=config(); model=model_for(c,torch.device('cpu')); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    x=torch.zeros(3,3); model(x).square().mean().backward(); opt.step()
    for s in opt.state.values(): s['step'].fill_(20000)
    checkpoint=tmp_path/'parent.pt'; rng=torch.Generator().manual_seed(9876).get_state()
    torch.save(dict(config=c,step=20000,model=model.state_dict(),optimizer=opt.state_dict(),sample_rng=rng),checkpoint)
    d=nested_data(c); arm=next(a for a in ARMS if a['name']==name); order=schedule(name,0)
    train(c,checkpoint,20007,tmp_path/'whole',d,arm,order)
    train(c,checkpoint,20003,tmp_path/'first',d,arm,order)
    train(c,tmp_path/'first/checkpoint.pt',20007,tmp_path/'second',d,arm,order)
    whole=torch.load(tmp_path/'whole/checkpoint.pt',weights_only=True); resumed=torch.load(tmp_path/'second/checkpoint.pt',weights_only=True)
    assert all(same(whole[k],resumed[k]) for k in ['model','optimizer','sample_rng'])
    assert torch.equal(rng,whole['sample_rng'])
    # Direct independent update-loop reference checks schedule position and MSE.
    for j in range(7):
        start=int(order[j])*16384; opt.param_groups[0]['lr']=lr_at(j+1); opt.zero_grad(set_to_none=True)
        (model(d['points'][start:start+16384])-d['target'][start:start+16384]).square().mean().backward(); opt.step()
    assert same(model.state_dict(),whole['model']) and same(opt.state_dict(),whole['optimizer'])


def test_stream_repeatability_and_order_difference():
    d=nested_data(config()); arm=ARMS[1]
    hashes=[stream_hash(d,arm,schedule(n,0),20000,20013) for n in ['canonical','phase-1','shuffle']]
    assert len(set(hashes))==3
    assert hashes[0]==stream_hash(d,arm,schedule('canonical',0),20000,20013)


def row(policy,seed,name,end):
    return dict(metadata=dict(config=dict(policy=policy,seed=seed),step=end),training=dict(arm=dict(name=name),seconds=10 if name=='dense' else 3),
        primary_gate=True,independent_fit=dict(gap=dict(rmse=.001)),full_gap=dict(regions=dict(whole_gap=dict(all_margins=dict(sign_accuracy=1.)))),
        evaluations={r:dict(mesh_present=True,reference_resolved=True,symmetric_chamfer=.001) for r in ['192','384']})


def fixtures():
    rr=[row(p,s,a['name'],end) for p in ['global_narrow','local'] for s in [31,32,33] for a in ARMS for end in [26000,30000]]
    exact=[dict(model_exact=True,optimizer_exact=True,sample_rng_exact=True) for _ in range(24)]
    return rr,exact


def test_qualification_pass_and_strict_failures():
    rr,exact=fixtures(); assert qualify(rr,exact)['gate_pass']
    assert not qualify(rr[:-1],exact)['gate_pass']
    bad=copy.deepcopy(rr); bad[0]['evaluations']['384']['mesh_present']=False
    assert not qualify(bad,exact)['gate_pass']
    bad=copy.deepcopy(rr); bad[0]['evaluations']['384']['reference_resolved']=False
    assert not qualify(bad,exact)['gate_pass']
    for v in [None,float('inf'),float('nan')]:
        bad=copy.deepcopy(rr); bad[0]['independent_fit']['gap']['rmse']=v
        assert not qualify(bad,exact)['gate_pass']
    bad=copy.deepcopy(rr); bad[2]['primary_gate']=False
    assert not qualify(bad,exact)['gate_pass']
    bad=copy.deepcopy(exact); bad[0]['model_exact']=False
    assert not qualify(rr,bad)['gate_pass']
    assert not qualify(rr+rr[:1],exact)['gate_pass']


def test_budget_closed_and_exhausted():
    assert allowance(dict(limit_seconds=7200,attempts=[dict(finished=True,charged_seconds=7200)]))==0
    with pytest.raises(RuntimeError,match='Unclosed'): allowance(dict(limit_seconds=7200,attempts=[dict(finished=False)]))


def test_source_mismatch_rejected():
    with pytest.raises(RuntimeError,match='source drift'): verify_manifest(dict(source_hashes={}))


def test_unknown_order_rejected():
    with pytest.raises(ValueError): schedule('retuned',0)
