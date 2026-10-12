import copy
import pytest
import torch

from neural_geometry_scale.geometry import Scene
from neural_geometry_scale.study import model_for
from scripts.gap_lr_controls import make_fixed, lr_at
from scripts.gap_training_controls import train_segment
from scripts.gap_density_controls import nested_data, prefix, overlap, stratified, full_gradient, train


def config(device='cpu'):
    return dict(device=device,scene='parallel_plates',feature=.02,angle=0,model='siren',width=8,
                depth=2,seed=31,lr=1e-4,batch_size=32,policy='local')


def test_nested_exact_prior_points_proportions_and_independence():
    torch.set_num_threads(2)
    c=config(); all_data=nested_data(c); old=make_fixed(c)
    assert all(torch.equal(old[k],all_data[k][:4096]) for k in old)
    for n in [4096,16384,65536]:
        small=nested_data(c,n)
        assert all(torch.equal(small[k],all_data[k][:n]) for k in small)
        assert torch.bincount(small['groups']).tolist()==[n//2,n//4,n//4]
    independent=nested_data(c,seed=61002)
    assert overlap(all_data['points'],independent['points'])==0
    assert overlap(all_data['points'],old['points'])==4096


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_chunk_gradient_matches_monolithic_with_unequal_tail(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(73)
    model=torch.nn.Sequential(torch.nn.Linear(3,12),torch.nn.Tanh(),torch.nn.Linear(12,1)).to(device).double()
    other=copy.deepcopy(model)
    p=torch.randn(103,3,device=device,dtype=torch.float64)
    t=torch.randn(103,1,device=device,dtype=torch.float64)
    expected=(model(p)-t).square().mean(); expected.backward()
    actual=full_gradient(other,p,t,17)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    for a,b in zip(model.parameters(),other.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-11,atol=1e-12)


def test_surface_distance_partition_and_analytic_error():
    d=nested_data(config(),4096)
    r=stratified(Scene('parallel_plates',.02).sdf,d,'cpu')
    for group in r.values():
        # No exact zero distances in this deterministic fixture.
        assert sum(v['samples'] for k,v in group.items() if k!='all')==group['all']['samples']
        assert group['all']['rmse']==0 and group['all']['sign_errors']==0


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_adam_resume_and_single_step_per_full_dataset(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    c=config(device); train_segment(c,None,2,tmp_path/'base')
    checkpoint=tmp_path/'base/checkpoint.pt'
    saved=torch.load(checkpoint,map_location=device,weights_only=True)
    saved['step']=20000
    for s in saved['optimizer']['state'].values(): s['step'].fill_(20000)
    torch.save(saved,checkpoint)
    data=nested_data(c,4096)
    model=model_for(c,torch.device(device)); model.load_state_dict(saved['model'])
    opt=torch.optim.Adam(model.parameters(),lr=1e-4); opt.load_state_dict(saved['optimizer'])
    for j in [1,2]:
        opt.param_groups[0]['lr']=lr_at(j); opt.zero_grad(set_to_none=True)
        full_gradient(model,data['points'].to(device),data['target'].to(device),777); opt.step()
    train(c,checkpoint,20002,tmp_path/'whole',data,777)
    train(c,checkpoint,20001,tmp_path/'first',data,777)
    train(c,tmp_path/'first/checkpoint.pt',20002,tmp_path/'second',data,777)
    a=torch.load(tmp_path/'whole/checkpoint.pt',map_location='cpu',weights_only=True)
    b=torch.load(tmp_path/'second/checkpoint.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(a['model'][k],v.cpu()) for k,v in model.state_dict().items())
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert torch.equal(a['sample_rng'],saved['sample_rng'].cpu())
    for k,state in a['optimizer']['state'].items():
        assert int(state['step'])==20002
        assert all(torch.equal(v,b['optimizer']['state'][k][name]) for name,v in state.items())
