import math
import torch
import pytest

from scripts.gap_training_controls import train_segment as baseline_train
from scripts.gap_lr_controls import lr_at, train_segment, full_gap, make_fixed, fixed_metrics
from neural_geometry_scale.geometry import Scene
from neural_geometry_scale.study import read
from neural_geometry_scale.study import model_for


def test_cosine_endpoints_and_monotonicity():
    rates=[lr_at(i) for i in range(1,10001)]
    assert rates[0]==1e-4 and rates[-1]==1e-6
    assert all(a>=b for a,b in zip(rates,rates[1:]))
    assert all(lr_at(i,schedule='constant')==1e-4 for i in (1,3000,10000))
    assert math.isclose(rates[0]+rates[-1],rates[4999]+rates[5000])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_restored_adam_stream_and_split_schedule(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(2)
    c=dict(device=device,scene='parallel_plates',feature=.02,angle=0,model='siren',width=8,depth=2,seed=31,lr=1e-4,batch_size=32,policy='local')
    baseline_train(c,None,2,tmp_path/'base')
    initial=tmp_path/'base/checkpoint.pt'
    baseline_train(c,initial,8,tmp_path/'reference')
    train_segment(c,initial,8,tmp_path/'constant','constant',2)
    train_segment(c,initial,8,tmp_path/'cosine','cosine',2)
    train_segment(c,initial,5,tmp_path/'first','cosine',2)
    train_segment(c,tmp_path/'first/checkpoint.pt',8,tmp_path/'second','cosine',2)
    def load(name):
        return torch.load(tmp_path/name/'checkpoint.pt',map_location='cpu',weights_only=True)
    for a,b in [(load('reference'),load('constant')),(load('cosine'),load('second'))]:
        assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
        assert torch.equal(a['sample_rng'],b['sample_rng'])
        for k in a['optimizer']['state']:
            assert all(torch.equal(a['optimizer']['state'][k][v],b['optimizer']['state'][k][v]) for v in a['optimizer']['state'][k])
    assert read(tmp_path/'constant/training.json')['stream_sha256']==read(tmp_path/'cosine/training.json')['stream_sha256']


def test_full_footprint_catches_outer_errors_and_partitions_margins():
    scene=Scene('parallel_plates',.02)
    good=full_gap(scene.sdf,scene,'cpu',count=4096)
    assert good['regions']['whole_gap']['all_margins']['errors']==0
    def bad(p):
        return torch.where((p[:,:2].abs()>.48).any(-1,keepdim=True),-scene.sdf(p),scene.sdf(p))
    r=full_gap(bad,scene,'cpu',count=4096)['regions']
    assert r['central_footprint']['all_margins']['errors']==0
    assert r['outer_footprint']['all_margins']['sign_accuracy']==0
    for region in r.values():
        assert sum(v['samples'] for k,v in region.items() if k!='all_margins')==region['all_margins']['samples']


def test_fixed_dataset_and_analytic_fit():
    c=dict(scene='parallel_plates',feature=.02,angle=0,seed=31,policy='local')
    a,b=make_fixed(c),make_fixed(c)
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert (a['target'][a['groups']==1]>0).all()
    assert (a['target'][a['groups']==2]<0).all()
    assert fixed_metrics(Scene('parallel_plates',.02).sdf,a,'cpu')['fit_gate']


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fixed_training_is_exact_full_batch_without_rng_consumption(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    c=dict(device=device,scene='parallel_plates',feature=.02,angle=0,model='siren',width=8,depth=2,seed=31,lr=1e-4,batch_size=32,policy='local')
    baseline_train(c,None,2,tmp_path/'base')
    data=make_fixed(c)
    initial=tmp_path/'base/checkpoint.pt'
    saved=torch.load(initial,map_location=device,weights_only=True)
    model=model_for(c,torch.device(device));model.load_state_dict(saved['model'])
    opt=torch.optim.Adam(model.parameters(),lr=1e-4);opt.load_state_dict(saved['optimizer'])
    for i in [1,2]:
        opt.param_groups[0]['lr']=lr_at(i)
        opt.zero_grad(set_to_none=True)
        ((model(data['points'].to(device))-data['target'].to(device))**2).mean().backward()
        opt.step()
    train_segment(c,initial,4,tmp_path/'fixed','cosine',2,data)
    actual=torch.load(tmp_path/'fixed/checkpoint.pt',map_location=device,weights_only=True)
    assert all(torch.equal(model.state_dict()[k],actual['model'][k]) for k in actual['model'])
    assert torch.equal(saved['sample_rng'],actual['sample_rng'])
