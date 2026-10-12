import copy
import torch
import pytest
from scripts.gap_sampling_controls import selections, train, stream_hash
from scripts.gap_density_controls import nested_data
from scripts.gap_lr_controls import lr_at
from scripts.gap_training_controls import train_segment
from neural_geometry_scale.study import model_for


def config():
    return dict(device='cpu',scene='parallel_plates',feature=.02,angle=0,model='siren',width=8,
                depth=2,seed=31,lr=1e-4,batch_size=32,policy='local')


@pytest.mark.parametrize('n',[4096,8192,16384,32768,65536])
def test_balanced_exhaustive_cycle(n):
    d=nested_data(config())
    for phase in [0,1]:
        choices=selections(n,phase)
        assert torch.equal(torch.cat(choices).sort().values,torch.arange(65536))
        for i in choices:
            assert torch.bincount(d['groups'][i]).tolist()==[n//2,n//4,n//4]


def test_stream_changes_and_is_repeatable():
    d=nested_data(config())
    assert stream_hash(d,16384,0,20000,20004)==stream_hash(d,16384,0,20000,20004)
    assert stream_hash(d,16384,0,20000,20004)!=stream_hash(d,16384,1,20000,20004)


def test_resume_preserves_adam_and_cycle_position(tmp_path):
    torch.set_num_threads(2)
    c=config(); train_segment(c,None,2,tmp_path/'base')
    checkpoint=tmp_path/'base/checkpoint.pt'
    saved=torch.load(checkpoint,weights_only=True); saved['step']=20000
    for s in saved['optimizer']['state'].values(): s['step'].fill_(20000)
    torch.save(saved,checkpoint)
    d=nested_data(c); arm=dict(name='cycle',points=16384,phase=1)
    model=model_for(c,torch.device('cpu')); model.load_state_dict(saved['model'])
    opt=torch.optim.Adam(model.parameters(),lr=1e-4); opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    # Independent direct monolithic ordinary-MSE reference, crossing the cycle boundary.
    for j in range(1,6):
        start=((j-1+1)%4)*16384
        opt.param_groups[0]['lr']=lr_at(j); opt.zero_grad(set_to_none=True)
        (model(d['points'][start:start+16384])-d['target'][start:start+16384]).square().mean().backward(); opt.step()
    train(c,checkpoint,20002,tmp_path/'a',d,arm)
    train(c,tmp_path/'a/checkpoint.pt',20005,tmp_path/'b',d,arm)
    actual=torch.load(tmp_path/'b/checkpoint.pt',weights_only=True)
    assert all(torch.equal(v,actual['model'][k]) for k,v in model.state_dict().items())
    assert torch.equal(actual['sample_rng'],saved['sample_rng'])
    for k,s in actual['optimizer']['state'].items():
        assert int(s['step'])==20005
        assert all(torch.equal(v,opt.state_dict()['state'][k][name]) for name,v in s.items())
