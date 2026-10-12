import copy
import numpy as np
import pytest
import torch
from neural_geometry_scale.geometry import Scene
from neural_geometry_scale.study import model_for
from scripts.gap_density_controls import nested_data
from scripts.gap_confirmation import same
from scripts.gap_reserved_confirmation import pool,full_gap,random_schedule,ARMS,train,selection_stream


def config(angle=0,feature=.02):
    return dict(scene='parallel_plates',feature=feature,angle=angle,policy='local',seed=13,model='siren',width=8,depth=2,device='cpu',lr=1e-4,batch_size=32)


def test_generalized_pool_matches_legacy_exactly():
    torch.set_num_threads(2)
    assert same(pool(config()),nested_data(config()))


def test_unreserved_rotation_and_local_sampling():
    c=config(11,.04); data=pool(c); scene=Scene(c['scene'],c['feature'],c['angle']); local=data['points']@scene.rotation(data['points'])
    gap=local[data['groups']==1]; inside=local[data['groups']==2]
    assert gap[:,2].abs().max()<=.020001 and gap[:,:2].abs().max()<=.600001
    assert inside[:,2].abs().min()>=.019999 and inside[:,2].abs().max()<=.030001
    reference=Scene(c['scene'],c['feature'],0).sdf(local)
    assert torch.allclose(reference,data['target'],atol=2e-7,rtol=2e-5)
    assert torch.bincount(data['groups']).tolist()==[32768,16384,16384]


def test_full_gap_all_width_and_rotation():
    scene=Scene('parallel_plates',.04,11)
    class Oracle:
        def __call__(self,p): return scene.sdf(p)
    m=full_gap(Oracle(),scene,'cpu',1007,4096)
    whole=m['regions']['whole_gap']; assert whole['all_margins']['errors']==0
    assert whole['margin_0.011_0.0200001']['samples']>0
    assert sum(v['samples'] for k,v in whole.items() if k!='all_margins')==whole['all_margins']['samples']


def test_random_schedule_unstratified_unique_repeatable(tmp_path):
    a=random_schedule(tmp_path/'a.npy',981,8); b=random_schedule(tmp_path/'b.npy',981,8)
    x=np.load(tmp_path/'a.npy'); assert a['sha256']==b['sha256'] and x.shape==(8,16384)
    assert all(len(np.unique(row))==16384 and row.min()>=0 and row.max()<65536 for row in x)
    assert a['minimum_group_counts']!=[8192,4096,4096]
    assert a['minimum_group_counts']!=a['maximum_group_counts']


@pytest.mark.parametrize('name',['dense','canonical','static','random'])
def test_recovery_resume_noncycle_preserves_model_adam_rng(tmp_path,name):
    torch.set_num_threads(2); c=config(); model=model_for(c,torch.device('cpu')); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    model(torch.zeros(2,3)).square().mean().backward(); opt.step()
    for s in opt.state.values(): s['step'].fill_(20000)
    rng=torch.Generator().manual_seed(81).get_state(); parent=tmp_path/'parent.pt'
    torch.save(dict(config=c,step=20000,model=model.state_dict(),optimizer=opt.state_dict(),sample_rng=rng),parent)
    data=pool(c); random_schedule(tmp_path/'plan.npy',181,7); plan=np.load(tmp_path/'plan.npy',mmap_mode='r')
    arm=next(a for a in ARMS if a['name']==name)
    train(c,parent,20007,tmp_path/'whole',data,arm,plan)
    train(c,parent,20003,tmp_path/'a',data,arm,plan); train(c,tmp_path/'a/checkpoint.pt',20007,tmp_path/'b',data,arm,plan)
    whole=torch.load(tmp_path/'whole/checkpoint.pt',weights_only=True); resumed=torch.load(tmp_path/'b/checkpoint.pt',weights_only=True)
    assert all(same(whole[k],resumed[k]) for k in ['model','optimizer','sample_rng'])
    assert torch.equal(whole['sample_rng'],rng)
    assert selection_stream(data,arm,20000,20007,plan)==selection_stream(data,arm,20000,20007,plan)
