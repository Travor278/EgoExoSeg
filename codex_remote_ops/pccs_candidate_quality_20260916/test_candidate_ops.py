import numpy as np
from candidate_ops import select_reliable,stable_pair_seed
def test_numpy():
    sim=np.full((9,9),-.2,dtype=np.float32);np.fill_diagonal(sim,.9)
    chosen,info=select_reliable(sim,np.ones(9,bool),(3,3),(3,3),max_points=3,separation=2)
    assert [c['source'] for c in chosen]==[0,2,6]
    assert len({c['target'] for c in chosen})==3
    chosen,_=select_reliable(sim,np.zeros(9,bool),(3,3),(3,3));assert chosen==[]
    tied=np.ones((9,9),np.float32);chosen,_=select_reliable(tied,np.ones(9,bool),(3,3),(3,3));assert chosen==[]
    sim[0,1]=.899;chosen,_=select_reliable(sim,np.eye(3,dtype=bool).reshape(-1),(3,3),(3,3));assert all(c['source']!=0 for c in chosen)
    try:select_reliable(np.full((9,9),np.nan),np.ones(9,bool),(3,3),(3,3))
    except ValueError:pass
    else:raise AssertionError('NaNs must fail')
    assert stable_pair_seed('a')==stable_pair_seed('a') and stable_pair_seed('a')!=stable_pair_seed('b')
    print('NUMPY_TESTS_PASS')
def test_torch():
    import torch
    from candidate_ops import probability_weighted_pool
    f=torch.tensor([[[1.],[2.],[3.],[4.]]]);m=torch.tensor([[[0.,0.],[.25,.75]],[[0.,0.],[0.,0.]]])
    out=probability_weighted_pool(f,[m],torch.float32)[0]
    assert out.shape==(2,1,1);torch.testing.assert_close(out[:,0,0],torch.tensor([3.75,0.]))
    out=probability_weighted_pool(f,[torch.ones(1,2,2)],torch.float32)[0];torch.testing.assert_close(out.flatten(),torch.tensor([2.5]))
    print('TORCH_WEIGHTED_POOL_TESTS_PASS')
if __name__=='__main__':
    import sys
    test_numpy()
    if '--numpy-only' not in sys.argv:test_torch()
