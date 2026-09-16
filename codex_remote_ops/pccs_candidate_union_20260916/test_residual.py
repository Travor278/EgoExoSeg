import types
import torch
from residual_pool import install_residual_pool
from candidate_ops import probability_weighted_pool
class Sampler:
    def forward(self,f,m,original_dtype,return_dtype):return [torch.rand(len(mask),1,f.shape[-1]).to(return_dtype) for mask in m]
def model():return types.SimpleNamespace(visual_anchor_expert=types.SimpleNamespace(region_sampler=Sampler()),fusion_expert=types.SimpleNamespace(region_sampler=Sampler()))
f=torch.tensor([[[1.],[2.],[3.],[4.]]]);m=[torch.tensor([[[0.,0.],[.25,.75]]])]
a=model();torch.manual_seed(7);old=a.visual_anchor_expert.region_sampler.forward(f,m,torch.float32,torch.float32);state=torch.get_rng_state()
install_residual_pool(a,0.);torch.manual_seed(7);new=a.visual_anchor_expert.region_sampler.weighted_forward(f,m,torch.float32,torch.float32);assert torch.equal(old[0],new[0]) and torch.equal(state,torch.get_rng_state())
b=model();install_residual_pool(b,.1);torch.manual_seed(7);v=b.visual_anchor_expert.region_sampler.weighted_forward(f,m,torch.float32,torch.float32);w=probability_weighted_pool(f,m,torch.float32);torch.testing.assert_close(v[0],old[0]*.9+w[0]*.1);assert torch.equal(state,torch.get_rng_state())
from metric_adapter import Adapter
torch.manual_seed(8);adapter=Adapter();x=torch.nn.functional.normalize(torch.randn(3,768),dim=-1);torch.testing.assert_close(adapter(x),x,atol=1e-6,rtol=1e-6)
print('RESIDUAL_RNG_AND_ADAPTER_ZERO_TESTS_PASS')
