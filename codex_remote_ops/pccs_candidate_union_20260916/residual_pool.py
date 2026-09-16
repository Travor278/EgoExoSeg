import types
from candidate_ops import probability_weighted_pool
def install_residual_pool(model,alpha):
    if not 0<=alpha<=1:raise ValueError('Invalid mixture')
    for expert in (model.visual_anchor_expert,model.fusion_expert):
        sampler=expert.region_sampler
        def mixed(self,features,masks,original_dtype,return_dtype):
            old=self.forward(features,masks,original_dtype,return_dtype)
            weighted=probability_weighted_pool(features,masks,return_dtype)
            return [None if a is None else ((1-alpha)*a.float()+alpha*b.float()).to(return_dtype) for a,b in zip(old,weighted)]
        sampler.weighted_forward=types.MethodType(mixed,sampler);sampler.weighted_enabled=True
