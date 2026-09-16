import time
import numpy as np
from roi_views import crop_view,encode_views
MODES={'local15':(1.5,False),'local20':(2.,False),'object20':(2.,True)}
KEYS=('valid','object_cos','forward_mass','backward_mass','cycle_mass','soft_cycle','log_forward','log_backward','log_cycle','source_scale','target_scale')
def pair_stats(q,t,qmask,tmask):
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        a=F.interpolate(torch.as_tensor(qmask.copy(),device=q.device).float()[None,None],size=q.shape[-2:],mode='area').flatten()
        b=F.interpolate(torch.as_tensor(tmask.copy(),device=q.device).float()[None,None],size=t.shape[-2:],mode='area').flatten()
        qf=q.flatten(1).T.float();tf=t.flatten(1).T.float();s=qf@tf.T;qt=(s/.07).softmax(1);tq=(s.T/.07).softmax(1)
        reverse=qt@b;forward=tq@a;u=(b*forward).sum()/b.sum().clamp_min(1e-8);v=(a*reverse).sum()/a.sum().clamp_min(1e-8);cy=(b*(tq@(a*reverse))).sum()/b.sum().clamp_min(1e-8)
        oq=F.normalize((qf*a[:,None]).sum(0),dim=0);ot=F.normalize((tf*b[:,None]).sum(0),dim=0)
        values=torch.stack([(oq*ot).sum(),u,v,cy,2*u*v/(u+v).clamp_min(1e-8),torch.log(u.clamp_min(1e-30))-torch.log(a.mean().clamp_min(1e-30)),torch.log(v.clamp_min(1e-30))-torch.log(b.mean().clamp_min(1e-30)),torch.log(cy.clamp_min(1e-30))-torch.log(a.mean().clamp_min(1e-30))-torch.log(b.mean().clamp_min(1e-30))]).cpu().numpy()
    return dict(zip(KEYS[1:9],map(float,values)))
def collect(matcher,source_image,target_image,source_mask,target_masks,batch_test=False):
    start=time.monotonic();views=[];indices={};metadata={}
    for mode,(scale,object_only) in MODES.items():
        source=crop_view(source_image,source_mask,scale,object_only)
        if source is None:continue
        indices[(mode,'source')]=len(views);views.append(source)
        for expert,mask in target_masks.items():
            view=crop_view(target_image,mask,scale,object_only)
            if view is not None:indices[(mode,expert)]=len(views);views.append(view)
    encoded=encode_views(matcher,views,4);parity=None
    if batch_test and views:
        import torch
        single=encode_views(matcher,views[:1],1)[0];diff=float((single-encoded[0]).abs().max());cos=float(torch.nn.functional.cosine_similarity(single,encoded[0],dim=0).mean());assert diff<.01 and cos>.999,('Batch feature discrepancy',diff,cos);parity={'max_abs':diff,'mean_cosine':cos}
    result={e:{} for e in target_masks}
    for e in target_masks:
        for mode in MODES:
            value={k:0. for k in KEYS}
            if (mode,e) in indices:
                qi=indices[(mode,e)];ti=indices[(mode,'source')];v=pair_stats(encoded[qi],encoded[ti],views[qi]['mask'],views[ti]['mask']);value.update(v,valid=1.,source_scale=views[ti]['effective_scale'],target_scale=views[qi]['effective_scale']);metadata[mode+'/'+e]={'source_box':views[ti]['box'],'target_box':views[qi]['box']}
            result[e].update({mode+'_'+k:v for k,v in value.items()})
    assert all(np.isfinite(v) for x in result.values() for v in x.values())
    return result,{'encoded_views':len(views),'elapsed_seconds':time.monotonic()-start,'geometry':metadata,'batch_single_parity':parity}
