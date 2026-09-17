"""Native DINOv3 soft-cycle and local-confound evidence. No O-MaMa modules."""
import types
import numpy as np

def annulus(mask,scale):
    y,x=np.where(mask);out=np.zeros_like(mask,dtype=bool)
    if not len(x):return out
    cy=(y.min()+y.max())/2;cx=(x.min()+x.max())/2;hy=max(1.,(y.max()-y.min()+1)*scale/2);hx=max(1.,(x.max()-x.min()+1)*scale/2)
    out[max(0,int(cy-hy)):min(mask.shape[0],int(np.ceil(cy+hy))+1),max(0,int(cx-hx)):min(mask.shape[1],int(np.ceil(cx+hx))+1)]=True
    return out & ~mask

def extract(result,predicted_masks,source_masks):
    import torch
    import torch.nn.functional as F
    q=result['features_query'].float();t=result['features_target'].float();qh,qw=q.shape[-2:];th,tw=t.shape[-2:];qf=q.flatten(1).T;tf=t.flatten(1).T
    affinity=qf@tf.T
    # Bidirectional soft correspondence, computed once for all object candidates.
    qt=(affinity/.07).softmax(1);tq=(affinity.T/.07).softmax(1)
    def mask(x,h,w):return F.interpolate(torch.as_tensor(np.asarray(x).copy(),device=q.device).float()[None,None],size=(h,w),mode='area')[0,0]
    def norm(x):return F.normalize(x,dim=0)
    def pool(f,w):return norm((f*w[:,None]).sum(0)/w.sum().clamp_min(1e-8))
    out=[]
    for pm,sm in zip(predicted_masks,source_masks):
        qa=mask(pm,qh,qw);ta=mask(sm,th,tw);a=qa.flatten();b=ta.flatten();valid=bool(a.sum()>0 and b.sum()>0)
        oq=pool(qf,a);ot=pool(tf,b);obj=(oq*ot).sum()
        # Source -> candidate -> original source-mask mass, plus reverse precision.
        into_source=qt@b;into_candidate=tq@a
        fwd=(b*into_candidate).sum()/b.sum().clamp_min(1e-8);back=(a*into_source).sum()/a.sum().clamp_min(1e-8)
        cycle=(b*(tq@(a*into_source))).sum()/b.sum().clamp_min(1e-8)
        soft_f=2*fwd*back/(fwd+back).clamp_min(1e-8)
        base=dict(object_cos=float(obj),forward_mass=float(fwd),backward_mass=float(back),cycle_mass=float(cycle),soft_cycle=float(soft_f),q_area=float(a.mean()),s_area=float(b.mean()),valid=valid)
        for scale,tag in ((1.5,'15'),(2.,'20')):
            qr=annulus((qa>.05).cpu().numpy(),scale);tr=annulus((ta>.05).cpu().numpy(),scale)
            for wrong,prefix in ((False,'ctx'+tag),(True,'wrong'+tag)):
                sr=np.roll(tr,(th//2,tw//2),(0,1)) & ~(ta>.05).cpu().numpy() if wrong else tr
                ar=torch.as_tensor(qr.reshape(-1),device=q.device).float();br=torch.as_tensor(sr.reshape(-1),device=q.device).float();ok=bool(ar.sum()>=2 and br.sum()>=2 and valid)
                cq=pool(qf,ar);ct=pool(tf,br)
                # The neighboring background is a local confound, not a veto signal.
                context=(cq*ct).sum();cross_bg=.5*((oq*ct).sum()+(ot*cq).sum());contrast=obj-cross_bg
                residual=(norm(oq-.5*cq)*norm(ot-.5*ct)).sum()
                # Is true object-cycle support stronger than context leakage?
                bg_back=(ar*into_source).sum()/ar.sum().clamp_min(1e-8)
                bg_fwd=(br*into_candidate).sum()/br.sum().clamp_min(1e-8)
                fg_contrast=soft_f-(bg_back+bg_fwd)/2
                values={'valid':float(ok),'context_cos':float(context),'cross_bg':float(cross_bg),'contrast':float(contrast),'residual_cos':float(residual),'cycle_contrast':float(fg_contrast)}
                base.update({prefix+'_'+k:v if ok else 0. for k,v in values.items()})
        assert all(np.isfinite(v) for v in base.values());out.append(base)
    return out

class Capture:
    def __init__(self,model):
        self.enabled=True;self.records={};self.source=None
        for name,expert in [('va',model.visual_anchor_expert),('fusion',model.fusion_expert)]:
            m=expert.sparse_correspondence_backward;original=m.forward
            def wrap(this,*args,_original=original,_name=name,**kwargs):
                if not self.enabled:return _original(*args,**kwargs)
                wanted=kwargs.get('return_features',False);kwargs['return_features']=True;result=_original(*args,**kwargs)
                self.records[_name].append(extract(result,kwargs['mask_query'],self.source))
                if not wanted:result.pop('features_query');result.pop('features_target')
                return result
            m.forward=types.MethodType(wrap,m)
    def begin(self,source):self.source=source.detach().cpu().numpy();self.records={'va':[],'fusion':[]}
    def finish(self):
        assert len(self.records['va'])==2 and len(self.records['fusion'])==1
        v,a=self.records['va'];f=self.records['fusion'][0]
        return [{e:r[j] for e,r in [('visual',v),('anchor',a),('fusion',f)]} for j in range(len(v))]
