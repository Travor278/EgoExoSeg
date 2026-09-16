"""Context evidence for native PCCS cycle votes; no target annotations or O-MaMa head."""
import numpy as np

DEFAULTS=dict(scale=1.75,min_similarity=.4,min_margin=.02,min_matches=4,reliability_min=.2,conflict_max=.25,strength=.5)

def ring(mask,scale=1.75):
    mask=np.asarray(mask,dtype=bool);out=np.zeros_like(mask);ys,xs=np.where(mask)
    if not len(xs):return out
    cy=(ys.min()+ys.max())/2;cx=(xs.min()+xs.max())/2;hy=max(1.,(ys.max()-ys.min()+1)*scale/2);hx=max(1.,(xs.max()-xs.min()+1)*scale/2)
    out[max(0,int(np.floor(cy-hy))):min(mask.shape[0],int(np.ceil(cy+hy))+1),max(0,int(np.floor(cx-hx))):min(mask.shape[1],int(np.ceil(cx+hx))+1)]=True
    return out & ~mask

def evidence_from_matches(qmask,tmask,qbest,tbest,qvalid,tvalid,scale=1.75,wrong=False,min_matches=4):
    """q is prediction view, t is original source; matches address full native grids."""
    qring=ring(qmask,scale).reshape(-1);tring=ring(tmask,scale)
    if wrong:tring=np.roll(tring,(tring.shape[0]//2,tring.shape[1]//2),(0,1)) & ~np.asarray(tmask,bool)
    tring=tring.reshape(-1);qfg=np.asarray(qmask,bool).reshape(-1);tfg=np.asarray(tmask,bool).reshape(-1)
    qa=qring & qvalid & ~tfg[qbest];ta=tring & tvalid & ~qfg[tbest]
    nq=int(qa.sum());nt=int(ta.sum());total=nq+nt
    support=(int(tring[qbest[qa]].sum())+int(qring[tbest[ta]].sum()))/max(total,1)
    coverage=.5*(nq/max(int(qring.sum()),1)+nt/max(int(tring.sum()),1))
    reliability=min(1.,total/8)*np.sqrt(coverage) if total>=min_matches and nq and nt else 0.
    return dict(support=float(support),reliability=float(reliability),matches=total,query_matches=nq,source_matches=nt,query_context_patches=int(qring.sum()),source_context_patches=int(tring.sum()))

def recheck_fusion(info,cfg):
    if not cfg.get('review') or cfg.get('strength',0)==0:return False
    x=info.get('fusion',{});return x.get('reliability',0)>=cfg.get('reliability_min',.2) and x.get('support',1)<cfg.get('conflict_max',.25)

def weighted_count(metric,points,mask,evidence,cfg):
    original=metric._count_points_in_mask(points,mask)
    if not cfg.get('cycle') or cfg.get('strength',0)==0 or evidence.get('reliability',0)<cfg.get('reliability_min',.2):return original
    # Preserve the original hard success test; context attenuates its support mass.
    w=1-cfg['strength']*evidence['reliability']*(1-evidence['support'])
    return float(original*w)

def compute_context(matcher,result,predicted_masks,source_masks,cfg):
    import torch
    import torch.nn.functional as F
    q=result['features_query'];t=result['features_target'];qh,qw=q.shape[-2:];th,tw=t.shape[-2:]
    # Existing backward-matcher's normalized feature maps; no extra encoder pass.
    s=q.flatten(1).T.float() @ t.flatten(1).float()
    qv,qi=s.topk(2,dim=1);tv,ti=s.topk(2,dim=0);qb=qi[:,0];tb=ti[0]
    qok=(tb[qb]==torch.arange(len(qb),device=q.device))&(qv[:,0]>=cfg['min_similarity'])&((qv[:,0]-qv[:,1])>=cfg['min_margin'])
    tok=(qb[tb]==torch.arange(len(tb),device=q.device))&(tv[0]>=cfg['min_similarity'])&((tv[0]-tv[1])>=cfg['min_margin'])
    qb,tb,qok,tok=[v.cpu().numpy() for v in (qb,tb,qok,tok)]
    def quant(m,h,w):
        # The original matcher stretches each mask to its own square feature grid.
        x=torch.as_tensor(np.asarray(m).copy(),device=q.device).float()
        return (F.interpolate(x[None,None],size=(h,w),mode='area')[0,0]>.05).cpu().numpy()
    if np.asarray(predicted_masks).ndim==2:predicted_masks=np.asarray(predicted_masks)[None]
    assert len(predicted_masks)==len(source_masks)
    normal=[];wrong=[]
    for pm,sm in zip(predicted_masks,source_masks):
        qm=quant(pm,qh,qw);tm=quant(sm,th,tw)
        args=(qm,tm,qb,tb,qok,tok)
        normal.append(evidence_from_matches(*args,scale=cfg['scale'],min_matches=cfg['min_matches']))
        wrong.append(evidence_from_matches(*args,scale=cfg['scale'],wrong=True,min_matches=cfg['min_matches']))
    return normal,wrong

class ContextCapture:
    """Instrument two existing backward matchers without changing their point outputs."""
    def __init__(self,model,cfg=None):
        import types
        self.cfg={**DEFAULTS,**(cfg or {})};self.records={};self.source_masks=None;self.enabled=True
        for name,expert in [('visual_anchor',model.visual_anchor_expert),('fusion',model.fusion_expert)]:
            matcher=expert.sparse_correspondence_backward;original=matcher.forward
            def wrap(this,*args,_original=original,_name=name,**kwargs):
                if not self.enabled:return _original(*args,**kwargs)
                wanted=kwargs.get('return_features',False);kwargs['return_features']=True
                result=_original(*args,**kwargs)
                masks=kwargs.get('mask_query',args[2] if len(args)>2 else None)
                assert masks is not None and self.source_masks is not None
                self.records[_name].append(compute_context(this,result,masks,self.source_masks,self.cfg))
                if not wanted:
                    result.pop('features_query',None);result.pop('features_target',None)
                return result
            matcher.forward=types.MethodType(wrap,matcher)
    def begin(self,source_masks):
        self.source_masks=source_masks.detach().cpu().numpy();self.records={'visual_anchor':[],'fusion':[]}
    def finish(self):
        assert len(self.records['visual_anchor'])==2 and len(self.records['fusion'])==1
        v,a=self.records['visual_anchor'];f=self.records['fusion'][0]
        return {name:[{e:r[i][j] for e,r in [('visual',v),('anchor',a),('fusion',f)]} for j in range(len(v[0]))] for i,name in enumerate(('normal','wrong'))}
