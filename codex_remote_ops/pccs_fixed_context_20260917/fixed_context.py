"""Expanded-box evidence on existing native PCCS DINOv3 tokens.

px: 50/100/150 pixels in the native resized image (height 768).
om100: 100 pixels in the O-MaMa canonical source 532x952 / target
700x700 coordinate system, mapped by normalized image coordinates.
This borrows region geometry only, not O-MaMa's encoder or learned head.
"""
import numpy as np

VARIANTS = tuple(f'px{p}_{mode}' for p in (50,100,150) for mode in ('box','ring')) + tuple(f'scale{s}_{mode}' for s in ('15','20') for mode in ('box','ring')) + ('om100_box','om100_ring')
KEYS=('valid','context_cos','cross_bg','contrast','residual_cos','cycle_contrast')

def region(mask, grid_hw, variant, source=False):
    """Exact fractional feature-cell coverage of the expanded pixel bbox."""
    h,w=grid_hw;binary=np.asarray(mask)>0;y,x=np.where(binary)
    if not len(x):return np.zeros((h,w),np.float32)
    ih,iw=binary.shape;y0,y1=y.min()/ih,(y.max()+1)/ih;x0,x1=x.min()/iw,(x.max()+1)/iw
    rule=variant.rsplit('_',1)[0]
    if rule.startswith('px'):
        p=float(rule[2:]);dy=p/(h*16);dx=p/(w*16)
    elif rule=='om100':
        ch,cw=(532,952) if source else (700,700);dy=100/ch;dx=100/cw
    else:
        scale={'scale15':1.5,'scale20':2.}[rule];dy=(y1-y0)*(scale-1)/2;dx=(x1-x0)*(scale-1)/2
    y0,y1=max(0,y0-dy)*h,min(1,y1+dy)*h;x0,x1=max(0,x0-dx)*w,min(1,x1+dx)*w
    yy=np.maximum(0,np.minimum(np.arange(h)+1,y1)-np.maximum(np.arange(h),y0))
    xx=np.maximum(0,np.minimum(np.arange(w)+1,x1)-np.maximum(np.arange(w),x0))
    return np.outer(yy,xx).astype(np.float32)

def evidence(qf,tf,qa,ta,pm,sm,oq,ot,obj,soft_f,into_source,into_candidate):
    import torch
    import torch.nn.functional as F
    def weights(mask,area,source):
        result=[]
        for v in VARIANTS:
            r=torch.as_tensor(region(mask,area.shape,v,source),device=qf.device)
            if v.endswith('_ring'):r=(r-area).clamp_min(0)
            result.append(r.flatten())
        return torch.stack(result)
    ar=weights(pm,qa,False);br=weights(sm,ta,True)
    # Wrong-neighborhood control: same area and features, spatially displaced source.
    wrong=br.reshape(len(VARIANTS),*ta.shape).roll((ta.shape[0]//2,ta.shape[1]//2),(-2,-1)).flatten(1)
    result={}
    for prefix,bs in (('',br),('wrong_',wrong)):
        sa=ar.sum(1);sb=bs.sum(1);ok=(sa>=2)&(sb>=2)&(qa.sum()>0)&(ta.sum()>0)
        cq=F.normalize(ar@qf/sa[:,None].clamp_min(1e-8),dim=1);ct=F.normalize(bs@tf/sb[:,None].clamp_min(1e-8),dim=1)
        cross=((cq*ot).sum(1)+(ct*oq).sum(1))/2
        values=torch.stack([ok.float(),(cq*ct).sum(1),cross,obj-cross,(F.normalize(oq-.5*cq,dim=1)*F.normalize(ot-.5*ct,dim=1)).sum(1),soft_f-((ar@into_source)/sa.clamp_min(1e-8)+(bs@into_candidate)/sb.clamp_min(1e-8))/2],1)
        values=torch.where(ok[:,None],values,torch.zeros_like(values)).detach().cpu().numpy()
        assert np.isfinite(values).all()
        result.update({prefix+v+'_'+k:float(values[i,j]) for i,v in enumerate(VARIANTS) for j,k in enumerate(KEYS)})
    return result
