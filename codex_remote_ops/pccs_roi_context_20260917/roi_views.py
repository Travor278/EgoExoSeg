"""Native PCCS local views: retain object pixels, vary available surrounding context."""
import math
import numpy as np
from PIL import Image

def crop_view(image,mask,scale=2.,object_only=False):
    image=image.convert('RGB');w,h=image.size;m=(np.asarray(mask)>0).astype(np.uint8)
    if m.shape!=(h,w):m=np.asarray(Image.fromarray(m).resize((w,h),Image.Resampling.NEAREST))
    ys,xs=np.where(m)
    if len(xs)<16:return None
    extent=max(int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1))
    if extent>min(w,h):return None
    side=min(min(w,h),max(32,int(math.ceil(scale*extent))));cx=(xs.min()+xs.max()+1)/2;cy=(ys.min()+ys.max()+1)/2
    left=int(np.clip(round(cx-side/2),0,w-side));top=int(np.clip(round(cy-side/2),0,h-side));box=(left,top,left+side,top+side)
    cm=m[top:top+side,left:left+side].copy();assert int(cm.sum())==int(m.sum()),'Crop truncated foreground'
    view=image.crop(box)
    if object_only:
        pixels=np.asarray(view).copy();pixels[cm==0]=np.asarray([124,116,104],dtype=np.uint8);view=Image.fromarray(pixels)
    return {'image':view,'mask':cm,'box':box,'effective_scale':side/extent,'object_only':object_only,'original_hw':(h,w)}

def encode_views(matcher,views,batch_size=4):
    """Use the original matcher and its existing DINOv3 weights; no new backbone."""
    import torch
    import torch.nn.functional as F
    out=[]
    for start in range(0,len(views),batch_size):
        items=views[start:start+batch_size]
        pixels=torch.stack([(matcher._resize_transform(v['image']).to(matcher.mean.device)-matcher.mean)/matcher.std for v in items])
        with torch.inference_mode():
            f=matcher._extract_features(pixels)
            if f.ndim==3:f=f.unsqueeze(0)
            assert len(f)==len(items);f=F.normalize(f,dim=1)
            out.extend(f[i] for i in range(len(items)))
    return out
