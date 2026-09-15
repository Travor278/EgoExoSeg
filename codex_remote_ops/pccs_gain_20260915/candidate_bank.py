"""Freeze prediction masks once, without receiving target annotations."""
from pathlib import Path
import hashlib,os
import numpy as np

def save_bank(query_mask,candidates,baseline,video_id,object_id):
    rank=os.environ.get('RANK','0')
    root=Path(os.environ['PCCS_GAIN_BANK'])/('rank'+rank)
    root.mkdir(parents=True,exist_ok=True)
    stem=hashlib.sha256(str(video_id).encode()).hexdigest()[:20]+'_'+hashlib.sha256(str(object_id).encode()).hexdigest()[:10]
    p=root/(stem+'.npz')
    if p.exists():raise RuntimeError('Refusing to overwrite frozen candidate '+str(p))
    arrays={}
    for name,mask in {'query':query_mask,'baseline':baseline,**candidates}.items():
        if mask is None:raise ValueError('Missing candidate '+name)
        a=(mask.detach().cpu().numpy().squeeze()>0)
        if a.ndim!=2:raise ValueError('Expected H W mask')
        arrays[name]=np.packbits(a);arrays[name+'_shape']=np.asarray(a.shape)
    tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(p)
    return str(p)
