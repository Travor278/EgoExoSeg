"""Inference adapter for the unmodified published O-MaMa modules.

The upstream model/descriptor modules retain their AGPL-3.0 license under
reference/O-MaMa. This adapter changes the candidate source and supports
explicitly labeled positional-grid transfer, without fitting any parameters.
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'reference/O-MaMa'))
from model.model import Attention_projector
from descriptors.get_descriptors import DescriptorExtractor


def load_head(direction,device):
    state=torch.load(ROOT/'weights/omama_exo2ego.pt',map_location='cpu',weights_only=True)
    reverse=direction!='ego2exo'
    if not reverse:
        state=dict(state)
        state['pos_embed_Q'],state['pos_embed_T']=state['pos_embed_T'],state['pos_embed_Q']
    head=Attention_projector(reverse=reverse)
    receipt=head.load_state_dict(state,strict=True)
    head.eval().requires_grad_(False).to(device)
    grids=((38,68),(50,50)) if reverse else ((50,50),(38,68))
    return head,grids,{'missing':list(receipt.missing_keys),'unexpected':list(receipt.unexpected_keys),'loaded_keys':len(state),'position_tables_swapped':not reverse}


def position(table,old_grid,new_grid):
    if old_grid==new_grid:return table
    x=table.reshape(1,*old_grid,table.shape[-1]).permute(0,3,1,2)
    return F.interpolate(x,size=new_grid,mode='bilinear',align_corners=False).flatten(2).transpose(1,2)


def learned_scores(head,qdesc,tdesc,qfeat,tfeat,grids,drop_context=False,drop_cross=False):
    """Same arithmetic as upstream forward, exposes all scores without GT args."""
    qp=position(head.pos_embed_Q,grids[0],qfeat.shape[-2:])
    tp=position(head.pos_embed_T,grids[1],tfeat.shape[-2:])
    qt=qfeat.flatten(2).transpose(1,2)+qp
    tt=tfeat.flatten(2).transpose(1,2)+tp
    qc=head.CROSS_context_attn(qdesc[:,:,:768],tt,residual=False) if not drop_cross else torch.zeros_like(qdesc[:,:,:768])
    tc=head.CROSS_context_attn(tdesc[:,:,:768],qt,residual=False) if not drop_cross else torch.zeros_like(tdesc[:,:,:768])
    if drop_context:
        qdesc=torch.cat((qdesc[:,:,:768],torch.zeros_like(qdesc[:,:,768:])),dim=2)
        tdesc=torch.cat((tdesc[:,:,:768],torch.zeros_like(tdesc[:,:,768:])),dim=2)
    q=F.normalize(head.mlp(torch.cat((qc,qdesc),dim=2)),p=2,dim=2)
    t=F.normalize(head.mlp(torch.cat((tc,tdesc),dim=2)),p=2,dim=2)
    return F.cosine_similarity(q,t,dim=2)[0]


def bbox(mask):
    # Match upstream dataset_utils.bbox_from_mask (no +1 to width/height).
    y,x=torch.where(mask>0)
    if not len(x):return torch.zeros(4,device=mask.device)
    return torch.stack((x.min(),y.min(),x.max()-x.min(),y.max()-y.min())).float()


def preprocess(path,masks,mode,grid,device):
    im=Image.open(path).convert('RGB');w,h=im.size
    if mode=='canonical':new_h,new_w=grid[0]*14,grid[1]*14
    elif mode=='native_interp':
        scale=min(1.,1024/max(w,h))
        new_h=max(14,int(h*scale)//14*14);new_w=max(14,int(w*scale)//14*14)
    else:raise ValueError(mode)
    # Upstream uses nearest resize before ToTensor/ImageNet normalization.
    im=im.resize((new_w,new_h),Image.Resampling.NEAREST)
    x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float().to(device)/255
    x=(x-torch.tensor([.485,.456,.406],device=device)[:,None,None])/torch.tensor([.229,.224,.225],device=device)[:,None,None]
    ms=[F.interpolate(torch.as_tensor(m,device=device).float()[None,None],size=(new_h,new_w),mode='nearest')[0,0] for m in masks]
    return x[None],torch.stack(ms),torch.tensor([[new_h,new_w]],device=device)


class AlignedMatcher:
    def __init__(self,device='cuda'):
        self.device=torch.device(device)
        self.dino=torch.hub.load(str(ROOT/'reference/dinov2'),'dinov2_vitb14_reg',source='local',pretrained=False)
        state=torch.load(ROOT/'weights/dinov2_vitb14_reg4_pretrain.pth',map_location='cpu',weights_only=True)
        self.dino.load_state_dict(state,strict=True)
        self.dino.eval().requires_grad_(False).to(self.device)
        # Bypass only network loading; all descriptor methods are upstream code.
        self.extractor=DescriptorExtractor.__new__(DescriptorExtractor)
        self.extractor.model=self.dino;self.extractor.patch_size=14
        self.extractor.context_size=100;self.extractor.device=self.device
        self.heads={}
        self.load_receipts={}
        for d in ('exo2ego','ego2exo'):
            h,g,r=load_head(d,self.device);self.heads[d]=(h,g);self.load_receipts[d]=r

    @torch.inference_mode()
    def score(self,query_path,target_path,query_mask,target_masks,direction,mode,ablate=False):
        h,g=self.heads['ego2exo' if direction=='ego2exo' else 'exo2ego']
        qi,qm,qs=preprocess(query_path,[query_mask],mode,g[0],self.device)
        ti,tm,ts=preprocess(target_path,target_masks,mode,g[1],self.device)
        batch={'SOURCE_img':qi,'SOURCE_mask':qm,'SOURCE_bbox':bbox(qm[0])[None],'SOURCE_img_size':qs,
               'GT_img':ti,'DEST_SAM_masks':tm[None],'DEST_SAM_bbox':torch.stack([bbox(m) for m in tm])[None],'DEST_img_size':ts}
        qd,qf=self.extractor.get_SOURCE_descriptors(batch)
        td,tf=self.extractor.get_DEST_descriptors(batch)
        scores={
            'dino2_object':F.cosine_similarity(qd[:,:,:768],td[:,:,:768],dim=2)[0],
            'dino2_object_context':F.cosine_similarity(qd,td,dim=2)[0],
            'omama_learned':learned_scores(h,qd,td,qf,tf,g),
        }
        if ablate:
            scores['omama_context_zero']=learned_scores(h,qd,td,qf,tf,g,drop_context=True)
            scores['omama_cross_zero']=learned_scores(h,qd,td,qf,tf,g,drop_cross=True)
            scores['omama_context_cross_zero']=learned_scores(h,qd,td,qf,tf,g,drop_context=True,drop_cross=True)
        if not all(torch.isfinite(v).all() for v in scores.values()):raise ValueError('Nonfinite matcher output')
        # Check original forward equivalence for the canonical geometry.
        if mode=='canonical':
            orig=h(qd,td,qf,tf,None,None,tm[None],test_mode=True)
            torch.testing.assert_close(torch.sigmoid(scores['omama_learned'].max()),orig[0][0],atol=1e-6,rtol=1e-6)
            assert int(scores['omama_learned'].argmax())==int(orig[1][0])
        return {k:v.cpu().tolist() for k,v in scores.items()},{'query_hw':qi.shape[-2:],'target_hw':ti.shape[-2:],'query_grid':qf.shape[-2:],'target_grid':tf.shape[-2:]}
