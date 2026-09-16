"""Prediction-only candidate changes. No target masks or metric labels accepted."""
import hashlib
import numpy as np

def stable_pair_seed(video_id):
    return int.from_bytes(hashlib.sha256(('candidate-quality-v1:'+str(video_id)).encode()).digest()[:4],'little') % (2**31-1)

def select_reliable(similarity, source_fg, source_hw, target_hw, max_points=3, min_margin=.01, separation=2.):
    """Strict global mutual NN + target-neighbor margin + separation in both grids."""
    sim=np.asarray(similarity,dtype=np.float32);fg=np.asarray(source_fg,dtype=bool).reshape(-1)
    if sim.ndim!=2 or sim.shape!=(int(np.prod(source_hw)),int(np.prod(target_hw))):raise ValueError('Similarity geometry mismatch')
    if fg.size!=sim.shape[0] or max_points<1:raise ValueError('Bad foreground or cap')
    if not np.isfinite(sim).all():raise ValueError('Nonfinite similarity')
    if sim.shape[1]<2:return [],{'eligible':0,'reason':'fewer_than_two_target_patches'}
    target=sim.argmax(1);reverse=sim.argmax(0);row=np.arange(sim.shape[0]);scores=sim[row,target]
    runnerup=np.partition(sim,-2,axis=1)[:,-2];margin=scores-runnerup
    eligible=np.flatnonzero(fg & (reverse[target]==row) & (margin>=min_margin))
    order=eligible[np.argsort(-scores[eligible],kind='stable')];chosen=[]
    for source in order:
        dest=int(target[source]);q=np.array([source//source_hw[1],source%source_hw[1]],dtype=float);t=np.array([dest//target_hw[1],dest%target_hw[1]],dtype=float)
        if any(np.linalg.norm(q-c['q'])<separation or np.linalg.norm(t-c['t'])<separation for c in chosen):continue
        chosen.append({'source':int(source),'target':dest,'score':float(scores[source]),'margin':float(margin[source]),'q':q,'t':t})
        if len(chosen)>=max_points:break
    return chosen,{'eligible':int(len(eligible)),'selected':len(chosen),'min_margin':min_margin,'separation_patches':separation}

def probability_weighted_pool(feature_map, region_masks, return_dtype):
    import torch
    import torch.nn.functional as F
    if len(feature_map)!=len(region_masks):raise ValueError('Batch mismatch')
    result=[]
    for features,masks in zip(feature_map,region_masks):
        if len(masks)==0:result.append(None);continue
        side=int(features.shape[0]**.5)
        if side*side!=features.shape[0]:raise ValueError('Expected square SAM feature grid')
        m=torch.as_tensor(masks,device=features.device).float()
        if m.ndim==2:m=m[None]
        if m.ndim!=3 or not torch.isfinite(m).all():raise ValueError('Invalid coarse mask')
        weights=F.interpolate(m.clamp(0,1)[:,None],size=(side,side),mode='area')[:,0].flatten(1)
        mass=weights.sum(1,keepdim=True)
        pooled=(weights @ features.float()) / mass.clamp_min(1e-8)
        pooled=torch.where(mass>1e-8,pooled,torch.zeros_like(pooled))
        result.append(pooled[:,None].to(return_dtype))
    return result

def attach_reliable_points(model):
    import types,torch
    forward=model.forward_correspondence
    original=forward.forward
    def modified(self,image_ego,image_exo,mask_ego=None,mask_exo=None,return_features=False):
        if mask_exo is not None:raise ValueError('Target segmentation must not enter matching')
        base=original(image_ego,image_exo,mask_ego=mask_ego,mask_exo=None,return_features=True)
        fq,ft=base['features_ego'],base['features_exo'];qh,qw=fq.shape[-2:];th,tw=ft.shape[-2:]
        sim=(fq.flatten(1).T.float() @ ft.flatten(1).float()).cpu().numpy()
        raw=mask_ego.detach().cpu().numpy() if torch.is_tensor(mask_ego) else np.asarray(mask_ego)
        if raw.ndim==2:raw=raw[None]
        qheight=image_ego.size[1];theight=image_exo.size[1];diagnostics=[]
        for i,m in enumerate(raw):
            resized=self._resize_transform(m).to(self.mean.device)
            quant=self.patch_quant_filter(resized[None]).squeeze().reshape(-1)
            selected,info=select_reliable(sim,(quant>self.mask_fg_threshold).cpu().numpy(),(qh,qw),(th,tw))
            # Conservative fallback preserves the original point if insufficient reliable support.
            replace=len(selected)>=2;info['used_multipoint']=replace;info['legacy_count']=len(base['points_exo'][i]);diagnostics.append(info)
            if replace:
                q=np.array([[c['q'][1]+.5,c['q'][0]+.5] for c in selected],dtype=np.float32)*self.patch_size*(qheight/self.image_size)
                t=np.array([[c['t'][1]+.5,c['t'][0]+.5] for c in selected],dtype=np.float32)*self.patch_size*(theight/self.image_size)
                base['points_ego'][i]=q;base['points_exo'][i]=t;base['match_confidences'][i]=np.asarray([c['score'] for c in selected],np.float32);base['num_matches'][i]=len(selected)
        self._candidate_quality_diagnostics=diagnostics
        if not return_features:base.pop('features_ego');base.pop('features_exo')
        return base
    forward.forward=types.MethodType(modified,forward)

def install_weighted_pool(model):
    import types
    for expert in (model.visual_anchor_expert,model.fusion_expert):
        sampler=expert.region_sampler
        def weighted(self,features,masks,original_dtype,return_dtype):
            # Consume exactly the original sampler RNG to isolate later expert branches.
            self.forward(features,masks,original_dtype,return_dtype)
            return probability_weighted_pool(features,masks,return_dtype)
        sampler.weighted_forward=types.MethodType(weighted,sampler)

def patch_expert_calls(code_root):
    for name in ('pccs_visual_anchor_expert.py','pccs_fusion_expert.py'):
        p=code_root/'projects/v2sam_pccs/models'/name;s=p.read_text()
        needle='pred_mask_visual_embeds = self.region_sampler('
        if s.count(needle)!=1:raise RuntimeError('Unexpected upstream pooling call')
        s=s.replace(needle,"pred_mask_visual_embeds = (self.region_sampler.weighted_forward if getattr(self.region_sampler, 'weighted_enabled', False) else self.region_sampler)(")
        p.write_text(s)
