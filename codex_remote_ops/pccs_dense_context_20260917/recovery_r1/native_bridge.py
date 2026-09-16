"""Connect dense native confidence to PCCS after its original rule, with exact fallback."""
import hashlib
from pathlib import Path
from policies import hand_select,calibrated_select,EXPERTS

def native_select(metric,prediction,source_mask,object_index,original):
    cfg=getattr(metric,'native_dense_config',None)
    if not cfg or cfg.get('strength',1.)==0:return original
    row={'baseline':original,'source_valid':bool(source_mask.any()),'evidence':prediction['dense_context'][object_index],
         'mask_sha256':{e:hashlib.sha256(prediction['pred_masks_'+e][object_index].detach().cpu().contiguous().numpy().tobytes()).hexdigest() for e in EXPERTS}}
    if cfg['family']=='hand':return hand_select(row,cfg['config'])
    if not hasattr(metric,'_native_dense_model'):
        import joblib
        p=Path(cfg['checkpoint']);assert hashlib.sha256(p.read_bytes()).hexdigest()==cfg['sha256'];metric._native_dense_model=joblib.load(p)
    return calibrated_select(row,metric._native_dense_model,cfg['group'],cfg['threshold'])

def patch_metric(root):
    p=root/'projects/v2sam_pccs/evaluation/pccs_metric.py';s=p.read_text();old='                fallback_expert = best_expert';assert s.count(old)==1
    s='from native_bridge import native_select\n'+s
    s=s.replace(old,"                native_expert = native_select(self, data_samples[i], raw_prompt_mask_j, j, best_expert)\n                if native_expert != best_expert:\n                    best_expert = native_expert\n                    selection_reason = 'native_dense_cycle_challenger'\n"+old);p.write_text(s)
