import json,hashlib,copy
from pathlib import Path
from collections import defaultdict
import numpy as np
from data_utils import R,read,score
from roi_policy import predict
from policies import calibrated_select

def main(phase='exo2exo'):
    import joblib
    rows=read(phase);selection=json.loads((R/'selection.json').read_text());frozen=json.loads((R/'manifest.json').read_text())['frozen_cycle'];choices={};results={}
    def get(cfg):
        p=Path(cfg['checkpoint']);assert hashlib.sha256(p.read_bytes()).hexdigest()==cfg['sha256'];m=joblib.load(p)
        if cfg['family']=='roi':return [predict(r,m,cfg['group'],cfg['threshold'],cfg['gate']) for r in rows]
        return [calibrated_select(r,m,cfg['group'],cfg['threshold']) for r in rows]
    choices['baseline']=[r['baseline'] for r in rows]
    for name,cfg in [('primary',selection['selected']),('frozen_cycle',frozen)]+[(c['group']+'_matched',c) for c in selection['matched_controls']]:choices[name]=get(cfg)
    assert choices['local20_matched']==[r['integrated_local20'] for r in rows]
    assert choices['primary']==[r['integrated_primary'] for r in rows]
    assert choices['frozen_cycle']==[r['integrated_frozen_cycle'] for r in rows]
    if all(r.get('fixed_context_reference_expert') for r in rows):choices['frozen_fixed_context']=[r['fixed_context_reference_expert'] for r in rows]
    for name,chosen in choices.items():results[name]=score(rows,chosen)
    comparisons={}
    for a,b in [('primary','frozen_cycle'),('primary','global_matched'),('global_local20_matched','global_object20_matched'),('local20_matched','object20_matched'),('primary','frozen_fixed_context')]:
        if b not in choices:continue
        refs=copy.deepcopy(rows)
        for r,e in zip(refs,choices[b]):r['baseline']=e
        comparisons[a+' minus '+b]=score(refs,choices[a])
    summary={'seed_namespace':'candidate-quality-v2:','seed1_bank_changed_objects':sum(r['seed1_bank_changed'] for r in rows),'selection_sha256':hashlib.sha256((R/'selection.json').read_bytes()).hexdigest(),'phase':phase,'coverage':'exact','selected':selection['selected'],'methods':results,'paired_comparisons':comparisons,'no_target_fitting':True,'actual_bridge_replay_exact':True,'pretrained_omama_used_in_method':False,'fixed_context_reference_complete':'frozen_fixed_context' in choices}
    if phase=='exo2exo':
        assert all(r['frozen_omama_reference'] is not None for r in rows);refs=copy.deepcopy(rows)
        for r in refs:r['metrics']['omama_reference']=r['frozen_omama_reference'];r['mask_sha256']['omama_reference']=r['mask_sha256'][r['frozen_omama_expert']]
        summary['methods']['omama_reference']=score(refs,['omama_reference']*len(rows))
        for r in refs:r['baseline']='omama_reference'
        summary['primary_vs_omama']=score(refs,choices['primary'])
    bypair=defaultdict(list)
    for r in rows:bypair[r['video_id']].append(r)
    summary['cost']={'roi_seconds_per_object_mean':float(np.mean([r['roi_metadata']['elapsed_seconds'] for r in rows])),'roi_seconds_per_pair_mean':float(np.mean([sum(r['roi_metadata']['elapsed_seconds'] for r in rs) for rs in bypair.values()])),'encoder_views_per_object_mean':float(np.mean([r['roi_metadata']['encoded_views'] for r in rows])),'gpu_peak_bytes_max':max(r['gpu_peak_bytes'] for r in rows),'cached_candidate_evaluation':all(r.get('cached_candidates',False) for r in rows),'note':'ROI collection cost includes all three ablation views, excludes model initialization; not single-selected-method end-to-end latency.'}
    (R/(phase+'_results.json')).write_text(json.dumps(summary,indent=2));(R/(phase+'_predictions.jsonl')).write_text(''.join(json.dumps({'video_id':r['video_id'],'obj_id':r['obj_id'],'choices':{k:v[i] for k,v in choices.items()}})+'\n' for i,r in enumerate(rows)));print('ROI_TARGET_COMPLETE',phase,results['primary']['delta_pp'],flush=True)
if __name__=='__main__':main()
