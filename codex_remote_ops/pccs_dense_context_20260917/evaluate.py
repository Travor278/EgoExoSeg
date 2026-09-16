import json,hashlib,copy
import numpy as np
from data_utils import R,read,score
from policies import hand_select,calibrated_select
def main():
    import joblib
    rows=read('exo2exo');frozen=json.loads((R/'selection.json').read_text());main=frozen['selected'];results={};predictions={}
    def predict(cfg,wrong=False):
        if cfg['family']=='hand':return [hand_select(r,cfg['config'],wrong=wrong) for r in rows]
        path=R/'fitted'/cfg['checkpoint'].split('/')[-1];assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['sha256'];model=joblib.load(path);return [calibrated_select(r,model,cfg['group'],cfg['threshold'],wrong) for r in rows]
    base=[r['baseline'] for r in rows];results['baseline']=score(rows,base)
    for name,cfg,wrong in [('primary',main,False),('wrong_context_primary',main,True)]+[(r['group']+'_matched',r,False) for r in frozen['matched_controls']]:
        choices=predict(cfg,wrong);results[name]=score(rows,choices);predictions[name]=choices
    assert predictions['primary']==[r['integrated_primary'] for r in rows],'PCCS-integrated decisions differ from frozen policy replay'
    assert all(r['frozen_omama_reference'] is not None for r in rows)
    refs=copy.deepcopy(rows)
    for r in refs:r['metrics']['omama_reference']=r['frozen_omama_reference']
    for r in refs:r['mask_sha256']['omama_reference']=r['mask_sha256'][r['frozen_omama_expert']]
    results['omama_reference']=score(refs,['omama_reference']*len(rows))
    for r in refs:r['baseline']='omama_reference'
    vs_omama=score(refs,predictions['primary'])
    summary={'coverage':'exact','selected':main,'methods':results,'primary_vs_omama':vs_omama,'no_target_fitting':True,'within_run_candidates_identical':True,'historical_candidate_drift_objects':sum(r.get('historical_drift',False) for r in rows),'pretrained_omama_used_in_method':False,'reference_repair_separate_process':True}
    (R/'exo2exo_results.json').write_text(json.dumps(summary,indent=2));(R/'exo2exo_predictions.jsonl').write_text(''.join(json.dumps({'video_id':r['video_id'],'obj_id':r['obj_id'],'baseline':r['baseline'],'selected':{k:v[i] for k,v in predictions.items()},'metrics':r['metrics'],'omama_reference':r['frozen_omama_reference']})+'\n' for i,r in enumerate(rows)));print('TARGET_COMPLETE',flush=True)
if __name__=='__main__':main()
