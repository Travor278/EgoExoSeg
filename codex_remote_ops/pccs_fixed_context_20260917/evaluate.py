"""Frozen source-selected policies; paired frame metrics with take bootstrap."""
import json,hashlib,copy
from pathlib import Path
from data_utils import R,read,score
from policies import calibrated_select

def main(phase='exo2exo'):
    import joblib
    rows=read(phase);selection=json.loads((R/'selection.json').read_text());frozen=json.loads((R/'manifest.json').read_text())['frozen_cycle'];predictions={};results={}
    def predict(cfg,wrong=False):
        p=Path(cfg['checkpoint']);assert hashlib.sha256(p.read_bytes()).hexdigest()==cfg['sha256'];m=joblib.load(p)
        return [calibrated_select(r,m,cfg['group'],cfg['threshold'],wrong=wrong) for r in rows]
    results['baseline']=score(rows,[r['baseline'] for r in rows])
    configurations=[('primary',selection['selected'],False),('frozen_cycle',frozen,False),('wrong_context_primary',selection['selected'],True)]
    configurations += [(c['group']+'_matched',c,False) for c in selection['matched_controls']]
    # Always report 100px vs scale at identical complexity; never select on target.
    for name,cfg,wrong in configurations:
        choices=predict(cfg,wrong);predictions[name]=choices;results[name]=score(rows,choices)
    assert predictions['primary']==[r['integrated_primary'] for r in rows]
    assert predictions['frozen_cycle']==[r['integrated_frozen_cycle'] for r in rows]
    refs=copy.deepcopy(rows)
    for r,e in zip(refs,predictions['frozen_cycle']):r['baseline']=e
    incremental={name:score(refs,choices) for name,choices in predictions.items() if name!='frozen_cycle'}
    summary={'phase':phase,'coverage':'exact','selected':selection['selected'],'methods':results,'vs_frozen_cycle':incremental,'no_target_fitting':True,'within_run_candidates_identical':True,'pretrained_omama_used_in_method':False,'scope':json.loads((R/'manifest.json').read_text()).get('test_scope') if phase=='exo2ego' else 'Existing full 1094-pair Exo2Exo benchmark; repeatedly evaluated, not fresh blind test.'}
    if phase=='exo2exo':
        assert all(r['frozen_omama_reference'] is not None for r in rows)
        refs=copy.deepcopy(rows)
        for r in refs:r['metrics']['omama_reference']=r['frozen_omama_reference'];r['mask_sha256']['omama_reference']=r['mask_sha256'][r['frozen_omama_expert']]
        summary['methods']['omama_reference']=score(refs,['omama_reference']*len(rows))
        for r in refs:r['baseline']='omama_reference'
        summary['primary_vs_omama']=score(refs,predictions['primary'])
    (R/(phase+'_results.json')).write_text(json.dumps(summary,indent=2))
    (R/(phase+'_predictions.jsonl')).write_text(''.join(json.dumps({'video_id':r['video_id'],'obj_id':r['obj_id'],'baseline':r['baseline'],'choices':{k:v[i] for k,v in predictions.items()}})+'\n' for i,r in enumerate(rows)))
    print('FROZEN_TARGET_COMPLETE',phase,summary['methods']['primary'],flush=True)
if __name__=='__main__':main()
