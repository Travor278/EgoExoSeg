"""Validate downloaded receipts, source parity and paired target aggregation."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
from data_utils import read,R

def main():
    checks={};prior=R.parent/'pccs_dense_context_20260917'
    assert json.loads((R/'folds.json').read_text())==json.loads((prior/'folds.json').read_text())
    takes={}
    for phase in ('train','calibration'):
        rows=read(phase);old={}
        for p in (prior/'runs'/phase).glob('rank*/records.jsonl'):
            old.update({(r['video_id'],r['obj_id']):r for r in map(json.loads,p.read_text().splitlines())})
        assert len(old)==len(rows)
        for r in rows:
            o=old[(r['video_id'],r['obj_id'])];assert r['mask_sha256']==o['mask_sha256'] and r['baseline']==o['baseline']
            for e,x in o['evidence'].items():
                for k,v in x.items():assert np.isclose(r['evidence'][e][k],v,atol=1e-6,rtol=1e-5)
        takes[phase]={r['take_id'] for r in rows};checks[phase]={'objects':len(rows),'native_feature_and_candidate_parity':True}
    assert takes['train'].isdisjoint(takes['calibration'])
    for phase in ('exo2ego','exo2exo'):
        if not (R/(phase+'_results.json')).exists():continue
        rows=read(phase);result=json.loads((R/(phase+'_results.json')).read_text());preds={(x['video_id'],x['obj_id']):x for x in map(json.loads,(R/(phase+'_predictions.jsonl')).read_text().splitlines())};assert len(preds)==len(rows)
        if phase=='exo2ego':assert {r['take_id'] for r in rows}.isdisjoint(takes['train']|takes['calibration'])
        for name,summary in result['methods'].items():
            values=defaultdict(list)
            for r in rows:
                if name=='omama_reference':v=r['frozen_omama_reference']
                else:
                    e=r['baseline'] if name=='baseline' else preds[(r['video_id'],r['obj_id'])]['choices'][name];v=r['metrics'][e]
                    if name=='primary':assert e==r['integrated_primary']
                    if name=='frozen_cycle':assert e==r['integrated_frozen_cycle']
                values[r['video_id']].append(v)
            computed=np.mean([np.mean(v,axis=0) for v in values.values()],axis=0);assert np.allclose(computed,summary['frame'],atol=1e-12,rtol=0)
        checks[phase]={'pairs':len({r['video_id'] for r in rows}),'objects':len(rows),'baseline_zero_iou_objects':sum(r['metrics'][r['baseline']][0]==0 for r in rows),'receipt_sha_and_paired_frame_aggregation':'passed','actual_pipeline_route_vs_frozen_policy':'passed','historical_drift_objects':sum(r.get('historical_drift',False) for r in rows)}
    checks['selection_sha256']=hashlib.sha256((R/'selection.json').read_bytes()).hexdigest()
    frozen=json.loads((R/'resume_capsule_manifest.json').read_text())['frozen_selection_sha256'];assert checks['selection_sha256']==frozen
    path=R/('final_validation.json' if all(x in checks for x in ('exo2ego','exo2exo')) else 'source_validation.json');path.write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
if __name__=='__main__':main()
