from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
from data_utils import R,read
from policies import BASE,DENSE

def main():
    checks={};prior=R.parent/'pccs_dense_context_20260917';takes={}
    for phase in ('smoke','cache_smoke','train','calibration'):
        if not all((R/'runs'/phase/f'rank{k}/receipt.json').exists() for k in range(4)):continue
        rows=read(phase);checks[phase]={'objects':len(rows),'pairs':len({r['video_id'] for r in rows}),'receipt_sha_and_coverage':'passed'}
        if phase in ('train','calibration'):
            old={}
            for p in (prior/'runs'/phase).glob('rank*/records.jsonl'):old.update({(r['video_id'],r['obj_id']):r for r in map(json.loads,p.read_text().splitlines())})
            assert len(rows)==len(old)
            for r in rows:
                o=old[(r['video_id'],r['obj_id'])];assert r['mask_sha256']==o['mask_sha256'] and r['baseline']==o['baseline']
                for e,x in r['evidence'].items():
                    for k in BASE+DENSE:assert np.isclose(x[k],o['evidence'][e][k],atol=1e-6,rtol=1e-5)
            takes[phase]={r['take_id'] for r in rows};checks[phase]['global_evidence_and_candidate_parity']=True
    if 'train' in takes and 'calibration' in takes:assert takes['train'].isdisjoint(takes['calibration'])
    if (R/'selection.json').exists():
        sel=json.loads((R/'selection.json').read_text());assert json.loads((R/'folds.json').read_text())==json.loads((prior/'folds.json').read_text())
        for cfg in sel['matched_controls']+[sel['selected']]:
            if cfg['family']=='baseline':continue
            p=R/'fitted'/cfg['checkpoint'].split('/')[-1];assert hashlib.sha256(p.read_bytes()).hexdigest()==cfg['sha256']
        checks['selection_sha256']=hashlib.sha256((R/'selection.json').read_bytes()).hexdigest()
    for phase in ('exo2exo','exo2ego'):
        p=R/(phase+'_results.json')
        if not p.exists():continue
        rows=read(phase);result=json.loads(p.read_text());preds={(r['video_id'],r['obj_id']):r['choices'] for r in map(json.loads,(R/(phase+'_predictions.jsonl')).read_text().splitlines())};assert len(preds)==len(rows)
        if phase=='exo2ego':assert {r['take_id'] for r in rows}.isdisjoint(takes['train']|takes['calibration'])
        for name,v in result['methods'].items():
            pairs=defaultdict(list)
            for r in rows:
                if name=='omama_reference':metric=r['frozen_omama_reference']
                else:
                    expert=preds[(r['video_id'],r['obj_id'])][name];metric=r['metrics'][expert]
                    if name=='primary':assert expert==r['integrated_primary']
                    if name=='frozen_cycle':assert expert==r['integrated_frozen_cycle']
                pairs[r['video_id']].append(metric)
            frame=np.mean([np.mean(v,axis=0) for v in pairs.values()],axis=0);assert np.allclose(frame,v['frame'],rtol=0,atol=1e-12)
        checks[phase]={'objects':len(rows),'pairs':len({r['video_id'] for r in rows}),'baseline_zero_iou_objects':sum(r['metrics'][r['baseline']][0]==0 for r in rows),'actual_interface_and_frame_aggregation':'passed','cached_candidates':all(r.get('cached_candidates',False) for r in rows)}
    dest=R/('final_validation.json' if 'exo2ego' in checks and 'exo2exo' in checks else 'partial_validation.json');dest.write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
if __name__=='__main__':main()
