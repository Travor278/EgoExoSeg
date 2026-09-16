from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np
from data_utils import R,read
from randomness import stable_pair_seed

def main():
    oldroot=R.parent/'pccs_roi_context_20260917';m=json.loads((R/'manifest.json').read_text());assert hashlib.sha256((R/'selection.json').read_bytes()).hexdigest()==m['frozen_selection_sha256']==hashlib.sha256((oldroot/'selection.json').read_bytes()).hexdigest();selection=json.loads((R/'selection.json').read_text())
    for cfg in selection['matched_controls']+[selection['selected']]:assert hashlib.sha256((oldroot/'fitted'/cfg['checkpoint'].split('/')[-1]).read_bytes()).hexdigest()==cfg['sha256']
    checks={'frozen_selection_sha256':m['frozen_selection_sha256'],'seed_namespace':'candidate-quality-v2:','no_refit':True,'phases':{}}
    for phase in ('exo2exo','exo2ego'):
        path=R/(phase+'_results.json')
        if not path.exists():continue
        rows=read(phase);old={}
        for p in (oldroot/'runs'/phase).glob('rank*/records.jsonl'):
            assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads((p.parent/'receipt.json').read_text())['records_sha256'];old.update({(r['video_id'],r['obj_id']):r for r in map(json.loads,p.read_text().splitlines())})
        assert {(r['video_id'],r['obj_id']) for r in rows}==set(old)
        for r in rows:
            assert r['candidate_seed']==stable_pair_seed(r['video_id']) and r['seed_namespace']=='candidate-quality-v2:'
            prev=old[(r['video_id'],r['obj_id'])];same=r['mask_sha256']==prev['mask_sha256'] and r['baseline']==prev['baseline'] and np.allclose(r['metrics'][r['baseline']],prev['metrics'][prev['baseline']],atol=1e-7,rtol=0);assert r['seed1_bank_changed']==(not same)
        results=json.loads(path.read_text());preds={(r['video_id'],r['obj_id']):r['choices'] for r in map(json.loads,(R/(phase+'_predictions.jsonl')).read_text().splitlines())};assert len(preds)==len(rows)
        for name,summary in results['methods'].items():
            pairs=defaultdict(list)
            for r in rows:
                if name=='omama_reference':value=r['frozen_omama_reference']
                else:
                    e=preds[(r['video_id'],r['obj_id'])][name];value=r['metrics'][e]
                    for key,field in [('primary','integrated_primary'),('local20_matched','integrated_local20'),('frozen_cycle','integrated_frozen_cycle')]:
                        if name==key:assert e==r[field]
                pairs[r['video_id']].append(value)
            assert np.allclose(np.mean([np.mean(v,axis=0) for v in pairs.values()],axis=0),summary['frame'],atol=1e-12,rtol=0)
        checks['phases'][phase]={'pairs':len({r['video_id'] for r in rows}),'objects':len(rows),'baseline_zero_iou_objects':sum(r['metrics'][r['baseline']][0]==0 for r in rows),'changed_from_seed1_objects':sum(r['seed1_bank_changed'] for r in rows),'expert_masks_changed':{e:sum(not r['seed1_expert_mask_match'][e] for r in rows) for e in ('visual','anchor','fusion')},'same_keys_and_seed_verified':True,'actual_primary_local20_cycle_and_frame_aggregation':'passed'}
        if phase=='exo2exo':
            refs={}
            for rank in range(4):
                p=R/'runs/reference'/f'rank{rank}'/'references.jsonl';receipt=json.loads((p.parent/'receipt.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['records_sha256'] and receipt['witness']['max_score_error']<1e-4;refs.update({(r['video_id'],r['obj_id']):r for r in map(json.loads,p.read_text().splitlines())})
            assert set(refs)==set(old)
            for r in rows:
                ref=refs[(r['video_id'],r['obj_id'])];assert ref['mask_sha256']==r['mask_sha256'] and ref['metrics']==r['metrics'][ref['expert']]==r['frozen_omama_reference']
                if ref['scores'] is not None:
                    names=[r['baseline']];used={r['mask_sha256'][r['baseline']]}
                    for e in ('visual','anchor','fusion'):
                        if r['mask_sha256'][e] not in used:used.add(r['mask_sha256'][e]);names.append(e)
                    chosen=[]
                    for mode in ('canonical','native_interp'):
                        scores=ref['scores'][mode];assert len(scores)==len(names) and np.isfinite(scores).all();valid=[i for i,e in enumerate(names) if r['evidence'][e]['area']>0];best=max(valid,key=lambda i:scores[i]) if valid else 0
                        if not r['source_valid'] or (r['evidence'][r['baseline']]['area']>0 and scores[best]<=scores[0]+.05):best=0
                        chosen.append(names[best])
                    assert (chosen[0] if chosen[0]==chosen[1] else r['baseline'])==ref['expert']
            checks['phases'][phase]['same_current_bank_external_reference']='passed'
            checks['phases'][phase]['external_decisions_replayed_from_saved_scores']='passed'
    path=R/('final_validation.json' if len(checks['phases'])==2 else 'partial_validation.json');path.write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
if __name__=='__main__':main()
