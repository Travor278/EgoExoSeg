from pathlib import Path
import json,hashlib,sys
import numpy as np
from data_utils import R,read,score
def main():
    m=json.loads((R/'manifest.json').read_text());sel=R/'selection.json';assert hashlib.sha256(sel.read_bytes()).hexdigest()==m['frozen_selection_sha256'];ds=json.loads((R/'temporal_expansion_manifest.json').read_text());assert ds['additional_pairs']==6534 and ds['new_independent_takes']==0 and ds['original_prompt_and_target_mask_bytes_preserved'];checks={'frozen_model_and_dataset_contract':'passed','new_pairs':6534,'independent_takes':19,'new_independent_takes':0}
    p=R/'exo2exo_results.json'
    if not p.exists():return
    rows=read('exo2exo');assert len(rows)==6534 and len({r['take_id'] for r in rows})==19
    original=json.loads((R/'private_inputs.json').read_text(encoding='utf-8-sig'))['old'];oldpairs={(r['candidate_meta']['source_candidate_id'],r['candidate_meta']['target_gt_image']) for r in original.values()};expected={(r['candidate_meta']['source_candidate_id'],r['candidate_meta']['target_gt_image']) for r in json.loads((R/'temporal_expanded_annotation.json').read_text()).values()}
    actual={(r['expanded_pair_metadata']['source_candidate_id'],r['expanded_pair_metadata']['target_gt_image']) for r in rows};assert actual==expected and actual.isdisjoint(oldpairs)
    result=json.loads(p.read_text());preds={(r['video_id'],r['obj_id']):r['choices'] for r in map(json.loads,(R/'exo2exo_predictions.jsonl').read_text().splitlines())};assert len(preds)==len(rows)
    for name,v in result['methods'].items():
        if name=='omama_reference':continue
        choices=[preds[(r['video_id'],r['obj_id'])][name] for r in rows];computed=score(rows,choices);assert np.allclose(computed['frame'],v['frame'],atol=1e-12,rtol=0) and np.allclose(computed['ci95_pp'],v['ci95_pp'],atol=1e-10,rtol=0)
        for key,field in [('primary','integrated_primary'),('local20_matched','integrated_local20'),('frozen_cycle','integrated_frozen_cycle')]:
            if name==key:assert choices==[r[field] for r in rows]
    reference={}
    for rank in range(4):
        p=R/f'runs/reference/rank{rank}/references.jsonl';receipt=json.loads((p.parent/'receipt.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['records_sha256'] and receipt['witness']['max_score_error']<1e-4;reference.update({(r['video_id'],r['obj_id']):r for r in map(json.loads,p.read_text().splitlines())})
    assert len(reference)==6534
    for r in rows:
        ref=reference[(r['video_id'],r['obj_id'])];assert ref['mask_sha256']==r['mask_sha256'] and ref['metrics']==r['frozen_omama_reference']==r['metrics'][ref['expert']]
        assert ref['scores'] is not None and not ref['reused_identical_bank']
        names=[r['baseline']];used={r['mask_sha256'][r['baseline']]}
        for e in ('visual','anchor','fusion'):
            if r['mask_sha256'][e] not in used:used.add(r['mask_sha256'][e]);names.append(e)
        chosen=[]
        for mode in ('canonical','native_interp'):
            scores=ref['scores'][mode];assert len(scores)==len(names) and np.isfinite(scores).all();valid=[i for i,e in enumerate(names) if r['evidence'][e]['area']>0];best=max(valid,key=lambda i:scores[i]) if valid else 0
            if not r['source_valid'] or (r['evidence'][r['baseline']]['area']>0 and scores[best]<=scores[0]+.05):best=0
            chosen.append(names[best])
        assert (chosen[0] if chosen[0]==chosen[1] else r['baseline'])==ref['expert']
    refs=[{**r,'metrics':{**r['metrics'],'external':r['frozen_omama_reference']},'mask_sha256':{**r['mask_sha256'],'external':r['mask_sha256'][r['frozen_omama_expert']]}} for r in rows];external=score(refs,['external']*len(rows));assert np.allclose(external['frame'],result['methods']['omama_reference']['frame'],atol=1e-12,rtol=0) and np.allclose(external['ci95_pp'],result['methods']['omama_reference']['ci95_pp'],atol=1e-10,rtol=0)
    checks.update(coverage='exact',pair_disjointness_and_identity='passed',actual_routes_and_frame_bootstrap='passed',same_candidate_external_reference='passed',baseline_zero_iou_objects=sum(r['metrics'][r['baseline']][0]==0 for r in rows));(R/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
if __name__=='__main__':main()
