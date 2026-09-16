from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
from data_utils import read
R=Path(__file__).parent;rows=read('exo2exo');keys={(r['video_id'],r['obj_id']) for r in rows};assert len(keys)==len(rows)==1094
old={ (r['video_id'],r['obj_id']):r for r in map(json.loads,(R.parent/'pccs_candidate_union_20260916/runs/exo2exo/baseline/per_object.jsonl').read_text().splitlines())};assert keys==old.keys()
for r in rows:
    p=old[(r['video_id'],r['obj_id'])];assert r['mask_sha256']==p['mask_sha256'] and r['baseline']==p['pccs_expert'];np.testing.assert_allclose(r['metrics'][r['baseline']],p['metrics']['pccs'],atol=1e-7,rtol=0)
def verify(chosen,expected):
    groups=defaultdict(list);pair=defaultdict(list)
    for r in rows:
        e=chosen[(r['video_id'],r['obj_id'])];v=r['metrics'][e];b=r['metrics'][r['baseline']];pair[r['video_id']].append(v);groups[r['take_id']].append(v[0]-b[0])
    frame=np.mean([np.mean(v,axis=0) for v in pair.values()],axis=0);np.testing.assert_allclose(frame,expected['frame'],atol=1e-12,rtol=0)
    sums=np.asarray([sum(v) for v in groups.values()]);counts=np.asarray([len(v) for v in groups.values()]);draw=np.random.default_rng(20260917).integers(0,len(counts),size=(10000,len(counts)));ci=np.quantile(sums[draw].sum(1)/counts[draw].sum(1),[.025,.975])*100;np.testing.assert_allclose(ci,expected['ci95_pp'],atol=1e-10,rtol=0)
    return {'frame_iou':float(frame[0]),'delta_pp':float(sums.sum()/counts.sum()*100),'ci95_pp':ci.tolist()}
p1={ (r['video_id'],r['obj_id']):r['selected']['primary'] for r in map(json.loads,(R/'exo2exo_predictions.jsonl').read_text().splitlines())};p2={ (r['video_id'],r['obj_id']):r['selected'] for r in map(json.loads,(R/'gate_predictions.jsonl').read_text().splitlines())};p3={ (r['video_id'],r['obj_id']):r['selected']['primary'] for r in map(json.loads,(R/'lift_study/predictions.jsonl').read_text().splitlines())}
assert all(p1[(r['video_id'],r['obj_id'])]==r['integrated_primary'] for r in rows)
audit={'coverage':'exact','pairs':1094,'objects':1094,'takes':len({r['take_id'] for r in rows}),'final_bank_historical_hashes_exact':True,'original_pccs_zero_iou_objects':sum(r['metrics'][r['baseline']][0]==0 for r in rows),'reference_repairs_applied':json.loads((R/'reference_updates.json').read_text())['count'],'r1_native_bridge_exact':True,'r1':verify(p1,json.loads((R/'exo2exo_results.json').read_text())['methods']['primary']),'gate':verify(p2,json.loads((R/'gate_target_results.json').read_text())['methods']['primary']),'lift':verify(p3,json.loads((R/'lift_study/target_results.json').read_text())['methods']['primary'])}
(R/'final_validation.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
