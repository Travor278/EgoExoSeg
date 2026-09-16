"""Independent aggregation and trace checks for the native-PCCS experiment."""
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib
import numpy as np
R=Path(__file__).parent
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
audit={}
for phase in ('smoke','screen','calibration'):
    rows=[];receipts=[]
    for rank in range(4):
        p=R/'runs'/phase/f'rank{rank}'/'records.jsonl';receipt=load(p.parent/'receipt.json');assert receipt['coverage']=='exact' and receipt['records_sha256']==hashlib.sha256(p.read_bytes()).hexdigest();rows.extend(map(json.loads,p.read_text().splitlines()));receipts.append(receipt)
    result=load(R/(phase+'_results.json'));assert len(rows)==len({(r['video_id'],r['obj_id']) for r in rows})==result['objects']
    pairs=defaultdict(list)
    for r in rows:pairs[r['video_id']].append(r)
    assert len(pairs)==result['pairs'];maxerr=0.
    for name,value in result['methods'].items():
        metrics=np.mean([np.mean([r['methods'][name]['metrics'] for r in group],axis=0) for group in pairs.values()],axis=0);maxerr=max(maxerr,float(np.max(np.abs(metrics-value['frame']))))
        g=defaultdict(list)
        for group in pairs.values():g[group[0]['take_id']].append(float(np.mean([r['methods'][name]['metrics'][0]-r['methods']['baseline']['metrics'][0] for r in group])))
        sums=np.array([sum(v) for v in g.values()]);counts=np.array([len(v) for v in g.values()]);draw=np.random.default_rng(20260916).integers(0,len(counts),size=(10000,len(counts)));ci=np.quantile(sums[draw].sum(1)/counts[draw].sum(1),[.025,.975])*100
        np.testing.assert_allclose(ci,value['ci95_pp'],atol=1e-12,rtol=0)
    assert maxerr<1e-12
    for r in rows:assert r['methods']['baseline']['expert']==r['methods']['zero']['expert'] and r['methods']['baseline']['metrics']==r['methods']['zero']['metrics']
    veto=[r for r in rows if r['original_fusion_accepted'] and r['fusion_context_recheck']];fallback=[r for r in rows if not r['original_fusion_accepted']]
    audit[phase]={'objects':len(rows),'pairs':len(pairs),'takes':len({r['take_id'] for r in rows}),'max_metric_error':maxerr,'bootstrap_recomputed':True,'lambda_zero_parity':True,'capture_on_off_pairs':sum(r['capture_on_off_pairs'] for r in receipts),'prior_candidate_hash_checks':all(r['all_prior_candidates_exact'] for r in receipts),'zero_iou_objects_preserved':sum(r['methods']['baseline']['metrics'][0]==0 for r in rows),'original_fallback_objects':len(fallback),'fallback_with_reliable_visual_or_anchor':sum(any(r['context'][e]['reliability']>=.2 for e in ('visual','anchor')) for r in fallback),'review_triggered_objects':len(veto),'triggered_original_mean_object_iou':float(np.mean([r['methods']['baseline']['metrics'][0] for r in veto])) if veto else None,'triggered_review_mean_object_iou':float(np.mean([r['methods']['review']['metrics'][0] for r in veto])) if veto else None,'fallback_reasons':dict(Counter(r['methods']['baseline']['reason'] for r in fallback))}
assert load(R/'selection.json')['selected']=='baseline' and not (R/'exo2exo_results.json').exists()
(R/'final_validation.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
