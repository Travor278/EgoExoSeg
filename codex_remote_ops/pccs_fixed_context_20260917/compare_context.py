"""Paired descriptive contrasts; no re-selection or refitting on target."""
import json,copy
from data_utils import R,read,score
PAIRS=(('px100_box_matched','cycle_matched'),('om100_box_matched','cycle_matched'),('primary','cycle_matched'),('primary','wrong_context_primary'),('px100_box_matched','px100_ring_matched'),('px100_box_matched','scale15_box_matched'),('px100_box_matched','scale20_box_matched'),('px100_ring_matched','scale15_ring_matched'),('px100_ring_matched','scale20_ring_matched'),('px100_box_matched','old_ring15_matched'),('px100_box_matched','old_ring20_matched'))
out={'scope':'Descriptive paired geometry contrasts for the stated ablations; no new primary selection or refitting','phases':{}}
for phase in ('exo2ego','exo2exo'):
    p=R/(phase+'_predictions.jsonl')
    if not p.exists():continue
    rows=read(phase);preds={(x['video_id'],x['obj_id']):x['choices'] for x in map(json.loads,p.read_text().splitlines())};comparisons={}
    for a,b in PAIRS:
        refs=copy.deepcopy(rows);choices=[]
        for r in refs:
            c=preds[(r['video_id'],r['obj_id'])];r['baseline']=c[b];choices.append(c[a])
        comparisons[a+' minus '+b]=score(refs,choices)
    out['phases'][phase]=comparisons
(R/'context_comparisons.json').write_text(json.dumps(out,indent=2))
for phase,v in out['phases'].items():
    for k,r in v.items():print(phase,k,round(r['delta_pp'],4),[round(x,4) for x in r['ci95_pp']])
