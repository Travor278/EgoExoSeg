"""Descriptive contrasts for declared secondary variants; never reselect a model."""
import json,copy
from data_utils import R,read,score
out={'scope':'Declared secondary variants; descriptive comparisons, no target selection or equivalence claim','phases':{}}
for phase in ('exo2exo','exo2ego'):
    path=R/(phase+'_results.json')
    if not path.exists():continue
    rows=read(phase);choices={(r['video_id'],r['obj_id']):r['choices'] for r in map(json.loads,(R/(phase+'_predictions.jsonl')).read_text().splitlines())};result={}
    for a,b in [('local20_matched','frozen_cycle'),('local20_matched','primary'),('local20_matched','omama_reference'),('global_multiscale_matched','frozen_cycle')]:
        if b=='omama_reference' and phase!='exo2exo':continue
        refs=copy.deepcopy(rows);chosen=[]
        for r in refs:
            c=choices[(r['video_id'],r['obj_id'])];chosen.append(c[a])
            if b=='omama_reference':r['metrics'][b]=r['frozen_omama_reference'];r['mask_sha256'][b]=r['mask_sha256'][r['frozen_omama_expert']];r['baseline']=b
            else:r['baseline']=c[b]
        result[a+' minus '+b]=score(refs,chosen)
    out['phases'][phase]=result
(R/'secondary_contrasts.json').write_text(json.dumps(out,indent=2))
for phase,items in out['phases'].items():
    for name,v in items.items():print(phase,name,round(v['delta_pp'],4),[round(x,4) for x in v['ci95_pp']])
