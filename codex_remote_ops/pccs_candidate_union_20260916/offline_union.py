from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np
from union_core import catalog,route
R=Path(__file__).parent;P=R.parent/'pccs_candidate_quality_20260916'
def read(phase,arm):return {(r['video_id'],r['obj_id']):r for r in (json.loads(s) for s in (P/'runs'/phase/arm/'per_object.jsonl').read_text(encoding='utf8').splitlines())}
def aggregate(rows,key):
    pairs=defaultdict(list)
    for r in rows:pairs[r['video_id']].append(r[key])
    return float(np.mean([np.mean(v) for v in pairs.values()]))
out={'stage':'1_cached_multipoint_union','threshold':.05,'refitting':False,'baseline':'previous baseline-arm geometry-consensus output, retained exactly','phases':{}}
for phase in ('screen','calibration'):
    base=read(phase,'baseline');extra=read(phase,'reliable_points');assert base.keys()==extra.keys();rows=[];maxerr=0
    for k,b in base.items():
        cs,ys,fallback,err=catalog([('baseline',b),('reliable_points',extra[k])]);maxerr=max(maxerr,err);chosen=route(cs,fallback)
        oracle=max(v[0] for v in ys.values());assert oracle+1e-7>=b['metrics']['oracle'][0]
        rows.append({'video_id':k[0],'obj_id':k[1],'baseline':b['metrics']['consensus'][0],'union':ys[chosen][0],'old_oracle':b['metrics']['oracle'][0],'union_oracle':oracle,'switched':chosen!=fallback})
    values={k:aggregate(rows,k)*100 for k in ('baseline','union','old_oracle','union_oracle')}
    out['phases'][phase]={'pairs':len({r['video_id'] for r in rows}),'objects':len(rows),'frame_iou_percent':values,'delta_final_pp':values['union']-values['baseline'],'delta_oracle_pp':values['union_oracle']-values['old_oracle'],'changed':sum(r['switched'] for r in rows),'duplicate_score_max_error':maxerr}
(R/'stage1_results.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
