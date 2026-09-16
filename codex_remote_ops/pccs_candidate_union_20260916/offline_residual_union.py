"""Stage2 descriptive controls; fixed margin, no fitting or target-test selection."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from union_core import catalog,route
R=Path(__file__).parent;P=R.parent/'pccs_candidate_quality_20260916'
def read(root,phase,arm):
    return {(r['video_id'],r['obj_id']):r for r in map(json.loads,(root/'runs'/phase/arm/'per_object.jsonl').read_text().splitlines())}
output={'threshold':.05,'scope':'descriptive TRAIN screen/cal controls, no target fitting','phases':{}}
for phase in ('screen','calibration'):
    base=read(P,phase,'baseline');mnn=read(P,phase,'reliable_points');arms={s:read(R,phase,s) for s in ('residual005','residual010','residual025')};result={}
    pools={s:[(s,rows)] for s,rows in arms.items()};pools['all_union']=[('reliable_points',mnn),*arms.items()]
    for name,extras in pools.items():
        data=defaultdict(list);changed=0;maxerr=0.
        for k,b in base.items():
            cs,ys,fallback,err=catalog([('baseline',b),*[(s,rows[k]) for s,rows in extras]]);selected=route(cs,fallback);maxerr=max(maxerr,err);oracle=max(v[0] for v in ys.values());assert oracle+1e-7>=b['metrics']['oracle'][0]
            data[k[0]].append([b['metrics']['consensus'][0],ys[selected][0],b['metrics']['oracle'][0],oracle]);changed+=selected!=fallback
        v=np.mean([np.mean(x,axis=0) for x in data.values()],axis=0)*100
        result[name]={'baseline_final':float(v[0]),'union_final':float(v[1]),'baseline_oracle':float(v[2]),'union_oracle':float(v[3]),'delta_final_pp':float(v[1]-v[0]),'delta_oracle_pp':float(v[3]-v[2]),'changed':changed,'duplicate_score_max_error':maxerr}
    output['phases'][phase]=result
(R/'stage2_results.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
