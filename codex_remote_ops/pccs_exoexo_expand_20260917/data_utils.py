from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
R=Path(__file__).parent
def read(phase):
    rows=[]
    for rank in range(4):
        p=R/'runs'/phase/f'rank{rank}'/'records.jsonl';rec=json.loads((p.parent/'receipt.json').read_text());assert rec['coverage']=='exact' and rec['records_sha256']==hashlib.sha256(p.read_bytes()).hexdigest();rows.extend(map(json.loads,p.read_text().splitlines()))
    spec=json.loads((R/'manifest.json').read_text())['phases'][phase];assert len(rows)==spec['objects']==len({(r['video_id'],r['obj_id']) for r in rows});assert len({r['video_id'] for r in rows})==spec['pairs']
    updates=R/'reference_updates.json'
    if phase=='exo2exo' and updates.exists():
        fixes={(r['video_id'],r['obj_id']):r for r in json.loads(updates.read_text())['updates']}
        for row in rows:
            if (row['video_id'],row['obj_id']) in fixes:
                f=fixes[(row['video_id'],row['obj_id'])];assert f['mask_sha256']==row['mask_sha256'];row['frozen_omama_reference']=f['metrics'];row['frozen_omama_expert']=f['expert']
    return rows
def score(rows,chosen):
    pairs=defaultdict(list);groups=defaultdict(list);changed=improved=harmed=0
    for r,e in zip(rows,chosen):
        v=r['metrics'][e];b=r['metrics'][r['baseline']];pairs[r['video_id']].append(v);groups[r['video_id']].append((r['take_id'],v[0]-b[0]));changed+=r['mask_sha256'][e]!=r['mask_sha256'][r['baseline']];improved+=v[0]>b[0]+1e-8;harmed+=v[0]<b[0]-1e-8
    takes=defaultdict(list)
    for group in groups.values():takes[group[0][0]].append(float(np.mean([x[1] for x in group])))
    sums=np.array([sum(v) for v in takes.values()]);counts=np.array([len(v) for v in takes.values()]);draw=np.random.default_rng(20260917).integers(0,len(counts),size=(10000,len(counts)));ci=np.quantile(sums[draw].sum(1)/counts[draw].sum(1),[.025,.975])*100
    return {'frame':np.mean([np.mean(v,axis=0) for v in pairs.values()],axis=0).tolist(),'delta_pp':float(sums.sum()/counts.sum()*100),'ci95_pp':ci.tolist(),'changed':changed,'improved':improved,'harmed':harmed,'pairs':len(pairs),'objects':len(rows),'takes':len(takes)}
