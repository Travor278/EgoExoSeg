"""Check one illustrative case against raw masks, frozen calibrators and full-run scalars."""
from pathlib import Path
import sys,json,hashlib,gzip
R=Path(__file__).parent;W=R/'walkthrough_exoego';P=R.parent/'pccs_roi_context_20260917'
sys.path.insert(0,str(R.parent/'pccs_dense_context_20260917/local_deps'))
import numpy as np,joblib
from roi_policy import features,permitted,predict
row=json.loads((W/'record.json').read_text());dec=json.loads((W/'decisions.json').read_text());science=json.loads((W/'science.json').read_text());key=(row['video_id'],row['obj_id'])
with np.load(W/'bank.npz') as z:
    masks={e:np.unpackbits(z[e])[:int(np.prod(z[e+'_shape']))].reshape(z[e+'_shape']).astype(np.uint8) for e in ('visual','anchor','fusion')}
assert {e:hashlib.sha256(m.tobytes()).hexdigest() for e,m in masks.items()}==row['mask_sha256']==science['mask_sha256']
sel=json.loads((R/'selection.json').read_text());scores={}
for group,choice in [('local15','primary'),('local20','local20_matched')]:
    cfg=next(x for x in sel['matched_controls'] if x['group']==group);p=P/'fitted'/Path(cfg['checkpoint']).name;assert hashlib.sha256(p.read_bytes()).hexdigest()==cfg['sha256'];model=joblib.load(p)
    result=predict(row,model,group,cfg['threshold'],cfg['gate']);assert result==dec['choices'][choice]
    scores[group]={'chosen':result,'threshold':cfg['threshold'],'checkpoint_sha256':cfg['sha256'],'candidates':{e:{'permitted':permitted(row,e,cfg['gate']),'predicted_gain':float(model.predict(features(row,e,group)[None])[0])} for e in masks if e!=row['baseline']}}
    without_gt={**row,'metrics':None};assert predict(without_gt,model,group,cfg['threshold'],cfg['gate'])==result
full={}
for name,p in [('roi_seed1',R/'full1_compact.jsonl.gz'),('roi_seed2',R/'full2_compact.jsonl.gz'),('shared_reference',R.parent/'pccs_omama_shared_full_20260918/full1_paired_compact.jsonl.gz')]:
    with gzip.open(p,'rt') as f:x=next(x for x in map(json.loads,f) if (x['video_id'],x['obj_id'])==key)
    full[name]=x
assert all(full[n]['mask_sha256']==row['mask_sha256'] for n in ('roi_seed1','shared_reference'))
assert full['roi_seed1']['choices']['primary']==dec['choices']['primary']
(W/'scores.json').write_text(json.dumps(scores,indent=2))
(W/'full_crosscheck.json').write_text(json.dumps(full,indent=2))
print(json.dumps({'scores':scores,'choices_by_run':{k:v['choices'] for k,v in full.items()},'science_choices':science['choices'],'evidence':row['evidence']},indent=2))
