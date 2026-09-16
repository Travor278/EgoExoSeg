"""Refresh only the external comparison for documented historical bank drift."""
from pathlib import Path
import json,hashlib,importlib.util,sys
import numpy as np,torch,yaml
from data_utils import R,read
Q=R.parent/'pccs_candidate_quality_20260916';U=R.parent/'pccs_candidate_union_20260916';rows=read('exo2exo');pending=[r for r in rows if r['frozen_omama_reference'] is None]
if not pending:
    (R/'reference_updates.json').write_text(json.dumps({'updates':[],'count':0}));print('NO_REFERENCE_REPAIR_NEEDED');sys.exit(0)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
spec=importlib.util.spec_from_file_location('frozen_omama_reference',Q/'aligned_model.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);matcher=module.AlignedMatcher('cuda')
manifest=json.loads((R/'manifest.json').read_text());ann={};cfg=None
for shard in manifest['phases']['exo2exo']['shards']:ann.update(json.loads(Path(shard['annotation']).read_text()));cfg=yaml.safe_load(Path(shard['config']).read_text())
images=Path(cfg['data']['images']);old={(r['video_id'],r['obj_id']):r for r in map(json.loads,(U/'runs/exo2exo/baseline/per_object.jsonl').read_text().splitlines())}
def run(row,path):
    z=np.load(path)
    def unpack(e):return np.unpackbits(z[e])[:int(np.prod(z[e+'_shape']))].reshape(z[e+'_shape']).astype(np.uint8)
    qm=unpack('query');masks={e:unpack(e) for e in ('visual','anchor','fusion')};assert {e:hashlib.sha256(v.tobytes()).hexdigest() for e,v in masks.items()}==row['mask_sha256'];base=row['baseline'];names=[base];used={row['mask_sha256'][base]}
    for e in ('visual','anchor','fusion'):
        h=row['mask_sha256'][e]
        if h not in used:used.add(h);names.append(e)
    ms=[masks[e] for e in names];a=ann[row['video_id']];q=a['prompt']['first_frame_image'];t=a['video_path'];q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t;chosen=[];scores={}
    for mode in ('canonical','native_interp'):
        values,_=matcher.score(images/q,images/t,qm,ms,'exo2ego',mode) if qm.any() else ({'omama_learned':[0.]*len(ms)},None);s=values['omama_learned'];scores[mode]=s;valid=[i for i,m in enumerate(ms) if m.any()];best=max(valid,key=lambda i:s[i]) if valid else 0
        if not qm.any() or (ms[0].any() and s[best]<=s[0]+.05):best=0
        chosen.append(names[best])
    return chosen[0] if chosen[0]==chosen[1] else base,scores
witness=next(r for r in rows if not r.get('historical_drift',False));prior=old[(witness['video_id'],witness['obj_id'])];expert,ws=run(witness,prior['bank_file']);assert expert==prior['consensus_expert'];error=max(float(np.max(np.abs(np.asarray(ws[m])-prior['scores'][m]))) for m in ws);assert error<1e-4,('Reference witness score drift',error)
fixes=[]
for row in pending:
    expert,scores=run(row,row['bank_file']);fixes.append({'video_id':row['video_id'],'obj_id':row['obj_id'],'expert':expert,'metrics':row['metrics'][expert],'mask_sha256':row['mask_sha256'],'scores':scores})
(R/'reference_updates.json').write_text(json.dumps({'updates':fixes,'count':len(fixes),'witness_max_score_error':error,'same_current_bank':True,'used_only_for_external_reference':True},indent=2));print('REFERENCE_REPAIRED',len(fixes),flush=True)
