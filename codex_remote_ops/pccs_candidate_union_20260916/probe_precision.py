"""Diagnose cached-score drift without relaxing the original parity threshold."""
import json
import numpy as np
import torch,yaml
from build_dataset import R,P,SOURCES,load_rows,unpack,embed
from union_core import catalog
from aligned_model import AlignedMatcher
from pathlib import Path
phase='screen';rows={s:load_rows(P if s in SOURCES[:2] else R,phase,s) for s in SOURCES}
key=sorted(rows['baseline'])[0];records=[(s,rows[s][key]) for s in SOURCES];cs,_,_,_=catalog(records);pool={}
for source,row in records:
    z=np.load(row['bank_file'])
    if source=='baseline':qm=unpack(z,'query')
    for e in ('visual','anchor','fusion'):pool[row['mask_sha256'][e]]=unpack(z,e)
spec=json.loads((R/'manifest.json').read_text())['phases'][phase];ann=json.loads(Path(spec['annotation']).read_text());cfg=yaml.safe_load(Path(spec['config']).read_text());images=Path(cfg['data']['images']);rec=ann[key[0]]
q=rec['prompt']['first_frame_image'];t=rec['video_path'];q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t
matcher=AlignedMatcher('cuda');masks=[pool[c['id']] for c in cs];result={'key':key,'threshold':1e-4,'cases':{}}
base=records[0][1];base_masks=[pool[base['mask_sha256'][base['pccs_expert'] if n=='baseline' else n]] for n in base['candidate_names']]
indices={c['id']:i for i,c in enumerate(cs)};groups=[[indices[row['mask_sha256'][row['pccs_expert'] if n=='baseline' else n]] for n in row['candidate_names']] for _,row in records]
for gemm,cudnn in ((False,False),(False,True),(True,False),(True,True)):
    torch.backends.cuda.matmul.allow_tf32=gemm;torch.backends.cudnn.allow_tf32=cudnn;values={}
    for mi,mode in enumerate(('canonical','native_interp')):
        _,_,scores=embed(matcher,images/q,images/t,qm,masks,mode,groups);_,_,original=embed(matcher,images/q,images/t,qm,base_masks,mode)
        values[mode]={'union':float(np.max(np.abs(scores-np.asarray([c['scores'][mi] for c in cs])))),'original_group':float(np.max(np.abs(original-np.asarray(base['scores'][mode]))))}
    result['cases'][f'{gemm},{cudnn}']=values
passing=[k for k,v in result['cases'].items() if max(x['union'] for x in v.values())<1e-4]
result['selected']=passing[0] if passing else None
(R/'precision_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
assert passing,'No precision setting matches cached union scores; stop for diagnosis'
