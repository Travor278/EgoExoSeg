"""Freeze an unused test cohort before the gate is fitted or calibrated."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,re,time
R=Path(__file__).parent;G=R/'gate_training';P=R.parent/'context_pccs_20260914'
def take(rec):
 q=rec['prompt']['first_frame_image'];q=q[0] if isinstance(q,list) else q
 return re.search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',q).group()
def key(s):return hashlib.sha256(s.encode()).hexdigest()
audit=json.loads((R/'split_audit.json').read_text());ann=json.loads(Path(audit['test']['path']).read_text())
old=json.loads((P/'ablations_20260915/exo2ego_probe64.json').read_text());first=json.loads((R/'confirmation_plan.json').read_text())
excluded={take(v) for v in old.values()}|set(first['takes']);groups=defaultdict(list)
for k,v in ann.items():
 if take(v) not in excluded:groups[take(v)].append(k)
takes=sorted((t for t,ks in groups.items() if len(ks)>=8),key=lambda t:key('gain-gate-holdout-v1-take:'+t))[:64]
assert len(takes)==64 and not set(takes)&excluded
keys=[k for t in takes for k in sorted(groups[t],key=lambda k:key('gain-gate-holdout-v1-pair:'+k))[:8]]
selected={k:ann[k] for k in keys};p=G/'holdout512.json'
if p.exists():raise RuntimeError('Holdout already frozen')
p.write_text(json.dumps(selected));plan={'frozen_at':time.time(),'pairs':512,'objects':sum(len(v['objects']) for v in selected.values()),'takes':takes,'excluded_takes':sorted(excluded),
 'annotation_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'primary':'the single gate and threshold chosen using gate calibration only; if baseline wins calibration, do not claim a learned model',
 'secondary':['fixed O-MaMa margin0.05','original PCCS'],'test_parameter_tuning':False,'selection':'fixed hash; excludes previous exploratory and confirmation takes',
 'inference':'generate candidates once; same masks for every method','report':'paired take-bootstrap95CI for primary gate gain; exactcoverage and zero skipped required'}
(G/'holdout_plan.json').write_text(json.dumps(plan,indent=2));print('HOLDOUT_FROZEN',len(selected),plan['objects'])
