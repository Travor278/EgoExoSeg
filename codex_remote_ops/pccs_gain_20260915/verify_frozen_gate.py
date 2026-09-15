"""Verify checkpoint round-trip against already-selected calibration results."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np,joblib
from gain_features import vector
R=Path(__file__).parent;G=R/'gate_training';H=G/'holdout'
fit=json.loads((G/'fit_results.json').read_text());manifest=json.loads((H/'frozen_manifest.json').read_text());ckpt=H/'frozen_gate.joblib'
assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==fit['checkpoint_sha256']==manifest['checkpoint_sha256']
model=joblib.load(ckpt);groups=defaultdict(list);changed=0;objects=0
for line in (G/'training_records.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r['split']!='calibration':continue
 allowed=[i for i in range(1,len(r['inputs']['names'])) if r['query_nonempty'] and r['nonempty'][i]]
 xs=[]
 for i in allowed:
  x,columns=vector(r['inputs'],i);assert columns==manifest['feature_columns'];xs.append(x)
 preds=model.predict(np.asarray(xs)) if xs else []
 assert np.isfinite(preds).all()
 best=int(np.argmax(preds)) if len(preds) else None
 idx=allowed[best] if best is not None and preds[best]>manifest['threshold'] else 0
 groups[r['video_id']].append(r['labels'][idx][0]);changed+=idx!=0;objects+=1
score=float(np.mean([np.mean(v) for v in groups.values()]));error=abs(score-fit['selected']['frame_iou'])
assert error<1e-12 and changed==fit['selected']['changed_objects']
out={'checkpoint_sha256':manifest['checkpoint_sha256'],'calibration_replay_error':error,'objects':objects,'pairs':len(groups),'changed':changed,'features':len(manifest['feature_columns']),'state':'passed'}
(H/'checkpoint_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
