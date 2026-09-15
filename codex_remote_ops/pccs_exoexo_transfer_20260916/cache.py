from pathlib import Path
import hashlib,numpy as np
def masks(row):
 p=Path(row['bank_file']);z=np.load(p)
 def mask(k):return np.unpackbits(z[k])[:int(np.prod(z[k+'_shape']))].reshape(z[k+'_shape']).astype(np.uint8)
 names=[];ms=[];seen=set()
 for k in ('baseline','visual','anchor','fusion'):
  m=mask(k);digest=hashlib.sha256(m.tobytes()).hexdigest()
  if digest in seen:continue
  seen.add(digest);names.append(k);ms.append(m)
 return mask('query'),names,ms,hashlib.sha256(p.read_bytes()).hexdigest()
def select(scores,names,nonempty,margin=.05):
 allowed=[i for i,v in enumerate(nonempty) if v]
 if not allowed:return 'baseline'
 assert np.isfinite(scores).all()
 best=max(allowed,key=lambda i:scores[i])
 return 'baseline' if nonempty[0] and scores[best]<=scores[0]+margin else names[best]
def metrics(r,e):
 return [r[k] for k in ('IoU','Dice','shape_acc','location_score')] if e==r['best_expert'] else [r[k+'_'+e] for k in ('iou','dice','shape_acc','location_score')]
