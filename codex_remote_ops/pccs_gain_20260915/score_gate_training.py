from pathlib import Path
import hashlib,json,sys,time
import numpy as np,torch
from pycocotools import mask as mu
from aligned_model import AlignedMatcher
from gain_features import prediction_inputs
R=Path(__file__).parent;G=R/'gate_training'
sys.path.insert(0,str(R.parent/'context_pccs_20260914/ablations_20260915'))
from offline_ablation import metrics
def status(**x):
 p=G/'scoring_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({**x,'updated_at':time.time()}));tmp.replace(p)
try:
 ann=json.loads((G/'combined.json').read_text());plan=json.loads((G/'plan.json').read_text())
 split_by_key={k:s for s,v in plan['splits'].items() for k in v['keys']}
 rows=[json.loads(s) for s in (G/'runs/exo2ego/per_object.jsonl').read_text().splitlines()]
 expected={(str(v),str(o)) for v,rec in ann.items() for o in rec['objects']}
 assert len(rows)==len(expected) and {(r['video_id'],r['obj_id']) for r in rows}==expected
 out=G/'training_records.jsonl'
 if out.exists():raise RuntimeError('Fresh training scores required')
 torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=AlignedMatcher('cuda');start=time.monotonic();max_error=0.
 with out.open('w') as handle:
  for i,r in enumerate(rows):
   rec=ann[r['video_id']];p=Path(r['bank_file']);z=np.load(p)
   def unpack(k):return np.unpackbits(z[k])[:int(np.prod(z[k+'_shape']))].reshape(z[k+'_shape']).astype(np.uint8)
   names=[];ms=[];seen=set()
   for k in ('baseline','visual','anchor','fusion'):
    m=unpack(k);digest=hashlib.sha256(m.tobytes()).hexdigest()
    if digest not in seen:seen.add(digest);names.append(k);ms.append(m)
   qm=unpack('query');q=rec['prompt']['first_frame_image'];t=rec['video_path'];q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t
   if qm.any():scores,_=model.score(G/'images'/q,G/'images'/t,qm,ms,'exo2ego','canonical',ablate=True)
   else:scores={k:[0.]*len(names) for k in ('omama_learned','omama_context_cross_zero','dino2_object','dino2_object_context')}
   inputs=prediction_inputs(r,scores,names)
   # Labels remain in a distinct field. Only fit/calibration consume them.
   gt=mu.decode(rec['objects'][r['obj_id']]['segmentation']);gt=torch.nn.functional.interpolate(torch.from_numpy(gt.copy()).float()[None,None],size=(1024,1024),mode='nearest')[0,0].numpy().astype(bool)
   labels=[]
   for name,m in zip(names,ms):
    expert=r['best_expert'] if name=='baseline' else name;vals=metrics(r,expert)
    value=float((gt&m.astype(bool)).sum())/(float((gt|m.astype(bool)).sum())+1e-6);diff=abs(value-vals[0]);max_error=max(max_error,diff);assert diff<1e-5
    labels.append(vals)
   item={'video_id':r['video_id'],'obj_id':r['obj_id'],'split':split_by_key[r['video_id']],
         'inputs':inputs,'nonempty':[bool(m.any()) for m in ms],'query_nonempty':bool(qm.any()),'labels':labels,'mask_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
   handle.write(json.dumps(item,allow_nan=False)+'\n');handle.flush()
   if (i+1)%10==0 or i+1==len(rows):status(state='running',done=i+1,total=len(rows),eta_seconds=(time.monotonic()-start)/(i+1)*(len(rows)-i-1))
 (G/'scoring_receipt.json').write_text(json.dumps({'coverage':'exact','objects':len(rows),'max_cache_metric_iou_error':max_error,'load_receipts':model.load_receipts,'records_sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))
 status(state='complete')
except Exception as e:status(state='failed',error=str(e));raise
