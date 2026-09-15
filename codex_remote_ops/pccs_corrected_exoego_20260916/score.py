"""Independent rank-sharded inference from an immutable candidate bank."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np,torch,yaml,joblib
from pycocotools import mask as mu
from aligned_model import AlignedMatcher
from gain_features import prediction_inputs,vector
from cache import masks,metrics,select
R=Path(__file__).parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--world-size',type=int,required=True);args=p.parse_args()
 S=R/'runs'/args.stage;outdir=S/'scored';outdir.mkdir(exist_ok=True,parents=True)
 status_path=outdir/f'rank{args.rank}_status.json'
 def status(**x):
  tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps({**x,'rank':args.rank,'updated_at':time.time()}));tmp.replace(status_path)
 try:
  manifest=json.loads((R/'manifest.json').read_text());frozen=manifest['frozen'];spec=manifest['stages'][args.stage]
  assert hashlib.sha256((R/'frozen_gate.joblib').read_bytes()).hexdigest()==frozen['checkpoint_sha256']
  assert hashlib.sha256((R/'gain_features.py').read_bytes()).hexdigest()==frozen['feature_code_sha256']
  ann=json.loads(Path(spec['annotation']).read_text());assert hashlib.sha256(Path(spec['annotation']).read_bytes()).hexdigest()==spec['annotation_sha256']
  allrows=[json.loads(s) for s in (S/'candidates/exo2ego/per_object.jsonl').read_text().splitlines()]
  expected={(str(v),str(o)) for v,rec in ann.items() for o in rec['objects']}
  assert len(allrows)==len(expected)==spec['objects'] and {(r['video_id'],r['obj_id']) for r in allrows}==expected
  rows=allrows[args.rank::args.world_size];del allrows
  image_root=Path(yaml.safe_load(Path(spec['config']).read_text())['data']['images'])
  output=outdir/f'rank{args.rank}.jsonl'
  if output.exists():raise RuntimeError('Fresh scoring shard required: '+str(output))
  gate=joblib.load(R/'frozen_gate.joblib');torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  matcher=AlignedMatcher('cuda');start=time.monotonic();max_error=0.
  with output.open('w') as handle:
   for i,r in enumerate(rows):
    rec=ann[r['video_id']];q=rec['prompt']['first_frame_image'];t=rec['video_path'];q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t
    qm,names,ms,digest=masks(r);nonempty=[bool(m.any()) for m in ms]
    if qm.any():scores,geometry=matcher.score(image_root/q,image_root/t,qm,ms,'exo2ego','canonical',ablate=True)
    else:scores={k:[0.]*len(names) for k in ('omama_learned','omama_context_cross_zero','dino2_object','dino2_object_context')};geometry={}
    inputs=prediction_inputs(r,scores,names);allowed=[j for j in range(1,len(names)) if qm.any() and nonempty[j]];xs=[]
    for j in allowed:
     x,columns=vector(inputs,j);assert columns==frozen['feature_columns'];xs.append(x)
    preds=gate.predict(np.asarray(xs)) if xs else np.asarray([]);assert np.isfinite(preds).all()
    best=int(np.argmax(preds)) if len(preds) else None
    chosen=names[allowed[best]] if best is not None and preds[best]>frozen['threshold'] else 'baseline'
    secondary=select(scores['omama_learned'],names,nonempty) if qm.any() else 'baseline'
    if qm.any():native_scores,native_geometry=matcher.score(image_root/q,image_root/t,qm,ms,'exo2ego','native_interp',ablate=False)
    else:native_scores={'omama_learned':[0.]*len(names)};native_geometry={}
    native=select(native_scores['omama_learned'],names,nonempty) if qm.any() else 'baseline'
    consensus=native if native==secondary else 'baseline'
    scores['native_omama_learned']=native_scores['omama_learned'];geometry['native_interp']=native_geometry
    selections={'baseline':'baseline','learned_gate':chosen,'omama_margin005':secondary,'native_margin005':native,'geometry_consensus':consensus}
    # Targets become accessible only after every prediction is fixed.
    gt=mu.decode(rec['objects'][r['obj_id']]['segmentation']);gt=torch.nn.functional.interpolate(torch.from_numpy(gt.copy()).float()[None,None],size=(1024,1024),mode='nearest')[0,0].numpy().astype(bool)
    for name,m in zip(names,ms):
     e=r['best_expert'] if name=='baseline' else name;value=float((gt&m.astype(bool)).sum())/(float((gt|m.astype(bool)).sum())+1e-6)
     error=abs(value-metrics(r,e)[0]);max_error=max(max_error,error);assert error<1e-5
    record={'video_id':r['video_id'],'obj_id':r['obj_id'],'candidate_mask_sha256':digest,'names':names,'selections':selections,'predicted_gain':{names[j]:float(v) for j,v in zip(allowed,preds)},'scores':scores,'geometry':geometry,'metrics':{}}
    for method,name in selections.items():record['metrics'][method]=metrics(r,r['best_expert'] if name=='baseline' else name)
    handle.write(json.dumps(record,allow_nan=False)+'\n');handle.flush()
    if (i+1)%10==0 or i+1==len(rows):status(state='running',done=i+1,total=len(rows),eta_seconds=(time.monotonic()-start)/(i+1)*(len(rows)-i-1))
  receipt={'state':'complete','rank':args.rank,'objects':len(rows),'cache_metric_max_error':max_error,'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((R/'frozen_gate.joblib').read_bytes()).hexdigest(),'load_receipts':matcher.load_receipts}
  assert receipt['checkpoint_sha256']==frozen['checkpoint_sha256']
  (outdir/f'rank{args.rank}_receipt.json').write_text(json.dumps(receipt,indent=2));status(state='complete',done=len(rows),total=len(rows),eta_seconds=0)
 except Exception as e:status(state='failed',error=str(e));raise
if __name__=='__main__':main()
