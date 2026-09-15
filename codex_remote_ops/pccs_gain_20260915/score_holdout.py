"""Frozen gate inference, then independent test scoring. No fitting API used."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,re,time
import numpy as np,torch,yaml,joblib
from pycocotools import mask as mu
from aligned_model import AlignedMatcher
from gain_features import prediction_inputs,vector
from evaluate_confirmation import masks,select,metrics,Accumulator
R=Path(__file__).parent;G=R/'gate_training';H=G/'holdout'
def status(**x):
 p=H/'scoring_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({**x,'updated_at':time.time()}));tmp.replace(p)
def main():
 manifest=json.loads((H/'frozen_manifest.json').read_text());checkpoint=H/'frozen_gate.joblib'
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
 assert hashlib.sha256((R/'gain_features.py').read_bytes()).hexdigest()==manifest['feature_code_sha256']
 assert hashlib.sha256((H/'annotation.json').read_bytes()).hexdigest()==manifest['dataset_plan']['annotation_sha256']
 gate=joblib.load(checkpoint);threshold=manifest['threshold']
 ann=json.loads((H/'annotation.json').read_text());rows=[json.loads(s) for s in (H/'runs/exo2ego/per_object.jsonl').read_text().splitlines()]
 expected={(str(v),str(o)) for v,rec in ann.items() for o in rec['objects']}
 assert len(rows)==len(expected)==manifest['dataset_plan']['objects'] and {(r['video_id'],r['obj_id']) for r in rows}==expected
 assert len({r['bank_file'] for r in rows})==len(rows)
 image_root=Path(yaml.safe_load((H/'runtime.yaml').read_text())['data']['images'])
 torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 matcher=AlignedMatcher('cuda');acc=defaultdict(Accumulator);out=H/'per_object.jsonl';start=time.monotonic();max_error=0.
 if out.exists():raise RuntimeError('Fresh holdout scoring output required')
 with out.open('w') as handle:
  for i,r in enumerate(rows):
   rec=ann[r['video_id']];q=rec['prompt']['first_frame_image'];t=rec['video_path'];q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t
   qm,names,ms,digest=masks(r);nonempty=[bool(m.any()) for m in ms]
   if qm.any():scores,geometry=matcher.score(image_root/q,image_root/t,qm,ms,'exo2ego','canonical',ablate=True)
   else:scores={k:[0.]*len(names) for k in ('omama_learned','omama_context_cross_zero','dino2_object','dino2_object_context')};geometry={}
   inputs=prediction_inputs(r,scores,names);allowed=[j for j in range(1,len(names)) if qm.any() and nonempty[j]]
   features=[]
   for j in allowed:
    x,columns=vector(inputs,j);assert columns==manifest['feature_columns'];features.append(x)
   predicted=gate.predict(np.asarray(features)) if features else np.asarray([])
   assert np.isfinite(predicted).all()
   best=int(np.argmax(predicted)) if len(predicted) else None
   chosen=names[allowed[best]] if best is not None and predicted[best]>threshold else 'baseline'
   secondary=select(scores['omama_learned'],names,nonempty,.05) if qm.any() else 'baseline'
   selections={'baseline':'baseline','learned_gate':chosen,'omama_margin005':secondary}
   # Prediction is complete before target annotation decoding or metric access.
   gt=mu.decode(rec['objects'][r['obj_id']]['segmentation']);gt=torch.nn.functional.interpolate(torch.from_numpy(gt.copy()).float()[None,None],size=(1024,1024),mode='nearest')[0,0].numpy().astype(bool)
   for name,m in zip(names,ms):
    expert=r['best_expert'] if name=='baseline' else name;value=float((gt&m.astype(bool)).sum())/(float((gt|m.astype(bool)).sum())+1e-6)
    error=abs(value-metrics(r,expert)[0]);max_error=max(max_error,error);assert error<1e-5
   record={'video_id':r['video_id'],'obj_id':r['obj_id'],'candidate_mask_sha256':digest,'names':names,'selections':selections,
           'predicted_gain':{names[j]:float(v) for j,v in zip(allowed,predicted)},'scores':scores,'geometry':geometry,'metrics':{}}
   for method,name in selections.items():
    expert=r['best_expert'] if name=='baseline' else name;vals=metrics(r,expert);record['metrics'][method]=vals
    acc[method].add(r['video_id'],vals,r['IoU'],expert!=r['best_expert'],'changed' if expert!=r['best_expert'] else 'baseline')
   handle.write(json.dumps(record,allow_nan=False)+'\n');handle.flush()
   if (i+1)%10==0 or i+1==len(rows):status(state='running',done=i+1,total=len(rows),eta_seconds=(time.monotonic()-start)/(i+1)*(len(rows)-i-1))
 # Cluster resampling preserves the dependence between frames in a take.
 take_by_vid={}
 for k,rec in ann.items():
  q=rec['prompt']['first_frame_image'];q=q[0] if isinstance(q,list) else q;take_by_vid[k]=re.search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',q).group()
 takes=sorted(set(take_by_vid.values()));assert set(takes)==set(manifest['dataset_plan']['takes'])
 rng=np.random.default_rng(20260915);sample=rng.integers(0,len(takes),size=(10000,len(takes)));comparisons={}
 for method in ('learned_gate','omama_margin005'):
  grouped=defaultdict(list)
  for vid,p in acc[method].pairs.items():
   b=acc['baseline'].pairs[vid];grouped[take_by_vid[vid]].append(p[1]/p[0]-b[1]/b[0])
  assert all(len(grouped[t])==8 for t in takes)
  delta=np.asarray([np.mean(grouped[t]) for t in takes]);comparisons[method]={'delta_frame_iou_pp':float(100*delta.mean()),'paired_take_bootstrap_95ci_pp':(100*np.quantile(delta[sample].mean(1),[.025,.975])).tolist()}
 result={'scope':'Independent512-pair965-object64-take Exo2Ego test cohort; all earlier exploratory/confirmation takes excluded; not full test',
 'coverage':'exact','cache_metric_iou_max_abs_error':max_error,'frozen_manifest':manifest,'methods':{k:v.result() for k,v in acc.items()},'paired_comparison':comparisons,
 'checkpoint_unchanged':hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['checkpoint_sha256'],'load_receipts':matcher.load_receipts,'per_object_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 (H/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 lines=['# Frozen gain-gate independent test','',result['scope'],'',f"Frozen {manifest['model']}, predicted-gain threshold {threshold}; fitted only on official train, calibrated on disjoint train takes.",'','| Method | Frame IoU (%) | Delta (pp) | 95% take bootstrap CI (pp) |','|---|---:|---:|---|']
 for m,v in result['methods'].items():
  c=comparisons.get(m);lines.append(f"| {m} | {100*v['frame']['IoU']:.4f} | {c['delta_frame_iou_pp'] if c else 0:.4f} | {c['paired_take_bootstrap_95ci_pp'] if c else '-'} |")
 lines+=['','Primary comparison: learned gate vs PCCS. Fixed O-MaMa margin0.05 is secondary. No test-time parameter selection. Training-internal calibration gains must not be reported as test gains.']
 (H/'REPORT.md').write_text('\n'.join(lines));status(state='complete')
if __name__=='__main__':
 try:main()
 except Exception as e:status(state='failed',error=str(e));raise
