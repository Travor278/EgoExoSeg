from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,re
import numpy as np
METHODS=('baseline','learned_gate','omama_margin005');METRICS=('IoU','Dice','ContA','LocE')
def aggregate(rows):
 pairs={m:defaultdict(list) for m in METHODS};sums={m:np.zeros(4) for m in METHODS};n=0;counts={m:{'changed':0,'improved':0,'harmed':0} for m in METHODS};seen=set()
 for r in rows:
  identity=(r['video_id'],r['obj_id']);assert identity not in seen,identity;seen.add(identity);n+=1
  for m in METHODS:
   v=np.asarray(r['metrics'][m],dtype=float);assert v.shape==(4,) and np.isfinite(v).all();sums[m]+=v;pairs[m][r['video_id']].append(v)
   counts[m]['changed']+=r['selections'][m]!='baseline';counts[m]['improved']+=int(v[0]>r['metrics']['baseline'][0]+1e-8);counts[m]['harmed']+=int(v[0]<r['metrics']['baseline'][0]-1e-8)
 means={m:{k:np.mean(v,axis=0) for k,v in d.items()} for m,d in pairs.items()}
 results={m:{'objects':n,'pairs':len(means[m]),'object':dict(zip(METRICS,(sums[m]/n).tolist())),'frame':dict(zip(METRICS,np.mean(list(means[m].values()),axis=0).tolist())),**counts[m]} for m in METHODS}
 return results,means,seen
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--world-size',required=True,type=int);args=p.parse_args();R=Path(__file__).parent;S=R/'runs'/args.stage
 manifest=json.loads((R/'manifest.json').read_text());spec=manifest['stages'][args.stage];ann=json.loads(Path(spec['annotation']).read_text());expected={(str(v),str(o)) for v,r in ann.items() for o in r['objects']}
 receipts=[]
 for rank in range(args.world_size):
  path=S/'scored'/f'rank{rank}.jsonl';receipt=json.loads((S/'scored'/f'rank{rank}_receipt.json').read_text());assert receipt['state']=='complete' and receipt['output_sha256']==hashlib.sha256(path.read_bytes()).hexdigest();receipts.append(receipt)
 def records():
  for rank in range(args.world_size):
   with (S/'scored'/f'rank{rank}.jsonl').open() as f:
    for line in f:yield json.loads(line)
 results,means,seen=aggregate(records());assert seen==expected and len(seen)==spec['objects']
 take_by_vid={}
 for v,rec in ann.items():
  q=rec['prompt']['first_frame_image'];q=q[0] if isinstance(q,list) else q;take_by_vid[v]=re.search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',q).group()
 takes=sorted(set(take_by_vid.values()));rng=np.random.default_rng(20260915);draws=rng.integers(0,len(takes),size=(10000,len(takes)));comparisons={}
 for m in METHODS[1:]:
  grouped=defaultdict(list)
  for v,vals in means[m].items():grouped[take_by_vid[v]].append(vals[0]-means['baseline'][v][0])
  # Unequal take lengths: preserve frame weighting within each bootstrap sample.
  delta_sum=np.asarray([sum(grouped[t]) for t in takes]);counts=np.asarray([len(grouped[t]) for t in takes]);boot=delta_sum[draws].sum(1)/counts[draws].sum(1)
  comparisons[m]={'delta_frame_iou_pp':float(delta_sum.sum()/counts.sum()*100),'paired_take_bootstrap_95ci_pp':(np.quantile(boot,[.025,.975])*100).tolist()}
 report={'stage':args.stage,'coverage':'exact','pairs':len(ann),'objects':len(seen),'takes':len(takes),'methods':results,'paired_comparison':comparisons,'rank_receipts':receipts,'frozen':manifest['frozen'],
 'annotation_sha256':spec['annotation_sha256'],'scope':'full Exo2Ego benchmark including prior exploratory/confirmation cohorts' if args.stage=='full' else 'runtime smoke test; do not infer scientific gains from it','same_candidate_bank':True,'parameter_fitting':False}
 (S/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({'stage':args.stage,'coverage':'exact','pairs':len(ann),'objects':len(seen),'paired_comparison':comparisons}),flush=True)
 lines=['# Frozen PCCS gain gate — '+args.stage,'',report['scope'],'','| Method | Frame IoU (%) | Delta (pp) | 95% take bootstrap CI (pp) |','|---|---:|---:|---|']
 for m,r in results.items():
  c=comparisons.get(m);lines.append(f"| {m} | {100*r['frame']['IoU']:.4f} | {c['delta_frame_iou_pp'] if c else 0:.4f} | {c['paired_take_bootstrap_95ci_pp'] if c else '-'} |")
 lines+=['','No model or threshold changes. All methods use the same freshly frozen V2-SAM candidate bank. Bootstrap resamples takes while preserving frame weighting for unequal take lengths. The full benchmark is not an entirely blind cohort: it includes earlier exploratory/confirmation test frames. See prior independent holdout for the separate confirmation.']
 (S/'REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
