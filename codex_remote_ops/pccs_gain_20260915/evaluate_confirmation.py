"""Pre-registered rules on the same frozen candidate bank; no fitting."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,sys,time
import numpy as np
import torch,yaml
from aligned_model import AlignedMatcher
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914'
sys.path.insert(0,str(P/'ablations_20260915'))
from offline_ablation import Accumulator,metrics

def select(scores,names,nonempty,margin=0.):
    allowed=[i for i,v in enumerate(nonempty) if v]
    if not allowed:return 'baseline'
    if not all(np.isfinite(scores[i]) for i in allowed):raise ValueError('Nonfinite score')
    best=max(allowed,key=lambda i:scores[i])
    if nonempty[0] and scores[best]<=scores[0]+margin:return 'baseline'
    return names[best]

def masks(row):
    p=Path(row['bank_file']);z=np.load(p)
    def unpack(k):return np.unpackbits(z[k])[:int(np.prod(z[k+'_shape']))].reshape(z[k+'_shape']).astype(np.uint8)
    names=[];ms=[];seen=set()
    for k in ('baseline','visual','anchor','fusion'):
        m=unpack(k);digest=hashlib.sha256(m.tobytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);names.append(k);ms.append(m)
    return unpack('query'),names,ms,hashlib.sha256(p.read_bytes()).hexdigest()

def write_status(x):
    p=R/'scoring_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x));tmp.replace(p)

def main():
    plan=json.loads((R/'confirmation_plan.json').read_text());ann=json.loads(Path(plan['annotation']).read_text())
    rows=[json.loads(s) for s in (R/'runs/confirmation/exo2ego/per_object.jsonl').read_text().splitlines()]
    expected={(str(v),str(o)) for v,r in ann.items() for o in r['objects']}
    assert len(rows)==len(expected) and {(r['video_id'],r['obj_id']) for r in rows}==expected
    assert len({r['bank_file'] for r in rows})==len(rows)
    image_root=Path(yaml.safe_load((R/'confirmation_runtime.yaml').read_text())['data']['images'])
    torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AlignedMatcher('cuda');(R/'load_receipts.json').write_text(json.dumps(model.load_receipts,indent=2))
    methods={'learned_object_only_top1':('omama_context_cross_zero',0.),'omama_margin005':('omama_learned',.05),'omama_top1':('omama_learned',0.)}
    acc=defaultdict(Accumulator);out=R/'confirmation_per_object.jsonl'
    if out.exists():raise RuntimeError('Fresh scoring output required')
    start=time.monotonic();gt_max_diff=0.
    from pycocotools import mask as mu
    with out.open('w') as handle:
        for i,r in enumerate(rows):
            rec=ann[r['video_id']];q=rec['prompt']['first_frame_image'];t=rec['video_path']
            q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t
            qm,names,ms,digest=masks(r)
            # Target labels are used ONLY for post-prediction cache/metric integrity.
            gt=mu.decode(rec['objects'][r['obj_id']]['segmentation']).astype(bool)
            # Match ReObjectRelator_Dataset._reshape_mask exactly: nearest 1024x1024.
            gt=torch.nn.functional.interpolate(torch.from_numpy(gt.copy()).float()[None,None],size=(1024,1024),mode='nearest')[0,0].numpy().astype(bool)
            if gt.shape!=ms[0].shape:raise ValueError(('GT/cache geometry mismatch',gt.shape,ms[0].shape))
            for name,m in zip(names,ms):
                e=r['best_expert'] if name=='baseline' else name
                val=float((gt&m.astype(bool)).sum())/(float((gt|m.astype(bool)).sum())+1e-6)
                diff=abs(val-metrics(r,e)[0]);gt_max_diff=max(gt_max_diff,diff)
                assert diff<1e-5,('Cache/metric mismatch',r['video_id'],r['obj_id'],name,diff)
            if qm.any():scores,geometry=model.score(image_root/q,image_root/t,qm,ms,'exo2ego','canonical',ablate=True)
            else:scores={k:[0.]*len(names) for k,_ in methods.values()};geometry={}
            nonempty=[bool(m.any()) for m in ms]
            selections={m:select(scores[k],names,nonempty,margin) if qm.any() else 'baseline' for m,(k,margin) in methods.items()}
            selections={'baseline':'baseline',**selections}
            record={'video_id':r['video_id'],'obj_id':r['obj_id'],'candidate_mask_sha256':digest,'candidate_names':names,'scores':scores,'selections':selections,'metrics':{},'geometry':geometry}
            for method,selected in selections.items():
                expert=r['best_expert'] if selected=='baseline' else selected
                vals=metrics(r,expert);record['metrics'][method]=vals
                acc[method].add(r['video_id'],vals,r['IoU'],expert!=r['best_expert'],'changed' if expert!=r['best_expert'] else 'baseline')
            handle.write(json.dumps(record,allow_nan=False)+'\n');handle.flush()
            if (i+1)%10==0 or i+1==len(rows):
                elapsed=time.monotonic()-start
                state={'state':'running','stage':'omama_scoring','done':i+1,'total':len(rows),'eta_seconds':elapsed/(i+1)*(len(rows)-i-1),'updated_at':time.time()}
                write_status(state);print(json.dumps(state),flush=True)
    # Paired cluster bootstrap: resample takes, preserving all their objects/frames.
    # Pair key paths are normalized in the audited source; support list form too.
    take_by_vid={k:(rec['prompt']['first_frame_image'][0] if isinstance(rec['prompt']['first_frame_image'],list) else rec['prompt']['first_frame_image']).split('/')[0] for k,rec in ann.items()}
    takes=sorted(set(take_by_vid.values()));rng=np.random.default_rng(20260915)
    indices=rng.integers(0,len(takes),size=(10000,len(takes)))
    ci={}
    for m in methods:
        grouped=defaultdict(list)
        for vid,p in acc[m].pairs.items():
            b=acc['baseline'].pairs[vid];grouped[take_by_vid[vid]].append(p[1]/p[0]-b[1]/b[0])
        deltas=np.asarray([np.mean(grouped[t]) for t in takes])
        assert all(len(grouped[t])==8 for t in takes)
        ci[m]={'delta_frame_iou_pp':float(deltas.mean()*100),'paired_take_bootstrap_95ci_pp':(np.quantile(deltas[indices].mean(1),[.025,.975])*100).tolist(),
               'primary_family_bonferroni_97_5ci_pp':(np.quantile(deltas[indices].mean(1),[.0125,.9875])*100).tolist() if m in plan['primary_methods'] else None}
    result={'scope':'512-pair,64-take confirmation; excludes every exploratory probe take; not full-test result','plan':plan,
            'coverage':'exact','cache_gt_iou_max_abs_error':gt_max_diff,'methods':{k:v.result() for k,v in acc.items()},'paired_comparison':ci,
            'per_object_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'no_parameter_fitting':True}
    (R/'confirmation_results.json').write_text(json.dumps(result,indent=2))
    lines=['# O-MaMa gain confirmation','',result['scope'],'','| Method | Frame IoU (%) | Delta (pp) | 95% take bootstrap CI (pp) |','|---|---:|---:|---|']
    for m,v in result['methods'].items():
        c=ci.get(m);lines.append(f"| {m} | {100*v['frame']['IoU']:.4f} | {c['delta_frame_iou_pp'] if c else 0:.4f} | {c['paired_take_bootstrap_95ci_pp'] if c else '-'} |")
    lines+=['','Two primary rules were fixed before this sample was scored. Check multiplicity-adjusted intervals in JSON. Learned object-only means test-time context/cross-attention zeroing of the published head, not a newly trained object-only model. Official validation imagery was unavailable; this is an independent confirmation cohort within the official test split, not validation tuning.']
    (R/'REPORT.md').write_text('\n'.join(lines))
    write_status({'state':'complete','updated_at':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:write_status({'state':'failed','error':str(e),'updated_at':time.time()});raise
