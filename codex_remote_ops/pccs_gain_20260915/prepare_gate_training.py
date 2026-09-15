"""Prepare a gate-only train/calibration split, using existing training archives."""
from pathlib import Path,PurePosixPath
from collections import defaultdict
import hashlib,json,re,tarfile,time,yaml
R=Path(__file__).parent;G=R/'gate_training';G.mkdir(exist_ok=True)
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def take(v):
 q=v['prompt']['first_frame_image'];q=q[0] if isinstance(q,list) else q
 return re.search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',q).group()
def status(**x):(G/'prepare_status.json').write_text(json.dumps({**x,'updated_at':time.time()},indent=2))
try:
 audit=json.loads((R/'split_audit.json').read_text());source=Path(audit['train']['path']);a=json.loads(source.read_text())
 groups=defaultdict(list)
 for k,v in a.items():groups[take(v)].append(k)
 takes=sorted((t for t,ks in groups.items() if len(ks)>=8),key=lambda t:digest('gate-v1-take:'+t))[:64]
 splits={};selected={};needed=set()
 for split,ts in (('fit',takes[:48]),('calibration',takes[48:])):
  keys=[k for t in ts for k in sorted(groups[t],key=lambda k:digest('gate-v1-pair:'+k))[:8]]
  subset={k:a[k] for k in keys};selected.update(subset)
  (G/(split+'.json')).write_text(json.dumps(subset));splits[split]={'takes':ts,'pairs':len(keys),'keys':keys}
 assert len(selected)==512 and not set(splits['fit']['takes'])&set(splits['calibration']['takes'])
 for rec in selected.values():
  for f in (rec['prompt']['first_frame_image'],rec['video_path']):
   rel=f[0] if isinstance(f,list) else f;p=PurePosixPath(rel)
   assert len(p.parts)==3 and not p.is_absolute() and '..' not in p.parts,rel
   needed.add(rel)
 (G/'combined.json').write_text(json.dumps(selected))
 plan={'purpose':'fit lightweight replacement gate on official TRAIN labels only; calibrate on disjoint TRAIN takes',
       'upstream_experts_saw_training_split':True,'official_validation':False,
       'selection':'fixed hash,48 fitting takes and16 calibration takes;8pairs/take; no target outcome selection',
       'splits':splits,'pairs':512,'objects':sum(len(r['objects']) for r in selected.values()),'image_count':len(needed),
       'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'models':{'ridge':[1.,10.,100.],'hist_gradient_boosting':{'max_iter':100,'max_leaf_nodes':7,'l2_regularization':10.,'min_samples_leaf':25}},
       'thresholds':[0.,.01,.03,.05],'selection_metric':'calibration frame-mean IoU; baseline retained if no positive calibration gain',
       'test_fitting':False,'future_test':'freeze gate after calibration, evaluate on as-yet unused official test takes'}
 (G/'plan.json').write_text(json.dumps(plan,indent=2))
 archives=R.parent/'V2SAM_Ego2Exo_202608/datasets/egoexo_train_mini_complete_v2/data'
 found={};start=time.monotonic();seen=0
 for archive in sorted(archives.glob('*.tar')):
  status(state='extracting_selected_images',archive=archive.name,found=len(found),needed=len(needed))
  with tarfile.open(archive,'r:') as tar:
   for member in tar:
    seen+=1
    rel='/'.join(PurePosixPath(member.name).parts[-3:])
    if rel not in needed or rel in found or not member.isfile():continue
    data=tar.extractfile(member).read();dest=G/'images'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(data);found[rel]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'archive':archive.name,'member':member.name}
  if len(found)==len(needed):break
 missing=sorted(needed-set(found));(G/'image_receipts.json').write_text(json.dumps({'found':found,'missing':missing},indent=2))
 assert not missing,('Training images missing',len(missing),missing[:5])
 conf=yaml.safe_load((R/'confirmation_runtime.yaml').read_text());conf['data']['annotations']['exo2ego']=str(G/'combined.json');conf['data']['images']=str(G/'images');conf['runtime']['ports']['exo2ego']=29919
 (G/'runtime.yaml').write_text(yaml.safe_dump(conf,sort_keys=False))
 status(state='complete',images=len(found),archive_members_seen=seen,elapsed_seconds=time.monotonic()-start)
 print('GATE_DATA_READY',len(selected),len(found),flush=True)
except Exception as e:status(state='failed',error=str(e));raise
