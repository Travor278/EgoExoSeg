"""Remote preparation: task=Exo2Exo; all learned weights remain Exo2Ego."""
from pathlib import Path
import hashlib,json,re,shutil,time,yaml
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914';S=R.parent/'pccs_gain_full_20260915'
if (R/'code').exists():raise RuntimeError('Transfer experiment already prepared')
source_manifest=json.loads((S/'manifest.json').read_text());frozen=source_manifest['frozen']
old=json.loads((P/'experiment_manifest.json').read_text())['datasets']['exo2exo'];anno=Path(old['annotation']);a=json.loads(anno.read_text())
assert len(a)==1094 and sum(len(v['objects']) for v in a.values())==1094
assert hashlib.sha256(anno.read_bytes()).hexdigest()==old['sha256']
conf=yaml.safe_load((P/'runtime_exo2exo.yaml').read_text());images=Path(conf['data']['images'])
# Geometry and take grouping must be established from these actual Exo2Exo records.
takes=set();examples=[]
for k,rec in a.items():
 q=rec['prompt']['first_frame_image'];t=rec['video_path'];q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t
 assert (images/q).is_file() and (images/t).is_file(),(k,q,t)
 match=re.search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',q)
 assert match,('Cannot audit take grouping',q);takes.add(match.group())
 if len(examples)<3:examples.append({'key':k,'query':q,'target':t})
shutil.copytree(S/'code',R/'code',symlinks=True)
for name in ('frozen_gate.joblib','aligned_model.py','gain_features.py'):shutil.copy2(S/name,R/name)
assert hashlib.sha256((R/'frozen_gate.joblib').read_bytes()).hexdigest()==frozen['checkpoint_sha256']
for name in ('weights','reference','python_deps'):(R/name).symlink_to(S/name,target_is_directory=True)
shutil.copy2(anno,R/'full_annotation.json');keys=sorted(a,key=lambda k:hashlib.sha256(('exoexo-runtime-smoke:'+k).encode()).hexdigest())[:32]
(R/'smoke_annotation.json').write_text(json.dumps({k:a[k] for k in keys}))
plan={'created_at':time.time(),'task_direction':'exo2exo','expert_weights_direction':'exo2ego','matcher_weights_direction':'exo2ego','gate_training_direction':'exo2ego',
 'frozen':frozen,'stages':{},'scope':'Exo2Exo transfer with frozen Exo2Ego experts/matcher/gate; no target-label tuning',
 'matcher_geometry':'canonical: source532x952,target700x700,100px per-side full-context box; source geometry transferred without adaptation',
 'takes':len(takes),'examples':examples,'prior_observation':'Earlier ring-context full evaluation and O-MaMa64-pair exploration exist; no fitting on these labels',
 'comparisons':['originalPCCS','frozen learned gate threshold0.03','frozen full O-MaMa cosine margin0.05'],
 'priority':'Primary downstream objective is Exo2Exo gains; Exo2Ego gains are supporting evidence only'}
for stage,path,port in [('smoke4090',R/'smoke_annotation.json',29941),('full',R/'full_annotation.json',29942)]:
 cfg=yaml.safe_load((P/'runtime_exo2exo.yaml').read_text());cfg['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');cfg['data']['annotations']['exo2ego']=str(path);cfg['runtime']['ports']['exo2ego']=port
 # exo2ego is the upstream config/weight key, not the task-direction label.
 cp=R/(stage+'_runtime.yaml');cp.write_text(yaml.safe_dump(cfg,sort_keys=False));ann=json.loads(path.read_text())
 plan['stages'][stage]={'annotation':str(path),'annotation_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pairs':len(ann),'objects':sum(len(v['objects']) for v in ann.values()),'config':str(cp)}
plan['frozen_files']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ('frozen_gate.joblib','aligned_model.py','gain_features.py','score.py','cache.py','summarize.py','run_stage.py','code/projects/v2sam_pccs/candidate_bank.py','code/projects/v2sam_pccs/evaluation/pccs_metric.py')}
(R/'manifest.json').write_text(json.dumps(plan,indent=2));print('EXO2EXO_PREPARED',len(a),len(takes),flush=True)
