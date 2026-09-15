from pathlib import Path
import hashlib,json,shutil,yaml,time
R=Path(__file__).parent;A=R.parent/'pccs_gain_20260915';P=R.parent/'context_pccs_20260914';H=A/'gate_training/holdout'
if (R/'code').exists():raise RuntimeError('Full code already prepared')
hold=json.loads((H/'results.json').read_text());gain=hold['paired_comparison']['learned_gate']
assert hold['coverage']=='exact' and hold['checkpoint_unchanged'] and gain['paired_take_bootstrap_95ci_pp'][0]>0
frozen=json.loads((H/'frozen_manifest.json').read_text());shutil.copy2(H/'frozen_gate.joblib',R/'frozen_gate.joblib')
assert hashlib.sha256((R/'frozen_gate.joblib').read_bytes()).hexdigest()==frozen['checkpoint_sha256']
shutil.copytree(A/'code',R/'code',symlinks=True)
for name in ('aligned_model.py','gain_features.py'):shutil.copy2(A/name,R/name)
for name,target in [('weights',A/'weights'),('reference',A/'reference'),('python_deps',A/'gate_training/python_deps')]:
 (R/name).symlink_to(target,target_is_directory=True)
audit=json.loads((A/'split_audit.json').read_text());source=Path(audit['test']['path']);ann=json.loads(source.read_text())
assert len(ann)==46515 and sum(len(v['objects']) for v in ann.values())==109253
shutil.copy2(source,R/'full_annotation.json')
small=json.loads((H/'annotation.json').read_text());keys=list(small)[:32];(R/'smoke_annotation.json').write_text(json.dumps({k:small[k] for k in keys}))
plan={'created_at':time.time(),'frozen':frozen,'prior_independent_holdout':gain,'stages':{},'scope':'full Exo2Ego benchmark; no further training or threshold selection','full_benchmark_includes_previous_test_cohorts':True}
for stage,ap in [('smoke4090',R/'smoke_annotation.json'),('smoke4',R/'smoke_annotation.json'),('full',R/'full_annotation.json')]:
 conf=yaml.safe_load((A/'confirmation_runtime.yaml').read_text());conf['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');conf['data']['annotations']['exo2ego']=str(ap);conf['runtime']['ports']['exo2ego']={'smoke4090':29931,'smoke4':29932,'full':29933}[stage]
 cp=R/(stage+'_runtime.yaml');cp.write_text(yaml.safe_dump(conf,sort_keys=False));a=json.loads(ap.read_text())
 plan['stages'][stage]={'annotation':str(ap),'annotation_sha256':hashlib.sha256(ap.read_bytes()).hexdigest(),'pairs':len(a),'objects':sum(len(v['objects']) for v in a.values()),'config':str(cp)}
plan['frozen_files']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ('frozen_gate.joblib','aligned_model.py','gain_features.py','score.py','summarize.py','run_stage.py','code/projects/v2sam_pccs/candidate_bank.py','code/projects/v2sam_pccs/evaluation/pccs_metric.py')}
(R/'manifest.json').write_text(json.dumps(plan,indent=2));print('PREPARED',plan['stages']['full'])
