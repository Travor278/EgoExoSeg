from pathlib import Path
import hashlib,json,shutil,yaml,time
R=Path(__file__).parent;G=R/'gate_training';H=G/'holdout'
if H.exists():raise RuntimeError('Holdout output already exists; do not overwrite')
fit=json.loads((G/'fit_results.json').read_text());plan=json.loads((G/'holdout_plan.json').read_text())
assert hashlib.sha256((G/'holdout512.json').read_bytes()).hexdigest()==plan['annotation_sha256']
assert fit['selected']['model']!='baseline','No learned gate selected on calibration'
ckpt=Path(fit['checkpoint']);assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==fit['checkpoint_sha256']
cfg=yaml.safe_load((R/'confirmation_runtime.yaml').read_text());ann=json.loads((G/'holdout512.json').read_text());root=Path(cfg['data']['images'])
for rec in ann.values():
 for p in (rec['prompt']['first_frame_image'],rec['video_path']):
  p=p[0] if isinstance(p,list) else p;assert (root/p).is_file(),p
H.mkdir();shutil.copy2(ckpt,H/'frozen_gate.joblib');shutil.copy2(G/'holdout512.json',H/'annotation.json')
cfg['data']['annotations']['exo2ego']=str(H/'annotation.json');cfg['runtime']['ports']['exo2ego']=29923
(H/'runtime.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
manifest={'frozen_at':time.time(),'dataset_plan':plan,'model':fit['selected']['model'],'threshold':fit['selected']['threshold'],
 'checkpoint_sha256':fit['checkpoint_sha256'],'feature_columns':fit['features'],'feature_code_sha256':hashlib.sha256((R/'gain_features.py').read_bytes()).hexdigest(),
 'fit_report_sha256':hashlib.sha256((G/'fit_results.json').read_bytes()).hexdigest(),'primary':'learned_gate','secondary':'omama_margin005',
 'baseline':'original PCCS on identical frozen candidates','parameter_updates_allowed':False,'official_test_outcomes_used_for_fit':False}
(H/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2));print('HOLDOUT_PREPARED',len(ann),plan['objects'],manifest['model'],manifest['threshold'])
