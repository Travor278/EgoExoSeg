from pathlib import Path
import hashlib,json,shutil,time,yaml
R=Path(__file__).parent;A=R.parent/'pccs_pipeline_audit_20260916';S=R.parent/'pccs_exoexo_transfer_20260916';F=R.parent/'pccs_gain_full_20260915'
audit=json.loads((A/'runtime_probe.json').read_text());assert audit['state']=='passed','Do not start before runtime audit passed'
assert json.loads((A/'metric_regression.json').read_text())['state']=='passed'
if (R/'code').exists():raise RuntimeError('Corrected experiment already prepared')
manifest=json.loads((S/'manifest.json').read_text());weights=json.loads((A/'corrected_weights.json').read_text());new={p['name']:p for p in weights['local_candidates'] if p['matches_diary']}
shutil.copytree(S/'code',R/'code',symlinks=True)
for name in ('frozen_gate.joblib','aligned_model.py','gain_features.py'):shutil.copy2(F/name,R/name)
for name in ('weights','reference','python_deps'):(R/name).symlink_to(F/name,target_is_directory=True)
for name in ('full_annotation.json','smoke_annotation.json'):shutil.copy2(S/name,R/name)
plan={'created_at':time.time(),'task_direction':'exo2exo','expert_weights_direction':'exo2ego_corrected','expert_assets':new,'huggingface_revision':weights['huggingface']['revision'],
 'frozen':manifest['frozen'],'stages':{},'primary_method':'native_margin005','secondary_methods':['omama_margin005','geometry_consensus','learned_gate'],
 'method_contract':{'native_margin005':'O-MaMa published head; retain both images aspect ratio, max side1024/floor14; interpolate learned position tables; replace if cosine advantage>0.05',
 'geometry_consensus':'Replace only when canonical margin0.05 and native margin0.05 choose exactly the same candidate; otherwise retain baseline',
 'learned_gate':'Unchanged earlier gate, trained on author-checkpoint outputs; distribution-transfer control, not refitted for corrected experts'},
 'no_exoexo_label_fitting':True,'primary_threshold_prespecified':0.05,'multiple_testing':'One primary comparison; secondary intervals descriptive, no selection of best test threshold',
 'audit_receipt_sha256':hashlib.sha256((A/'runtime_probe.json').read_bytes()).hexdigest()}
for stage,filename,port in [('smokeH100','smoke_annotation.json',29961),('full','full_annotation.json',29962)]:
 cfg=yaml.safe_load((S/'full_runtime.yaml').read_text());cfg['data']['annotations']['exo2ego']=str(R/filename);cfg['models']['experts']['exo2ego'].update(visual=new['vp_exo2ego_full.pth']['path'],fusion=new['fusion_exo2ego_full.pth']['path']);cfg['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');cfg['runtime']['ports']['exo2ego']=port
 cp=R/(stage+'_runtime.yaml');cp.write_text(yaml.safe_dump(cfg,sort_keys=False));a=json.loads((R/filename).read_text());plan['stages'][stage]={'annotation':str(R/filename),'annotation_sha256':hashlib.sha256((R/filename).read_bytes()).hexdigest(),'pairs':len(a),'objects':sum(len(v['objects']) for v in a.values()),'config':str(cp)}
plan['frozen_files']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ('frozen_gate.joblib','aligned_model.py','gain_features.py','score.py','cache.py','summarize.py','run_stage.py','code/projects/v2sam_pccs/candidate_bank.py','code/projects/v2sam_pccs/evaluation/pccs_metric.py')}
(R/'manifest.json').write_text(json.dumps(plan,indent=2));print('CORRECTED_EXOEXO_READY',flush=True)
