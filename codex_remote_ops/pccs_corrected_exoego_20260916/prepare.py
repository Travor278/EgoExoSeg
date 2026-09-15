from pathlib import Path
import hashlib,json,shutil,time,yaml
R=Path(__file__).parent;A=R.parent/'pccs_pipeline_audit_20260916';S=R.parent/'pccs_corrected_exoexo_20260916';F=R.parent/'pccs_gain_full_20260915'
assert json.loads((A/'runtime_probe.json').read_text())['state']=='passed'
assert json.loads((A/'metric_regression.json').read_text())['state']=='passed'
if (R/'code').exists():raise RuntimeError('New experiment already prepared')
old=json.loads((S/'manifest.json').read_text());assets=old['expert_assets']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for item in assets.values():assert sha(item['path'])==item['sha256']
shutil.copytree(S/'code',R/'code',symlinks=True)
for name in ('frozen_gate.joblib','aligned_model.py','gain_features.py'):shutil.copy2(S/name,R/name)
for name in ('weights','reference','python_deps'):(R/name).symlink_to(S/name,target_is_directory=True)
shutil.copy2(F/'full_annotation.json',R/'full_annotation.json');ann=json.loads((R/'full_annotation.json').read_text());assert len(ann)==46515 and sum(len(v['objects']) for v in ann.values())==109253
keys=list(ann);selected=[keys[i*(len(keys)-1)//31] for i in range(32)];(R/'smoke_annotation.json').write_text(json.dumps({k:ann[k] for k in selected}))
plan={'created_at':time.time(),'task_direction':'exo2ego','expert_weights_direction':'exo2ego_corrected','expert_assets':assets,'huggingface_revision':old['huggingface_revision'],
 'frozen':old['frozen'],'stages':{},'primary_method':'geometry_consensus','secondary_methods':['omama_margin005','native_margin005','learned_gate'],
 'method_contract':old['method_contract'],'no_test_fitting':True,'primary_thresholds_prespecified':{'cosine_advantage':0.05,'old_gain_gate':0.03},
 'scope':'Full corrected-weight Exo2Ego comparison. Includes previously observed test cohorts; not a new blind benchmark.',
 'report_target':'../pccs_corrected_exoexo_20260916/FINAL_REPORT.md','multiple_testing':'One prespecified primary consensus comparison; secondary intervals descriptive.'}
for stage,filename,port in [('smokeH100','smoke_annotation.json',29971),('full','full_annotation.json',29972)]:
 cfg=yaml.safe_load((F/'full_runtime.yaml').read_text());cfg['data']['annotations']['exo2ego']=str(R/filename);cfg['models']['experts']['exo2ego'].update(visual=assets['vp_exo2ego_full.pth']['path'],fusion=assets['fusion_exo2ego_full.pth']['path']);cfg['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');cfg['runtime']['ports']['exo2ego']=port
 cp=R/(stage+'_runtime.yaml');cp.write_text(yaml.safe_dump(cfg,sort_keys=False));a=json.loads((R/filename).read_text());plan['stages'][stage]={'annotation':str(R/filename),'annotation_sha256':sha(R/filename),'pairs':len(a),'objects':sum(len(v['objects']) for v in a.values()),'config':str(cp)}
plan['frozen_files']={n:sha(R/n) for n in ('frozen_gate.joblib','aligned_model.py','gain_features.py','score.py','cache.py','summarize.py','run_stage.py','code/projects/v2sam_pccs/candidate_bank.py','code/projects/v2sam_pccs/evaluation/pccs_metric.py')}
(R/'manifest.json').write_text(json.dumps(plan,indent=2));print('CORRECTED_EXOEGO_READY',flush=True)
