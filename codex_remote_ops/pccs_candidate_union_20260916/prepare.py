from pathlib import Path
import hashlib,json,shutil,yaml,time
R=Path(__file__).parent;P=R.parent/'pccs_candidate_quality_20260916';F=R.parent/'pccs_corrected_exoego_20260916'
assert json.loads((P/'status.json').read_text())['state']=='complete'
if (R/'code').exists():raise RuntimeError('Existing experiment; do not overwrite')
shutil.copytree(P/'code',R/'code',symlinks=True)
for n in ('aligned_model.py',):shutil.copy2(P/n,R/n)
for n in ('weights','reference'):(R/n).symlink_to(P/n,target_is_directory=True)
(R/'python_deps').symlink_to(F/'python_deps',target_is_directory=True)
old=json.loads((P/'manifest.json').read_text());manifest={'created_at':time.time(),'arms':['baseline','reliable_points','residual005','residual010','residual025'],'phases':{},'source_bank':str(P),'assets':old['assets'],
 'immutable_original_candidates':True,'residual_alphas':[.05,.10,.25],'screen_calibration_reused':True,
 'stages':['cached_multipoint_union','residual_pool_union','quality_gain_ranker','low_rank_OMaMa_embedding_adapter'],
 'adapter_scope':'Train a small residual metric projection on frozen published O-MaMa embeddings; do not finetune V2SAM prompt projection in this run.',
 'selection_rule':'Retain original baseline unless a candidate method improves fixed-consensus final frame IoU on BOTH screen and calibration. Select maximum calibration IoU among qualified methods, tie favor simpler. Freeze winner before Exo2Exo; no target fitting.',
 'scientific_scope':'TRAIN screen/cal reused in previous explorations; not an unseen validation set. Exo2Exo benchmark also observed previously. Report conditional transfer confirmation, not blind discovery.'}
for phase in ('smoke','screen','calibration'):
    shutil.copy2(P/(phase+'.json'),R/(phase+'.json'));cfg=yaml.safe_load((P/(phase+'_runtime.yaml')).read_text());cfg['data']['annotations']['exo2ego']=str(R/(phase+'.json'));cfg['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');cfg['runtime']['ports']['exo2ego']=29991
    cp=R/(phase+'_runtime.yaml');cp.write_text(yaml.safe_dump(cfg,sort_keys=False));spec=dict(old['phases'][phase]);spec.update(annotation=str(R/(phase+'.json')),config=str(cp));manifest['phases'][phase]=spec
manifest['code_sha256']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ('worker.py','candidate_ops.py','residual_pool.py','union_core.py','build_dataset.py','fit_models.py','metric_adapter.py','evaluate_selected.py','controller.py')}
(R/'manifest.json').write_text(json.dumps(manifest,indent=2));print('UNION_PREPARED',flush=True)
