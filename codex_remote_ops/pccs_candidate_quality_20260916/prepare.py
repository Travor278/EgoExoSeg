from pathlib import Path
import hashlib,json,shutil,time,yaml
from candidate_ops import patch_expert_calls
R=Path(__file__).parent;O=R.parent;S=O/'pccs_corrected_exoego_20260916';G=O/'pccs_gain_20260915/gate_training'
if (R/'code').exists():raise RuntimeError('Experiment already prepared; inspect existing outputs')
assert json.loads((O/'pccs_pipeline_audit_20260916/runtime_probe.json').read_text())['state']=='passed'
source=json.loads((S/'manifest.json').read_text());oldplan=json.loads((G/'plan.json').read_text());allann=json.loads((G/'combined.json').read_text())
shutil.copytree(S/'code',R/'code',symlinks=True);patch_expert_calls(R/'code')
shutil.copy2(S/'aligned_model.py',R/'aligned_model.py')
for n in ('reference','weights'):(R/n).symlink_to(S/n,target_is_directory=True)
cfg=yaml.safe_load((G/'runtime.yaml').read_text());cfg['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');cfg['models']['experts']['exo2ego'].update(visual=source['expert_assets']['vp_exo2ego_full.pth']['path'],fusion=source['expert_assets']['fusion_exo2ego_full.pth']['path'])
keys_screen=oldplan['splits']['fit']['keys'][:128];keys_cal=oldplan['splits']['calibration']['keys'];assert len(keys_screen)==len(keys_cal)==128
phases={'smoke':keys_screen[:8],'screen':keys_screen,'calibration':keys_cal}
manifest={'created_at':time.time(),'arms':['baseline','weighted_pool','reliable_points','both'],'phases':{},'assets':source['expert_assets'],'weights_frozen':True,'expert_training':False,
 'primary_selection':'Require positive oracle gains on BOTH screen and calibration, and positive calibration geometry-consensus gain; among qualifying arms maximize calibration geometry-consensus IoU; ties favor simpler arm. Otherwise stop and retain baseline.',
 'test_policy':'Only after calibration decision, compare fixed selected arm with baseline on Exo2Exo. Do not tune on its outcomes. Earlier Exo2Exo benchmark has been observed, so do not call it wholly blind.',
 'pool':'Predicted coarse-mask-only area-resized probability-weighted dense pooling. Original source-mask pooling unchanged. Execute discarded original sampler to preserve RNG consumption.',
 'points':{'max_points':3,'margin':0.01,'minimum_separation_patches':2,'acceptance':'strict global mutual NN; nonempty source foreground; use multipoint only if at least2 accepted; otherwise original legacy top1'},
 'randomness':'Same hash-derived seed per pair per arm; no arm-dependent seed. Fixed pair order; original sampler RNG advancement preserved.',
 'independent_data_warning':'Screen and calibration are disjoint official-train takes; upstream experts have seen the train split. Exo2Exo transfer confirmation is a separate phase.'}
images=Path(cfg['data']['images'])
for phase,keys in phases.items():
 ann={k:allann[k] for k in keys};ap=R/(phase+'.json');ap.write_text(json.dumps(ann));takes=set()
 for rec in ann.values():
  q=rec['prompt']['first_frame_image'];q=q[0] if isinstance(q,list) else q;takes.add(q.split('/')[0])
  for v in (rec['prompt']['first_frame_image'],rec['video_path']):
   v=v[0] if isinstance(v,list) else v;assert (images/v).is_file(),v
 c=json.loads(json.dumps(cfg));c['data']['annotations']['exo2ego']=str(ap);c['runtime']['ports']['exo2ego']=29981
 cp=R/(phase+'_runtime.yaml');cp.write_text(yaml.safe_dump(c,sort_keys=False));manifest['phases'][phase]={'annotation':str(ap),'config':str(cp),'pairs':len(ann),'objects':sum(len(v['objects']) for v in ann.values()),'takes':sorted(takes),'annotation_sha256':hashlib.sha256(ap.read_bytes()).hexdigest()}
assert set(manifest['phases']['screen']['takes']).isdisjoint(manifest['phases']['calibration']['takes'])
manifest['code_sha256']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ('candidate_ops.py','worker.py','summarize.py','controller.py','test_candidate_ops.py','code/projects/v2sam_pccs/models/pccs_visual_anchor_expert.py','code/projects/v2sam_pccs/models/pccs_fusion_expert.py')}
(R/'manifest.json').write_text(json.dumps(manifest,indent=2));print('PREPARE_OK',flush=True)
