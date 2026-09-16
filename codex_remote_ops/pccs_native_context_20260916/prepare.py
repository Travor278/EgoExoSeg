from pathlib import Path
import json,hashlib,shutil,yaml,time
from patch_native import patch
from configs import CONFIGS
R=Path(__file__).parent;Q=R.parent/'pccs_candidate_quality_20260916'
assert not (R/'code').exists(),'Do not overwrite prior experiment'
shutil.copytree(Q/'code',R/'code',symlinks=True);patch(R/'code')
old=json.loads((Q/'manifest.json').read_text());m={'created_at':time.time(),'phases':{},'configs':CONFIGS,'assets':old['assets'],'pretrained_omama_used':False,'new_encoder_passes':0,'candidate_change':False,'parameter_training':False,'selection':'Positive native-PCCS final frame IoU delta on both screen and calibration; maximize calibration; exclude wrong-context and zero controls; otherwise retain baseline.'}
for phase in ('smoke','screen','calibration'):
    ann=json.loads((Q/(phase+'.json')).read_text());keys=list(ann);cfg=yaml.safe_load((Q/(phase+'_runtime.yaml')).read_text());cfg['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');cfg['runtime']['ports']['exo2ego']=29881
    spec={'pairs':len(ann),'objects':sum(len(v['objects']) for v in ann.values()),'shards':[],'reference_root':str(Q),'reference_arm':'baseline'}
    for rank in range(4):
        a={k:ann[k] for k in keys[rank::4]};ap=R/f'{phase}_rank{rank}.json';ap.write_text(json.dumps(a));c=json.loads(json.dumps(cfg));c['data']['annotations']['exo2ego']=str(ap);cp=R/f'{phase}_rank{rank}.yaml';cp.write_text(yaml.safe_dump(c,sort_keys=False));spec['shards'].append({'annotation':str(ap),'config':str(cp),'pairs':len(a),'objects':sum(len(v['objects']) for v in a.values())})
    m['phases'][phase]=spec
m['code_sha256']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ('native_context.py','worker.py','configs.py','patch_native.py','code/projects/v2sam_pccs/evaluation/pccs_metric.py','code/projects/v2sam_pccs/native_context.py')}
(R/'manifest.json').write_text(json.dumps(m,indent=2));print('NATIVE_PREPARED',flush=True)
