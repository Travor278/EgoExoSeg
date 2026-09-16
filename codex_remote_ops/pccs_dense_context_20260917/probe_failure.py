"""Isolate historical-reference drift before resuming any target evaluation."""
from pathlib import Path
import os,sys,json,copy,hashlib
R=Path(__file__).parent;sys.path[:0]=[str(R/'code/mmengine'),str(R/'code')]
import numpy as np,torch,yaml
from mmengine.config import Config
from mmengine.runner import Runner,set_random_seed
from mmengine.utils import import_modules_from_strings
from tools.pccs_eval import load_yaml,resolve_common,resolve_direction,DIRECTION_CONFIGS
from tools.test import register_function
import projects.v2sam_pccs.evaluation.pccs_metric as mm
from dense_context import Capture
from randomness import stable_pair_seed
a=json.loads((R/'failure_pair.json').read_text());key=next(iter(a));cfgraw=yaml.safe_load((R/'exo2exo_rank2.yaml').read_text());cfgraw['data']['annotations']['exo2ego']=str(R/'failure_pair.json');cp=R/'probe_runtime.yaml';cp.write_text(yaml.safe_dump(cfgraw,sort_keys=False))
raw=load_yaml(cp);resolved=resolve_direction(resolve_common(raw,None,str(R/'probe_runner')),'exo2ego',None);visible=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(resolved['env']);os.environ['CUDA_VISIBLE_DEVICES']=visible;os.environ['PCCS_CONTEXT']='0';os.environ.pop('PCCS_GAIN_BANK',None)
cfg=Config.fromfile(str(DIRECTION_CONFIGS['exo2ego']));cfg.launcher='none';cfg.work_dir=str(R/'probe_runner');cfg.test_dataloader.num_workers=0;import_modules_from_strings(**cfg.custom_imports);register_function(cfg._cfg_dict);runner=Runner.from_cfg(cfg);model=runner.model;model.eval();capture=Capture(model);mm.save_bank=lambda *a,**k:None
batch=next(iter(runner.test_dataloader));assert batch['data']['video_id'][0]==key
prior=next(r for r in map(json.loads,(R.parent/'pccs_candidate_union_20260916/runs/exo2exo/baseline/per_object.jsonl').read_text().splitlines()) if r['video_id']==key)
bank=np.load(prior['bank_file'])
def unpack(e):return np.unpackbits(bank[e])[:int(np.prod(bank[e+'_shape']))].reshape(bank[e+'_shape']).astype(np.uint8)
report={'video_id':key,'numpy':np.__version__,'torch':torch.__version__,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'reference':{'masks':prior['mask_sha256'],'expert':prior['pccs_expert'],'metrics':prior['metrics']['pccs']},'runs':{}}
for name,enabled in [('off1',False),('off2',False),('on1',True),('on2',True),('off_after',False)]:
    capture.enabled=enabled
    if enabled:capture.begin(batch['data']['raw_prompt_masks'][0])
    set_random_seed(stable_pair_seed(key),deterministic=False);pred=model.test_step(copy.deepcopy(batch))
    if enabled:capture.finish()
    m=mm.PCCSMetric(collect_device='cpu',enable_vis=False,routing_policy='fusion_first',diagnostic_output=None);m.process(batch,pred);row=m.results[0][0];masks={e:pred[0]['pred_masks_'+e][0].cpu().numpy() for e in ('visual','anchor','fusion')}
    entry={'masks':{e:hashlib.sha256(x.tobytes()).hexdigest() for e,x in masks.items()},'pixel_differences_from_reference':{e:int(np.count_nonzero(x!=unpack(e))) for e,x in masks.items()},'expert':row['best_expert'],'metrics':[float(row[k]) for k in ('IoU','Dice','shape_acc','location_score')],'backward_points':{e:pred[0]['points_'+e+'_in_query'][0].tolist() for e in masks}}
    report['runs'][name]=entry;(R/'failure_probe.json').write_text(json.dumps(report,indent=2));print('PROBE',name,json.dumps(entry),flush=True)
report['all_reference_masks_exact']=all(x['masks']==prior['mask_sha256'] for x in report['runs'].values());report['all_reference_routes_exact']=all(x['expert']==prior['pccs_expert'] for x in report['runs'].values());report['all_reference_metrics_exact']=all(np.allclose(x['metrics'],prior['metrics']['pccs'],atol=1e-7,rtol=0) for x in report['runs'].values());(R/'failure_probe.json').write_text(json.dumps(report,indent=2))
assert report['all_reference_masks_exact'] and report['all_reference_routes_exact'] and report['all_reference_metrics_exact'],'Probe found reproducible historical drift; do not resume blindly'
