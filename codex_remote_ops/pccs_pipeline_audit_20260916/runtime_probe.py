"""Bounded real-model audit: initialization, checkpoint content, GT invariance and routing."""
from pathlib import Path
import copy,gc,hashlib,inspect,json,os,random,subprocess,sys,time
R=Path(__file__).parent;F=R.parent/'pccs_gain_full_20260915';P=R.parent/'context_pccs_20260914'
sys.path[:0]=[str(F/'code/mmengine'),str(F/'code')]
def save(obj):(R/'runtime_probe.json').write_text(json.dumps(obj,indent=2,allow_nan=False))
out={'state':'running','started_at':time.time(),'cases':{}}
try:
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 if gpu.strip():save({'state':'blocked_gpu_busy','gpu_processes':gpu});raise SystemExit(3)
 import numpy as np,torch,yaml
 from mmengine.config import Config
 from mmengine.runner import Runner,set_random_seed
 from mmengine.utils import import_modules_from_strings
 from tools.pccs_eval import load_yaml,resolve_common,resolve_direction,DIRECTION_CONFIGS
 from tools.test import register_function
 from xtuner.model.utils import guess_load_checkpoint
 from projects.v2sam_pccs.evaluation.pccs_metric import PCCSMetric
 corrected=json.loads((R/'corrected_weights.json').read_text())
 new={p['name']:p['path'] for p in corrected['local_candidates'] if p['matches_diary']}
 assert set(new)=={'vp_exo2ego_full.pth','fusion_exo2ego_full.pth'}
 config=load_yaml(F/'full_runtime.yaml');fullanno=json.loads(Path(config['data']['annotations']['exo2ego']).read_text())
 keys=[next(k for k,v in fullanno.items() if len(v['objects'])>=2),next(k for k,v in fullanno.items() if len(v['objects'])==1)]
 ap=R/'audit_two_pairs.json';ap.write_text(json.dumps({k:fullanno[k] for k in keys}))
 def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 def tensor_map(value,prefix=''):
  if torch.is_tensor(value):return {prefix:digest(value)}
  if isinstance(value,np.ndarray):return {prefix:hashlib.sha256(value.tobytes()).hexdigest()}
  if isinstance(value,dict):return {k:v for key,item in value.items() for k,v in tensor_map(item,prefix+'/'+str(key)).items()}
  if isinstance(value,(list,tuple)):return {k:v for i,item in enumerate(value) for k,v in tensor_map(item,prefix+'/'+str(i)).items()}
  return {}
 for weightset in ('official_used','corrected_selected'):
  cfgdata=copy.deepcopy(config);cfgdata['data']['annotations']['exo2ego']=str(ap)
  if weightset=='corrected_selected':cfgdata['models']['experts']['exo2ego'].update(visual=new['vp_exo2ego_full.pth'],fusion=new['fusion_exo2ego_full.pth'])
  common=resolve_common(cfgdata,'0',str(R/weightset/'runner'));resolved=resolve_direction(common,'exo2ego',None);os.environ.update(resolved['env']);os.environ['PCCS_CONTEXT']='0';os.environ['PCCS_GAIN_BANK']=str(R/weightset/'bank')
  cfg=Config.fromfile(str(DIRECTION_CONFIGS['exo2ego']));cfg.launcher='none';cfg.work_dir=str(R/weightset/'runner');cfg.test_dataloader.num_workers=0
  import_modules_from_strings(**cfg.custom_imports);register_function(cfg._cfg_dict)
  runner=Runner.from_cfg(cfg);model=runner.model;model.eval()
  case={'config':cfgdata,'initialization':{},'checkpoints':{},'gt_counterfactual':[]};out['cases'][weightset]=case;save(out)
  branches={'forward':model.forward_correspondence.dinov3_model,'visual_backward':model.visual_anchor_expert.sparse_correspondence_backward.dinov3_model,'fusion_backward':model.fusion_expert.sparse_correspondence_backward.dinov3_model}
  before={n:{k:digest(t) for k,t in m.state_dict().items()} for n,m in branches.items()}
  # Exercise the MMEngine top-level init hook implicated in the historical bug.
  model.init_weights()
  after={n:{k:digest(t) for k,t in m.state_dict().items()} for n,m in branches.items()}
  assert before==after,'DINO changed at init_weights'
  asset=guess_load_checkpoint(config['models']['dinov3']['weights'])
  for name,m in branches.items():
   st=m.state_dict();mismatch=[k for k,t in st.items() if k not in asset or not torch.equal(t.detach().cpu(),asset[k].to(t.dtype).cpu())]
   case['initialization'][name]={'tensors':len(st),'unchanged_after_init':True,'base_asset_mismatch':mismatch,'all_tensor_sha256':after[name]}
   assert not mismatch,(weightset,name,mismatch[:3])
  del asset
  for name,module in [('visual',model.visual_anchor_expert),('fusion',model.fusion_expert)]:
   ck=guess_load_checkpoint(cfgdata['models']['experts']['exo2ego'][name]);state=module.state_dict();missing=[];different=[];matched=0;extra=[]
   for k,v in ck.items():
    key='sparse_correspondence_backward.'+k.split('.',1)[1] if name=='fusion' and k.startswith('sparse_correspondence.') else k
    if key not in state:extra.append(key);continue
    matched+=1
    if not torch.equal(state[key].detach().cpu(),v.to(state[key].dtype).cpu()):different.append(key)
   assert not different and all(k=='contrast_schedule.step' for k in extra),(name,different,extra)
   case['checkpoints'][name]={'matched':matched,'content_mismatch':different,'ignored_training_only_keys':extra};del ck
  # Explicit current-frame count contract, not last-object or positive-IoU filtering.
  batches=list(runner.test_dataloader);assert len(batches)==2
  for bi,batch in enumerate(batches):
   outputs=[];routes=[];hashes=[]
   for variant in ('original','zeros','ones','random'):
    b=copy.deepcopy(batch)
    if variant!='original':
     rs=torch.Generator().manual_seed(773)
     b['data']['masks']=[torch.zeros_like(m) if variant=='zeros' else torch.ones_like(m) if variant=='ones' else torch.randint(0,2,m.shape,generator=rs,dtype=torch.int64).to(m.dtype) for m in b['data']['masks']]
    set_random_seed(20260916,deterministic=False)
    predictions=model.test_step(copy.deepcopy(b));hashes.append(tensor_map(predictions))
    assert hashes[-1]==hashes[0],(weightset,bi,variant,'GT changed predictions')
    os.environ['PCCS_GAIN_BANK']=str(R/weightset/f'gt_{bi}_{variant}')
    metric=PCCSMetric(collect_device='cpu',enable_vis=False,routing_policy='fusion_first',diagnostic_output=None)
    metric.process(b,predictions);selected=[r['best_expert'] for pair in metric.results for r in pair];routes.append(selected)
    assert routes[-1]==routes[0],(weightset,bi,variant,'GT changed selection')
   case['gt_counterfactual'].append({'pair_index':bi,'objects':len(routes[0]),'variants':['original','zeros','ones','random'],'prediction_tensor_hashes_equal':True,'routes_equal':True,'selected':routes[0],'tensors_compared':len(hashes[0])});save(out)
  case['state']='passed';save(out);del branches,model,runner,batches;gc.collect();torch.cuda.empty_cache()
 out['state']='passed';out['completed_at']=time.time();save(out);print('RUNTIME_PROBE_PASS',flush=True)
except Exception as e:out.update(state='failed',error=repr(e));save(out);raise
