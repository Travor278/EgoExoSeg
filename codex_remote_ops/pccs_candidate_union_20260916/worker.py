from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time
R=Path(__file__).parent
sys.path[:0]=[str(R/'code/mmengine'),str(R/'code')]
def main():
 import numpy as np,torch,yaml
 from mmengine.config import Config
 from mmengine.runner import Runner,set_random_seed
 from mmengine.utils import import_modules_from_strings
 from tools.pccs_eval import load_yaml,resolve_common,resolve_direction,DIRECTION_CONFIGS
 from tools.test import register_function
 from projects.v2sam_pccs.evaluation.pccs_metric import PCCSMetric
 from candidate_ops import stable_pair_seed,attach_reliable_points
 from residual_pool import install_residual_pool
 from aligned_model import AlignedMatcher
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--phase',required=True);args=p.parse_args()
 plan=json.loads((R/'manifest.json').read_text());assert args.arm in plan['arms'];spec=plan['phases'][args.phase];outdir=R/'runs'/args.phase/args.arm
 if outdir.exists():raise RuntimeError('Fresh arm output required')
 outdir.mkdir(parents=True)
 def status(**x):
  temp=outdir/'status.tmp';temp.write_text(json.dumps({**x,'updated_at':time.time(),'arm':args.arm,'phase':args.phase}));temp.replace(outdir/'status.json')
 try:
  for name,h in plan['code_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
  config=load_yaml(Path(spec['config']));common=resolve_common(config,None,str(outdir/'runner'));resolved=resolve_direction(common,'exo2ego',None)
  visible=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(resolved['env']);os.environ['CUDA_VISIBLE_DEVICES']=visible;os.environ['PCCS_CONTEXT']='0';os.environ['PCCS_GAIN_BANK']=str(outdir/'bank')
  cfg=Config.fromfile(str(DIRECTION_CONFIGS['exo2ego']));cfg.launcher='none';cfg.work_dir=str(outdir/'runner');cfg.test_dataloader.num_workers=0
  import_modules_from_strings(**cfg.custom_imports);register_function(cfg._cfg_dict);runner=Runner.from_cfg(cfg);model=runner.model;model.eval()
  if args.arm.startswith('residual'):install_residual_pool(model,{'residual005':.05,'residual010':.10,'residual025':.25}[args.arm])
  if args.arm=='reliable_points':attach_reliable_points(model)
  matcher=AlignedMatcher('cuda');ann=json.loads(Path(spec['annotation']).read_text());image_root=Path(config['data']['images']);expected={(str(v),str(o)) for v,rec in ann.items() for o in rec['objects']};seen=set();t0=time.monotonic();output=outdir/'per_object.jsonl'
  def digest(m):return hashlib.sha256(m.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
  with output.open('w') as handle:
   for bi,batch in enumerate(runner.test_dataloader):
    vid=batch['data']['video_id'][0];set_random_seed(stable_pair_seed(vid),deterministic=False)
    prediction=model.test_step(copy.deepcopy(batch));pred=prediction[0]
    metric=PCCSMetric(collect_device='cpu',enable_vis=False,routing_policy='fusion_first',diagnostic_output=None);metric.process(batch,prediction);rows=[r for pair in metric.results for r in pair]
    assert len(rows)==len(ann[vid]['objects'])
    rec=ann[vid];q=rec['prompt']['first_frame_image'];t=rec['video_path'];q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t
    pointdiag=getattr(model.forward_correspondence,'_candidate_quality_diagnostics',None)
    for oi,row in enumerate(rows):
     ident=(row['video_id'],row['obj_id']);assert ident in expected and ident not in seen;seen.add(ident)
     experts=('visual','anchor','fusion');masks={e:pred['pred_masks_'+e][oi].detach().cpu().numpy().astype(np.uint8) for e in experts};base=row['best_expert'];assert base in experts
     names=['baseline'];ms=[masks[base]];digests={e:digest(pred['pred_masks_'+e][oi]) for e in experts};used={hashlib.sha256(ms[0].tobytes()).hexdigest()}
     for e in experts:
      h=hashlib.sha256(masks[e].tobytes()).hexdigest()
      if h not in used:used.add(h);names.append(e);ms.append(masks[e])
     qm=batch['data']['raw_prompt_masks'][0][oi].cpu().numpy().astype(np.uint8);selections=[];score_sets={}
     for mode in ('canonical','native_interp'):
      scores,_=matcher.score(image_root/q,image_root/t,qm,ms,'exo2ego',mode) if qm.any() else ({'omama_learned':[0.]*len(ms)},None)
      s=scores['omama_learned'];score_sets[mode]=s;allowed=[i for i,m in enumerate(ms) if m.any()]
      best=max(allowed,key=lambda i:s[i]) if allowed else 0
      if not qm.any() or (ms[0].any() and s[best]<=s[0]+.05):best=0
      selections.append(names[best])
     chosen=selections[0] if selections[0]==selections[1] else 'baseline';expert=base if chosen=='baseline' else chosen
     def vals(e):return [float(row[k]) for k in ('IoU','Dice','shape_acc','location_score')] if e==base else [float(row[k+'_'+e]) for k in ('iou','dice','shape_acc','location_score')]
     oracle=max(experts,key=lambda e:row['iou_'+e]);metrics={'pccs':vals(base),'consensus':vals(expert),'oracle':vals(oracle),**{e:vals(e) for e in experts}}
     assert all(np.isfinite(v).all() for v in metrics.values())
     item={'video_id':vid,'obj_id':row['obj_id'],'arm':args.arm,'phase':args.phase,'seed':stable_pair_seed(vid),'mask_sha256':digests,'bank_file':row['bank_file'],'metrics':metrics,'pccs_expert':base,'consensus_expert':expert,'scores':score_sets,'candidate_names':names,'point_count':len(pred['forward_points_target'][oi]),'point_diagnostics':pointdiag[oi] if pointdiag else None}
     handle.write(json.dumps(item,allow_nan=False)+'\n');handle.flush()
    if (bi+1)%8==0 or bi+1==spec['pairs']:
     elapsed=time.monotonic()-t0;status(state='running',pairs_done=bi+1,pairs_total=spec['pairs'],objects_done=len(seen),eta_seconds=elapsed/(bi+1)*(spec['pairs']-bi-1))
  assert seen==expected;receipt={'state':'complete','pairs':spec['pairs'],'objects':len(seen),'coverage':'exact','records_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'arm':args.arm,'phase':args.phase}
  (outdir/'receipt.json').write_text(json.dumps(receipt,indent=2));status(**receipt,eta_seconds=0);print('ARM_COMPLETE',args.phase,args.arm,flush=True)
 except Exception as e:status(state='failed',error=repr(e));raise
if __name__=='__main__':main()
