from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time
R=Path(__file__).parent
sys.path[:0]=[str(R/'code/mmengine'),str(R/'code')]
def main():
    import numpy as np,torch
    from mmengine.config import Config
    from mmengine.runner import Runner,set_random_seed
    from mmengine.utils import import_modules_from_strings
    from tools.pccs_eval import load_yaml,resolve_common,resolve_direction,DIRECTION_CONFIGS
    from tools.test import register_function
    import projects.v2sam_pccs.evaluation.pccs_metric as metric_module
    from native_context import ContextCapture,recheck_fusion
    from configs import CONFIGS
    from randomness import stable_pair_seed
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--rank',type=int,required=True);args=p.parse_args()
    plan=json.loads((R/'manifest.json').read_text());phase=plan['phases'][args.phase];spec=phase['shards'][args.rank];out=R/'runs'/args.phase/f'rank{args.rank}';out.mkdir(parents=True,exist_ok=False)
    def status(**x):
        p=out/'status.tmp';p.write_text(json.dumps({**x,'updated_at':time.time()}));p.replace(out/'status.json')
    def signature(x):
        if isinstance(x,torch.Tensor):return str(x.dtype)+str(tuple(x.shape))+hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        if isinstance(x,np.ndarray):return str(x.dtype)+str(x.shape)+hashlib.sha256(x.tobytes()).hexdigest()
        if isinstance(x,dict):return {k:signature(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [signature(v) for v in x]
        return x
    try:
        for n,h in plan['code_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
        cfgraw=load_yaml(Path(spec['config']));resolved=resolve_direction(resolve_common(cfgraw,None,str(out/'runner')),'exo2ego',None);visible=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(resolved['env']);os.environ['CUDA_VISIBLE_DEVICES']=visible;os.environ['PCCS_CONTEXT']='0';os.environ.pop('PCCS_GAIN_BANK',None)
        cfg=Config.fromfile(str(DIRECTION_CONFIGS['exo2ego']));cfg.launcher='none';cfg.work_dir=str(out/'runner');cfg.test_dataloader.num_workers=0;import_modules_from_strings(**cfg.custom_imports);register_function(cfg._cfg_dict);runner=Runner.from_cfg(cfg);model=runner.model;model.eval();capture=ContextCapture(model)
        # Disable only optional artifact I/O, never the metric or routing computation.
        metric_module.save_bank=lambda *a,**k:None
        reference=Path(phase['reference_root'])/'runs'/args.phase/phase['reference_arm']/'per_object.jsonl';rec=json.loads((reference.parent/'receipt.json').read_text());assert rec['records_sha256']==hashlib.sha256(reference.read_bytes()).hexdigest();old={(r['video_id'],r['obj_id']):r for r in map(json.loads,reference.read_text().splitlines())}
        ann=json.loads(Path(spec['annotation']).read_text());expected={(str(v),str(o)) for v,a in ann.items() for o in a['objects']};seen=set();start=time.monotonic();off_checks=0;path=out/'records.jsonl'
        with path.open('w') as f:
            for bi,batch in enumerate(runner.test_dataloader):
                vid=batch['data']['video_id'][0];seed=stable_pair_seed(vid)
                if args.phase=='smoke':
                    capture.enabled=False;set_random_seed(seed,deterministic=False);original=model.test_step(copy.deepcopy(batch));capture.enabled=True
                capture.begin(batch['data']['raw_prompt_masks'][0]);set_random_seed(seed,deterministic=False);prediction=model.test_step(copy.deepcopy(batch));context=capture.finish()
                if args.phase=='smoke':assert signature(original)==signature(prediction),'Context capture changed a prediction';off_checks+=1
                results={}
                for name,setting in CONFIGS.items():
                    prediction[0]['native_context']=context['wrong' if name=='wrong_context' else 'normal'];metric=metric_module.PCCSMetric(collect_device='cpu',enable_vis=False,routing_policy='fusion_first',diagnostic_output=None);metric.native_context_cfg=setting;metric.process(batch,prediction);results[name]=[r for pair in metric.results for r in pair]
                for j,b in enumerate(results['baseline']):
                    key=(b['video_id'],b['obj_id']);assert key in expected and key not in seen;seen.add(key);prior=old[key]
                    masks={e:hashlib.sha256(prediction[0]['pred_masks_'+e][j].detach().cpu().contiguous().numpy().tobytes()).hexdigest() for e in ('visual','anchor','fusion')};assert masks==prior['mask_sha256'],('Original candidate changed',key)
                    assert b['best_expert']==prior['pccs_expert'];vals=lambda r:[float(r[k]) for k in ('IoU','Dice','shape_acc','location_score')]
                    assert np.allclose(vals(b),prior['metrics']['pccs'],atol=1e-7,rtol=0);assert results['zero'][j]['best_expert']==b['best_expert'] and vals(results['zero'][j])==vals(b)
                    q=ann[vid]['prompt']['first_frame_image'];q=q[0] if isinstance(q,list) else q
                    row={'video_id':vid,'obj_id':b['obj_id'],'take_id':q.split('/')[0],'seed':seed,'mask_sha256':masks,'context':context['normal'][j],'wrong_context':context['wrong'][j],'original_fusion_accepted':b['selection_reason']=='fusion_quality_pass','fusion_context_recheck':bool(recheck_fusion(context['normal'][j],CONFIGS['both'])),'methods':{name:{'expert':rs[j]['best_expert'],'metrics':vals(rs[j]),'reason':rs[j]['selection_reason']} for name,rs in results.items()}}
                    f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                if (bi+1)%4==0 or bi+1==spec['pairs']:status(state='running',done=bi+1,total=spec['pairs'],objects=len(seen),eta_seconds=(time.monotonic()-start)/(bi+1)*(spec['pairs']-bi-1))
        assert seen==expected;receipt={'state':'complete','coverage':'exact','pairs':spec['pairs'],'objects':len(seen),'capture_on_off_pairs':off_checks,'lambda_zero_parity':True,'all_prior_candidates_exact':True,'records_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};(out/'receipt.json').write_text(json.dumps(receipt,indent=2));status(**receipt,eta_seconds=0);print('NATIVE_WORKER_COMPLETE',args.phase,args.rank,flush=True)
    except Exception as e:status(state='failed',error=repr(e));raise
if __name__=='__main__':main()
