import argparse,copy,os,sys,time
from common import *
sys.path[:0]=[str(F),str(F/'code/mmengine'),str(F/'code')]
def main():
    import numpy as np,torch
    from mmengine.config import Config
    from mmengine.runner import Runner,set_random_seed
    from mmengine.utils import import_modules_from_strings
    from tools.pccs_eval import load_yaml,resolve_common,resolve_direction,DIRECTION_CONFIGS
    from tools.test import register_function
    import projects.v2sam_pccs.evaluation.pccs_metric as mm
    from dense_context import Capture
    from randomness import stable_pair_seed
    from encoder_contract import fingerprint
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True);ap.add_argument('--rank',type=int,required=True);a=ap.parse_args()
    out=R/'runs'/('rebuild_'+a.phase)/f'rank{a.rank}';out.mkdir(parents=True,exist_ok=True);sp=out/'status.json'
    try:
        plan=read(R/'manifest.json');spec=plan['phases'][a.phase]['shards'][a.rank];frozen=read(F/'manifest.json')
        for n,h in frozen['code_sha256'].items():assert sha(F/n)==h,n
        assert sha(spec['prior'])==spec['prior_sha256'] and sha(spec['annotation'])==spec['annotation_sha256']
        prior={(x['video_id'],x['obj_id']):x for x in map(json.loads,Path(spec['prior']).read_text().splitlines())}
        ann=read(spec['annotation']);os.environ['PCCS_EVAL_SEED']='1'
        raw=load_yaml(Path(spec['config']));resolved=resolve_direction(resolve_common(raw,None,str(out/'runner')),'exo2ego',None);visible=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(resolved['env']);os.environ['CUDA_VISIBLE_DEVICES']=visible;os.environ['PCCS_CONTEXT']='0';os.environ.pop('PCCS_GAIN_BANK',None)
        cfg=Config.fromfile(str(DIRECTION_CONFIGS['exo2ego']));cfg.launcher='none';cfg.work_dir=str(out/'runner');cfg.test_dataloader.num_workers=0;import_modules_from_strings(**cfg.custom_imports);register_function(cfg._cfg_dict);runner=Runner.from_cfg(cfg);model=runner.model;model.eval();capture=Capture(model);mm.save_bank=lambda *args,**kw:None
        contract=fingerprint(model.visual_anchor_expert.sparse_correspondence_backward);assert contract['state_sha256']==frozen['encoder_contract']['state_sha256'];write(out/'encoder_receipt.json',contract)
        path=out/'records.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else [];seen={(r['video_id'],r['obj_id']) for r in rows};assert len(seen)==len(rows) and seen<=set(prior)
        for row in rows:assert sha(row['bank_file'])==row['bank_sha256']
        done={v for v in ann if all((v,str(o)) in seen for o in ann[v]['objects'])};assert all(v in done for v,o in seen),'Incomplete pair requires audited recovery'
        initial=len(done);start=time.monotonic();bank=R/'candidate_bank'/a.phase/f'rank{a.rank}';bank.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:
            for bi,batch in enumerate(runner.test_dataloader):
                vid=batch['data']['video_id'][0]
                if vid in done:continue
                source=batch['data']['raw_prompt_masks'][0];capture.begin(source);seed=stable_pair_seed(vid);set_random_seed(seed,deterministic=False);pred=model.test_step(copy.deepcopy(batch));capture.finish()
                metric=mm.PCCSMetric(collect_device='cpu',enable_vis=False,routing_policy='fusion_first',diagnostic_output=None);metric.process(batch,pred);metrics=[r for group in metric.results for r in group];entries=[]
                for j,m in enumerate(metrics):
                    old=prior[(vid,m['obj_id'])];masks={e:pred[0]['pred_masks_'+e][j].detach().cpu().contiguous().numpy() for e in EXPERTS};hashes={e:hashlib.sha256(v.tobytes()).hexdigest() for e,v in masks.items()};values={e:[float(m[k+'_'+e]) for k in ('iou','dice','shape_acc','location_score')] for e in EXPERTS}
                    ok=hashes==old['mask_sha256'] and m['best_expert']==old['baseline'] and seed==old['candidate_seed'] and all(np.allclose(values[e],old['metrics'][e],atol=1e-7,rtol=0) for e in EXPERTS)
                    if not ok:
                        write(out/'drift.json',{'video_id':vid,'obj_id':m['obj_id'],'old_hash':old['mask_sha256'],'new_hash':hashes,'old_baseline':old['baseline'],'new_baseline':m['best_expert'],'new_metrics':values});raise RuntimeError('Current candidate identity mismatch; no sample exclusion permitted')
                    packed={};arrays={**masks,'query':source[j].detach().cpu().numpy()}
                    for e,v in arrays.items():
                        assert np.isin(v,[0,1]).all();packed[e]=np.packbits(v.astype(np.uint8));packed[e+'_shape']=np.asarray(v.shape)
                    bp=bank/(hashlib.sha256((vid+':'+str(m['obj_id'])).encode()).hexdigest()+'.npz');np.savez_compressed(bp,**packed)
                    entries.append({**old,'bank_file':str(bp),'bank_sha256':sha(bp),'query_sha256':hashlib.sha256(arrays['query'].astype(np.uint8).tobytes()).hexdigest()});seen.add((vid,m['obj_id']))
                f.write(''.join(json.dumps(r,allow_nan=False)+'\n' for r in entries));f.flush();done.add(vid)
                if len(done)==initial+1:write(out/'startup.json',{'candidate_hash_identity':True,'finite_metrics':True,'gpu_peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'candidate_seed':seed,'encoder_sha256':contract['state_sha256']})
                if (bi+1)%16==0 or len(done)==len(ann):status(sp,state='running',done=len(done),total=len(ann),objects=len(seen),eta_seconds=(time.monotonic()-start)/max(1,len(done)-initial)*(len(ann)-len(done)))
        assert seen==set(prior);receipt={'state':'complete','coverage':'exact','pairs':len(done),'objects':len(seen),'all_masks_identical_to_current_full1':True,'records_sha256':sha(path)};write(out/'receipt.json',receipt);status(sp,**receipt,eta_seconds=0)
    except Exception as e:status(sp,state='failed',error=repr(e));raise
if __name__=='__main__':main()
