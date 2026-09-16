from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time
R=Path(__file__).parent;sys.path[:0]=[str(R/'code/mmengine'),str(R/'code')]
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
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--rank',type=int,required=True);args=p.parse_args();plan=json.loads((R/'manifest.json').read_text());phase=plan['phases'][args.phase];spec=phase['shards'][args.rank];out=R/'runs'/args.phase/f'rank{args.rank}';out.mkdir(parents=True,exist_ok=args.phase=='exo2exo')
    def status(**x):
        t=out/'status.tmp';t.write_text(json.dumps({**x,'updated_at':time.time()}));t.replace(out/'status.json')
    def signature(x):
        if isinstance(x,torch.Tensor):return str(x.dtype)+str(tuple(x.shape))+hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        if isinstance(x,np.ndarray):return str(x.dtype)+str(x.shape)+hashlib.sha256(x.tobytes()).hexdigest()
        if isinstance(x,dict):return {k:signature(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [signature(v) for v in x]
        return x
    try:
        for n,h in plan['code_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
        raw=load_yaml(Path(spec['config']));resolved=resolve_direction(resolve_common(raw,None,str(out/'runner')),'exo2ego',None);visible=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(resolved['env']);os.environ['CUDA_VISIBLE_DEVICES']=visible;os.environ['PCCS_CONTEXT']='0';os.environ.pop('PCCS_GAIN_BANK',None)
        cfg=Config.fromfile(str(DIRECTION_CONFIGS['exo2ego']));cfg.launcher='none';cfg.work_dir=str(out/'runner');cfg.test_dataloader.num_workers=0;import_modules_from_strings(**cfg.custom_imports);register_function(cfg._cfg_dict);runner=Runner.from_cfg(cfg);model=runner.model;model.eval();capture=Capture(model);mm.save_bank=lambda *a,**k:None
        ref=Path(phase['reference_root'])/'runs'/phase['reference_phase']/'baseline/per_object.jsonl';receipt=json.loads((ref.parent/'receipt.json').read_text());assert receipt['records_sha256']==hashlib.sha256(ref.read_bytes()).hexdigest();prior={(r['video_id'],r['obj_id']):r for r in map(json.loads,ref.read_text().splitlines())}
        ann=json.loads(Path(spec['annotation']).read_text());expected={(str(v),str(o)) for v,a in ann.items() for o in a['objects']};seen=set();checked=off=drift_count=0;start=time.monotonic();path=out/'records.jsonl'
        previous=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        for row in previous:seen.add((row['video_id'],row['obj_id']));checked+=not row.get('historical_drift',False);drift_count+=row.get('historical_drift',False)
        assert len(seen)==len(previous) and seen<=expected
        if (out/'receipt.json').exists():
            receipt=json.loads((out/'receipt.json').read_text());assert receipt['records_sha256']==hashlib.sha256(path.read_bytes()).hexdigest() and seen==expected;return
        completed_videos={v for v in ann if all((v,str(o)) in seen for o in ann[v]['objects'])};initial_pairs=len(completed_videos);bankroot=R/'candidate_bank';bankroot.mkdir(exist_ok=True)
        with path.open('a' if previous else 'w') as f:
            for bi,batch in enumerate(runner.test_dataloader):
                vid=batch['data']['video_id'][0];seed=stable_pair_seed(vid)
                if vid in completed_videos:continue
                if args.phase=='smoke':capture.enabled=False;set_random_seed(seed,deterministic=False);original=model.test_step(copy.deepcopy(batch));capture.enabled=True
                source=batch['data']['raw_prompt_masks'][0];capture.begin(source);set_random_seed(seed,deterministic=False);pred=model.test_step(copy.deepcopy(batch));dense=capture.finish()
                if args.phase=='smoke':assert signature(original)==signature(pred),'Capture changes original predictions';off+=1
                metric=mm.PCCSMetric(collect_device='cpu',enable_vis=False,routing_policy='fusion_first',diagnostic_output=None);metric.process(batch,pred);rows=[r for pair in metric.results for r in pair]
                batch_records=[]
                for j,b in enumerate(rows):
                    key=(b['video_id'],b['obj_id']);assert key in expected and key not in seen;seen.add(key);evidence=dense[j];masks={};metrics={};src=source[j];pts_orig=metric._extract_representative_points_from_mask(src,num_points=1,strategy='center')
                    for e in ('visual','anchor','fusion'):
                        pm=pred[0]['pred_masks_'+e][j];masks[e]=hashlib.sha256(pm.detach().cpu().contiguous().numpy().tobytes()).hexdigest();quality,details=metric._check_mask_quality(pm,return_details=True);points=pred[0]['points_'+e+'_in_query'][j]
                        distance=metric._compute_point_distance(points,pts_orig)/float(np.hypot(*src.shape)) if len(points) and len(pts_orig) else 2.
                        evidence[e].update(hard_votes=float(metric._count_points_in_mask(points,src)),hard_distance=float(distance),has_points=float(bool(len(points))),sam_iou=float(pred[0]['pred_iou_score_'+e][j].float().mean()),area=float(pm.float().mean()),components=float(details['num_components']),quality_valid=bool(quality))
                        metrics[e]=[float(b[k+'_'+e]) for k in ('iou','dice','shape_acc','location_score')]
                    base=b['best_expert'];vals=[float(b[k]) for k in ('IoU','Dice','shape_acc','location_score')];assert np.allclose(metrics[base],vals,atol=1e-7,rtol=0)
                    ref_omama=None;reference_expert=None;drift=False;bank_file=None
                    if key in prior:
                        p=prior[key];identity=masks==p['mask_sha256'] and base==p['pccs_expert'] and np.allclose(vals,p['metrics']['pccs'],atol=1e-7,rtol=0)
                        if args.phase!='exo2exo':assert identity,('Training reference mismatch',key)
                        checked+=bool(identity);drift=not identity;drift_count+=drift
                        if args.phase=='exo2exo' and identity:ref_omama=p['metrics']['consensus'];reference_expert=p['consensus_expert']
                        if drift:
                            assert json.loads((R/'manifest.json').read_text()).get('within_bank_reference_repair'),'Reference protocol not declared'
                            arrays={};oldbank=np.load(p['bank_file']);difference={}
                            for e in ('query','visual','anchor','fusion'):
                                value=src.cpu().numpy() if e=='query' else pred[0]['pred_masks_'+e][j].cpu().numpy();binary=(value>0).astype(np.uint8);arrays[e]=np.packbits(binary);arrays[e+'_shape']=np.asarray(binary.shape)
                                if e!='query':old=np.unpackbits(oldbank[e])[:int(np.prod(oldbank[e+'_shape']))].reshape(oldbank[e+'_shape']);difference[e]=int(np.count_nonzero(old!=binary))
                            bank_file=str(bankroot/(hashlib.sha256((key[0]+'|'+key[1]).encode()).hexdigest()[:28]+'.npz'));assert not Path(bank_file).exists();np.savez_compressed(bank_file,**arrays)
                            (out/('drift_'+hashlib.sha256(key[0].encode()).hexdigest()[:16]+'.json')).write_text(json.dumps({'video_id':key[0],'obj_id':key[1],'pixel_differences':difference,'old_expert':p['pccs_expert'],'new_expert':base,'old_metrics':p['metrics']['pccs'],'new_metrics':vals,'new_mask_sha256':masks},indent=2))
                    q=ann[vid]['prompt']['first_frame_image'];q=q[0] if isinstance(q,list) else q
                    row={'video_id':vid,'obj_id':key[1],'take_id':q.split('/')[0],'baseline':base,'source_valid':bool(src.any()),'mask_sha256':masks,'evidence':evidence,'metrics':metrics,'original_reason':b['selection_reason'],'frozen_omama_reference':ref_omama,'frozen_omama_expert':reference_expert,'historical_drift':drift,'bank_file':bank_file}
                    batch_records.append(row)
                if args.phase=='exo2exo':
                    pred[0]['dense_context']=dense;native=mm.PCCSMetric(collect_device='cpu',enable_vis=False,routing_policy='fusion_first',diagnostic_output=None);native.native_dense_config=json.loads((R/'selection.json').read_text())['selected'];native.process(batch,pred);integrated=[r for pair in native.results for r in pair]
                    for row,decision in zip(batch_records,integrated):row['integrated_primary']=decision['best_expert']
                if args.phase=='smoke':
                    zero=mm.PCCSMetric(collect_device='cpu',enable_vis=False,routing_policy='fusion_first',diagnostic_output=None);zero.native_dense_config={'strength':0};zero.process(batch,pred);zero_rows=[r for pair in zero.results for r in pair];assert [r['best_expert'] for r in zero_rows]==[r['best_expert'] for r in rows]
                for j,row in enumerate(batch_records):
                    assert {e:hashlib.sha256(pred[0]['pred_masks_'+e][j].cpu().contiguous().numpy().tobytes()).hexdigest() for e in ('visual','anchor','fusion')}==row['mask_sha256'],'Native decision changed frozen masks'
                    f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                if (bi+1)%4==0 or bi+1==spec['pairs']:status(state='running',done=bi+1,total=spec['pairs'],objects=len(seen),eta_seconds=(time.monotonic()-start)/max(1,bi+1-initial_pairs)*(spec['pairs']-bi-1))
        assert seen==expected;receipt={'state':'complete','coverage':'exact','pairs':spec['pairs'],'objects':len(seen),'reference_objects_verified':checked,'historical_drift_objects':drift_count,'capture_on_off_pairs':off,'records_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};(out/'receipt.json').write_text(json.dumps(receipt,indent=2));status(**receipt,eta_seconds=0);print('DENSE_WORKER_COMPLETE',args.phase,args.rank,flush=True)
    except Exception as e:status(state='failed',error=repr(e));raise
if __name__=='__main__':main()
