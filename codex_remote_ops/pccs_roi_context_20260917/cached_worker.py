"""Add local native features to a byte-verified candidate bank, without rerunning SAM."""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,hashlib,importlib.util,json,os,sys,time
R=Path(__file__).parent

def main():
    import numpy as np,torch,yaml
    from PIL import Image
    from roi_evidence import collect,MODES,KEYS
    from native_bridge import native_select
    from roi_policy import predict
    from policies import BASE,DENSE
    from encoder_contract import fingerprint
    import joblib
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--rank',type=int,required=True);args=p.parse_args()
    plan=json.loads((R/'manifest.json').read_text());spec=plan['phases'][args.phase]['shards'][args.rank];out=R/'runs'/args.phase/f'rank{args.rank}';out.mkdir(parents=True,exist_ok=False)
    def status(**kw):
        p=out/'status.tmp';p.write_text(json.dumps({**kw,'updated_at':time.time()}));p.replace(out/'status.json')
    try:
        for n,h in plan['code_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
        raw=yaml.safe_load(Path(spec['config']).read_text());ann=json.loads(Path(spec['annotation']).read_text());images=Path(raw['data']['images']);expected={(str(v),str(o)) for v,a in ann.items() for o in a['objects']}
        cached_smoke=args.phase=='cache_smoke';src=R if cached_smoke else R.parent/'pccs_fixed_context_20260917';phase='smoke' if cached_smoke else 'exo2exo';records=[]
        for rank in range(4):
            q=src/'runs'/phase/f'rank{rank}'/'records.jsonl';receipt=json.loads((q.parent/'receipt.json').read_text());assert hashlib.sha256(q.read_bytes()).hexdigest()==receipt['records_sha256'];records.extend(map(json.loads,q.read_text().splitlines()))
        rows=[r for r in records if (r['video_id'],r['obj_id']) in expected];assert len(rows)==len(expected)
        banks={}
        if not cached_smoke:
            q=R.parent/'pccs_candidate_union_20260916/runs/exo2exo/baseline/per_object.jsonl';receipt=json.loads((q.parent/'receipt.json').read_text());assert hashlib.sha256(q.read_bytes()).hexdigest()==receipt['records_sha256'];banks={(r['video_id'],r['obj_id']):r['bank_file'] for r in map(json.loads,q.read_text().splitlines())}
        contract=json.loads((R/'runs/smoke/rank0/encoder_receipt.json').read_text())
        torch.backends.cuda.matmul.allow_tf32=contract['matmul_tf32'];torch.backends.cudnn.allow_tf32=contract['cudnn_tf32']
        specmod=importlib.util.spec_from_file_location('native_backward',R/'code/projects/v2sam_pccs/models/pccs_components/backward_correspondence.py');module=importlib.util.module_from_spec(specmod);specmod.loader.exec_module(module)
        model=module.load_dinov3_model(repo_path=raw['models']['dinov3']['repo'],weights_path=raw['models']['dinov3']['weights']).to(device='cuda',dtype=getattr(torch,contract['parameter_dtype'].split('.')[-1])).eval()
        matcher=module.BackwardCorrespondenceMatcher(model).to('cuda').eval();actual=fingerprint(matcher);assert actual['state_sha256']==contract['state_sha256'],'Standalone encoder differs from PCCS encoder';assert actual['mean']==contract['mean'] and actual['std']==contract['std']
        (out/'encoder_receipt.json').write_text(json.dumps(actual,indent=2));cfg=None
        if not cached_smoke:
            cfg=json.loads((R/'selection.json').read_text())['selected'];checkpoint=Path(cfg['checkpoint']);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==cfg['sha256'];calibrator=joblib.load(checkpoint);metric=SimpleNamespace(native_dense_config=cfg)
        seen=set();start=time.monotonic();path=out/'records.jsonl';total=len(rows)
        with path.open('w') as handle:
            for i,original in enumerate(rows):
                r=copy.deepcopy(original);key=(r['video_id'],r['obj_id']);assert key not in seen;seen.add(key);bank=r.get('bank_file') or banks[key]
                if not cached_smoke:r['fixed_context_reference_expert']=r['integrated_primary']
                keep=set(BASE+DENSE+('valid','quality_valid','q_area','s_area'))
                r['evidence']={e:{k:v for k,v in x.items() if k in keep} for e,x in r['evidence'].items()}
                with np.load(bank) as z:
                    def unpack(e):return np.unpackbits(z[e])[:int(np.prod(z[e+'_shape']))].reshape(z[e+'_shape']).astype(np.uint8)
                    source=unpack('query');masks={e:unpack(e) for e in ('visual','anchor','fusion')}
                assert {e:hashlib.sha256(m.tobytes()).hexdigest() for e,m in masks.items()}==r['mask_sha256']
                a=ann[r['video_id']];q=a['prompt']['first_frame_image'];t=a['video_path'];q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t
                with Image.open(images/q) as im:source_image=im.convert('RGB')
                with Image.open(images/t) as im:target_image=im.convert('RGB')
                with torch.autocast('cuda',enabled=contract['autocast_enabled'],dtype=getattr(torch,contract['autocast_dtype'].split('.')[-1])):
                    roi,meta=collect(matcher,source_image,target_image,source,masks,batch_test=cached_smoke and i==0)
                if cached_smoke:
                    for e in roi:
                        for k,v in roi[e].items():assert np.isclose(v,original['evidence'][e][k],atol=1e-5,rtol=1e-4),('Cached/source feature mismatch',key,e,k,v,original['evidence'][e][k])
                for e in roi:r['evidence'][e].update(roi[e])
                r['bank_file']=bank;r['roi_metadata']=meta;r['cached_candidates']=True;r['gpu_peak_bytes']=torch.cuda.max_memory_allocated()
                if cfg:
                    prediction={'dense_context':[r['evidence']],**{'pred_masks_'+e:torch.from_numpy(m)[None] for e,m in masks.items()}}
                    actual_choice=native_select(metric,prediction,torch.from_numpy(source),0,r['baseline']);assert actual_choice==predict(r,calibrator,cfg['group'],cfg['threshold'],cfg['gate'])
                    corrupt=copy.deepcopy(r);corrupt['metrics']=None;assert predict(corrupt,calibrator,cfg['group'],cfg['threshold'],cfg['gate'])==actual_choice
                    assert native_select(SimpleNamespace(native_dense_config={'strength':0}),prediction,torch.from_numpy(source),0,r['baseline'])==r['baseline']
                    r['integrated_primary']=actual_choice
                handle.write(json.dumps(r,allow_nan=False)+'\n');handle.flush()
                if (i+1)%4==0 or i+1==total:status(state='running',done=i+1,total=total,eta_seconds=(time.monotonic()-start)/(i+1)*(total-i-1))
        assert seen==expected;receipt={'state':'complete','coverage':'exact','pairs':len(ann),'objects':len(rows),'records_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cached_candidates':True,'source_encoder_state_parity':True,'source_roi_feature_parity':cached_smoke,'actual_bridge_and_label_counterfactual':bool(cfg)};(out/'receipt.json').write_text(json.dumps(receipt,indent=2));status(**receipt,eta_seconds=0)
    except Exception as e:status(state='failed',error=repr(e));raise
if __name__=='__main__':main()
