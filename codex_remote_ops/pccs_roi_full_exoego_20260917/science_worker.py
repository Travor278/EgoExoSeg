from pathlib import Path
from types import SimpleNamespace
import argparse,copy,hashlib,importlib.util,json,time
R=Path(__file__).parent;P=R.parent/'pccs_roi_context_20260917'
def main():
    import numpy as np,torch,yaml,joblib
    from PIL import Image
    from science_views import collect,ARMS
    from encoder_contract import fingerprint
    from native_bridge import native_select
    from roi_policy import predict
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--rank',type=int,required=True);args=p.parse_args();m=json.loads((R/'manifest.json').read_text());phase=m['phases'][args.phase];spec=phase['shards'][args.rank];out=R/'runs'/args.phase/f'rank{args.rank}';out.mkdir(parents=True,exist_ok=False)
    def status(**kw):
        q=out/'status.tmp';q.write_text(json.dumps({**kw,'updated_at':time.time()}));q.replace(out/'status.json')
    try:
        for n,h in m['code_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
        oldphase=phase['source_phase'];cfg=yaml.safe_load(Path(spec['config']).read_text());ann=json.loads(Path(spec['annotation']).read_text());images=Path(cfg['data']['images']);rows=[]
        for rank in range(4):
            q=P/'runs'/oldphase/f'rank{rank}'/'records.jsonl';assert hashlib.sha256(q.read_bytes()).hexdigest()==json.loads((q.parent/'receipt.json').read_text())['records_sha256'];rows.extend(r for r in map(json.loads,q.read_text().splitlines()) if r['video_id'] in ann)
        expected={(str(v),str(o)) for v,a in ann.items() for o in a['objects']};assert {(r['video_id'],r['obj_id']) for r in rows}==expected
        contract=m['encoder_contract'];torch.backends.cuda.matmul.allow_tf32=contract['matmul_tf32'];torch.backends.cudnn.allow_tf32=contract['cudnn_tf32'];assert not contract['autocast_enabled']
        modspec=importlib.util.spec_from_file_location('native_backward',R/'code/projects/v2sam_pccs/models/pccs_components/backward_correspondence.py');mod=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(mod)
        model=mod.load_dinov3_model(repo_path=cfg['models']['dinov3']['repo'],weights_path=cfg['models']['dinov3']['weights']).to('cuda').eval();matcher=mod.BackwardCorrespondenceMatcher(model).to('cuda').eval();fp=fingerprint(matcher);assert fp['state_sha256']==contract['state_sha256'];(out/'encoder_receipt.json').write_text(json.dumps(fp))
        selection=json.loads((R/'selection.json').read_text());configs={'local15':selection['selected'],'local20':next(c for c in selection['matched_controls'] if c['group']=='local20')};models={k:joblib.load(c['checkpoint']) for k,c in configs.items()};bridges={(arm,k):SimpleNamespace(native_dense_config=c) for arm in ARMS for k,c in configs.items()};start=time.monotonic();path=out/'records.jsonl'
        with path.open('w') as f:
            for i,original in enumerate(rows):
                r=copy.deepcopy(original);bank=r['bank_file'];assert bank
                with np.load(bank) as z:
                    def unpack(e):return np.unpackbits(z[e])[:int(np.prod(z[e+'_shape']))].reshape(z[e+'_shape']).astype(np.uint8)
                    source=unpack('query');masks={e:unpack(e) for e in ('visual','anchor','fusion')}
                assert {e:hashlib.sha256(v.tobytes()).hexdigest() for e,v in masks.items()}==r['mask_sha256']
                a=ann[r['video_id']];q=a['prompt']['first_frame_image'];t=a['video_path'];q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t
                with Image.open(images/q) as im:si=im.convert('RGB')
                with Image.open(images/t) as im:ti=im.convert('RGB')
                evidence,meta=collect(matcher,si,ti,source,masks,args.phase=='science_smoke');error=0.
                for e,values in evidence['real'].items():
                    for k,v in values.items():error=max(error,abs(v-r['evidence'][e][k]));assert np.isclose(v,r['evidence'][e][k],atol=1e-5,rtol=1e-4),('Real-view control drift',r['video_id'],e,k)
                decisions={}
                for arm in ARMS:
                    row=copy.deepcopy(r);row['metrics']=None
                    for e in row['evidence']:row['evidence'][e].update(evidence[arm][e])
                    prediction={'dense_context':[row['evidence']],**{'pred_masks_'+e:torch.from_numpy(v)[None] for e,v in masks.items()}}
                    for mode,c in configs.items():
                        direct=predict(row,models[mode],c['group'],c['threshold'],c['gate']);actual=native_select(bridges[(arm,mode)],prediction,torch.from_numpy(source),0,r['baseline']);assert direct==actual;decisions[arm+'_'+mode]=direct
                        if arm=='real':assert direct==predict(r,models[mode],c['group'],c['threshold'],c['gate'])
                output={k:r[k] for k in ('video_id','obj_id','take_id','baseline','mask_sha256','metrics')};output.update(choices=decisions,intervention_evidence=evidence,metadata=meta,real_max_feature_error=error);f.write(json.dumps(output,allow_nan=False)+'\n');f.flush()
                if (i+1)%8==0 or i+1==len(rows):status(state='running',done=i+1,total=len(rows),eta_seconds=(time.monotonic()-start)/(i+1)*(len(rows)-i-1))
        receipt={'state':'complete','coverage':'exact','pairs':len(ann),'objects':len(rows),'records_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'foreground_preserved':True,'real_control_feature_replay':True,'actual_bridge_matches_gt_free_policy':True};(out/'receipt.json').write_text(json.dumps(receipt,indent=2));status(**receipt,eta_seconds=0)
    except Exception as e:status(state='failed',error=repr(e));raise
if __name__=='__main__':main()
