"""External comparator only: frozen O-MaMa on the new masks, in a separate process."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np,torch,yaml
R=Path(__file__).parent;Q=R.parent/'pccs_candidate_quality_20260916';U=R.parent/'pccs_candidate_union_20260916'

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('reference_probe','reference'),required=True);p.add_argument('--rank',type=int,required=True);args=p.parse_args();out=R/'runs'/args.phase/f'rank{args.rank}';out.mkdir(parents=True,exist_ok=False)
    def status(**kw):
        p=out/'status.tmp';p.write_text(json.dumps({**kw,'updated_at':time.time()}));p.replace(out/'status.json')
    try:
        m=json.loads((R/'manifest.json').read_text());spec=m['phases']['exo2exo']['shards'][args.rank];ann=json.loads(Path(spec['annotation']).read_text());cfg=yaml.safe_load(Path(spec['config']).read_text());images=Path(cfg['data']['images'])
        oldpath=U/'runs/exo2exo/baseline/per_object.jsonl';assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==json.loads((oldpath.parent/'receipt.json').read_text())['records_sha256'];old={(r['video_id'],r['obj_id']):r for r in map(json.loads,oldpath.read_text().splitlines())}
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        module_spec=importlib.util.spec_from_file_location('external_frozen_omama',Q/'aligned_model.py');module=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=module;module_spec.loader.exec_module(module)
        matcher=module.AlignedMatcher('cuda')
        def run(row,path):
            with np.load(path) as z:
                def unpack(e):return np.unpackbits(z[e])[:int(np.prod(z[e+'_shape']))].reshape(z[e+'_shape']).astype(np.uint8)
                qm=unpack('query');masks={e:unpack(e) for e in ('visual','anchor','fusion')}
            assert {e:hashlib.sha256(v.tobytes()).hexdigest() for e,v in masks.items()}==row['mask_sha256'];base=row['baseline'];names=[base];used={row['mask_sha256'][base]}
            for e in ('visual','anchor','fusion'):
                h=row['mask_sha256'][e]
                if h not in used:used.add(h);names.append(e)
            masks=[masks[e] for e in names];a=ann[row['video_id']];q=a['prompt']['first_frame_image'];t=a['video_path'];q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t;chosen=[];scores={}
            for mode in ('canonical','native_interp'):
                values,_=matcher.score(images/q,images/t,qm,masks,'exo2ego',mode) if qm.any() else ({'omama_learned':[0.]*len(masks)},None);s=values['omama_learned'];scores[mode]=s;valid=[i for i,mask in enumerate(masks) if mask.any()];best=max(valid,key=lambda i:s[i]) if valid else 0
                if not qm.any() or (masks[0].any() and s[best]<=s[0]+.05):best=0
                chosen.append(names[best])
            return chosen[0] if chosen[0]==chosen[1] else base,scores
        # Always use a known OLD bank witness, never assume some new-seed masks stayed equal.
        vid=next(iter(ann));oid=str(next(iter(ann[vid]['objects'])));prior=old[(vid,oid)];witness={'video_id':vid,'obj_id':oid,'mask_sha256':prior['mask_sha256'],'baseline':prior['pccs_expert']};expert,ws=run(witness,prior['bank_file']);assert expert==prior['consensus_expert'];error=max(float(np.max(np.abs(np.asarray(ws[mode])-prior['scores'][mode]))) for mode in ws);assert error<1e-4,('Frozen reference witness score discrepancy',error)
        witness_receipt={'video_id':vid,'obj_id':oid,'max_score_error':error,'expert':expert,'old_bank_witness':True};(out/'witness.json').write_text(json.dumps(witness_receipt,indent=2))
        if args.phase=='reference_probe':status(state='complete',done=1,total=1,eta_seconds=0,witness=witness_receipt);return
        path=R/'runs/exo2exo'/f'rank{args.rank}'/'records.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads((path.parent/'receipt.json').read_text())['records_sha256'];rows=list(map(json.loads,path.read_text().splitlines()));expected={(str(v),str(o)) for v,a in ann.items() for o in a['objects']};assert {(r['video_id'],r['obj_id']) for r in rows}==expected
        start=time.monotonic();result=out/'references.jsonl';computed=reused=0
        with result.open('w') as f:
            for i,row in enumerate(rows):
                reuse=row['frozen_omama_reference'] is not None
                if reuse:expert=row['frozen_omama_expert'];scores=None;reused+=1;assert not row['seed1_bank_changed']
                else:expert,scores=run(row,row['bank_file']);computed+=1
                entry={'video_id':row['video_id'],'obj_id':row['obj_id'],'expert':expert,'metrics':row['metrics'][expert],'mask_sha256':row['mask_sha256'],'scores':scores,'reused_identical_bank':reuse};f.write(json.dumps(entry)+'\n');f.flush()
                if (i+1)%8==0 or i+1==len(rows):status(state='running',done=i+1,total=len(rows),computed=computed,reused=reused,eta_seconds=(time.monotonic()-start)/(i+1)*(len(rows)-i-1))
        receipt={'state':'complete','objects':len(rows),'computed':computed,'reused':reused,'records_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),'witness':witness_receipt};(out/'receipt.json').write_text(json.dumps(receipt,indent=2));status(**receipt,eta_seconds=0)
    except Exception as e:status(state='failed',error=repr(e));raise
if __name__=='__main__':main()
