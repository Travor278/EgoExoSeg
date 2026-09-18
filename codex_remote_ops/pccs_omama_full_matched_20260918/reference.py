import argparse,importlib.util,sys,time
from common import *
def main():
    import numpy as np,torch,yaml
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True);ap.add_argument('--rank',type=int,required=True);args=ap.parse_args()
    out=R/'runs'/('reference_'+args.phase)/f'rank{args.rank}';out.mkdir(parents=True,exist_ok=True);sp=out/'status.json'
    try:
        plan=read(R/'manifest.json');spec=plan['phases'][args.phase]['shards'][args.rank];ann=read(spec['annotation']);images=Path(yaml.safe_load(Path(spec['config']).read_text())['data']['images'])
        for n,h in plan['reference_code_sha256'].items():assert sha(C/n)==h
        for n,h in plan['reference_weights'].items():assert sha(C/'weights'/n)==h
        torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def load(name,path):
            s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
        adapter=load('historical_omama_adapter',C/'aligned_model.py');cache=load('historical_bank_cache',C/'cache.py');matcher=adapter.AlignedMatcher('cuda')
        def score(q,t,qm,ms,names):
            scores={};chosen={}
            for mode in ('canonical','native_interp'):
                s=matcher.score(images/q,images/t,qm,ms,'exo2ego',mode)[0]['omama_learned'] if qm.any() else [0.]*len(ms)
                assert np.isfinite(s).all();scores[mode]=s;chosen[mode]=cache.select(s,names,[bool(x.any()) for x in ms]) if qm.any() else 'baseline'
            chosen['consensus']=chosen['canonical'] if chosen['canonical']==chosen['native_interp'] else 'baseline'
            return scores,chosen
        # A historical known-bank witness checks the published head, precision and threshold.
        hist=C/'runs/full';scored=hist/'scored'/f'rank{args.rank}.jsonl';rec=read(scored.with_name(f'rank{args.rank}_receipt.json'));assert sha(scored)==rec['output_sha256']
        with scored.open() as f:w=json.loads(next(f))
        with (hist/'candidates/exo2ego/per_object.jsonl').open() as f:
            old=next(x for x in map(json.loads,f) if (x['video_id'],x['obj_id'])==(w['video_id'],w['obj_id']))
        qm,names,ms,digest=cache.masks(old);assert digest==w['candidate_mask_sha256'] and names==w['names'];wa=read(C/'full_annotation.json')[w['video_id']]
        def paths(a):
            q=a['prompt']['first_frame_image'];t=a['video_path'];return (q[0] if isinstance(q,list) else q,t[0] if isinstance(t,list) else t)
        scores,choice=score(*paths(wa),qm,ms,names);error=max(float(np.max(np.abs(np.asarray(scores[m])-w['scores'][k]))) for m,k in [('canonical','omama_learned'),('native_interp','native_omama_learned')]);assert error<1e-4
        assert choice['canonical']==w['selections']['omama_margin005'] and choice['native_interp']==w['selections']['native_margin005'] and choice['consensus']==w['selections']['geometry_consensus']
        write(out/'witness.json',{'video_id':w['video_id'],'obj_id':w['obj_id'],'max_score_error':error,'all_choices_reproduced':True,'tf32':False,'model_load_receipts':matcher.load_receipts})
        rows=verified_rows(R/'runs'/('rebuild_'+args.phase)/f'rank{args.rank}'/'records.jsonl');expected={(r['video_id'],r['obj_id']) for r in rows};path=out/'records.jsonl';previous=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else [];seen={(r['video_id'],r['obj_id']) for r in previous};assert len(seen)==len(previous) and seen<=expected
        start=time.monotonic();initial=len(seen)
        with path.open('a') as f:
            for row in rows:
                key=(row['video_id'],row['obj_id'])
                if key in seen:continue
                assert sha(row['bank_file'])==row['bank_sha256']
                with np.load(row['bank_file']) as z:
                    def unpack(e):return np.unpackbits(z[e])[:int(np.prod(z[e+'_shape']))].reshape(z[e+'_shape']).astype(np.uint8)
                    qm=unpack('query');masks={e:unpack(e) for e in EXPERTS}
                assert {e:hashlib.sha256(x.tobytes()).hexdigest() for e,x in masks.items()}==row['mask_sha256'];assert hashlib.sha256(qm.tobytes()).hexdigest()==row['query_sha256']
                base=row['baseline'];names=['baseline'];ms=[masks[base]];used={row['mask_sha256'][base]}
                for e in EXPERTS:
                    h=row['mask_sha256'][e]
                    if h not in used:names.append(e);ms.append(masks[e]);used.add(h)
                scores,choice=score(*paths(ann[row['video_id']]),qm,ms,names)
                selected={k:(base if e=='baseline' else e) for k,e in choice.items()}
                entry={'video_id':row['video_id'],'obj_id':row['obj_id'],'mask_sha256':row['mask_sha256'],'bank_sha256':row['bank_sha256'],'names':names,'scores':scores,'selected':selected}
                f.write(json.dumps(entry,allow_nan=False)+'\n');f.flush();seen.add(key)
                if len(seen)==initial+1:write(out/'startup.json',{'finite_scores':True,'same_candidate_bank':True,'gpu_peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'witness_passed':True})
                if len(seen)%16==0 or seen==expected:status(sp,state='running',done=len(seen),total=len(rows),eta_seconds=(time.monotonic()-start)/max(1,len(seen)-initial)*(len(rows)-len(seen)))
        assert seen==expected;receipt={'state':'complete','coverage':'exact','objects':len(seen),'records_sha256':sha(path),'same_current_seed1_candidates':True};write(out/'receipt.json',receipt);status(sp,**receipt,eta_seconds=0)
    except Exception as e:status(sp,state='failed',error=repr(e));raise
if __name__=='__main__':main()
