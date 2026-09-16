"""Build union features and frozen O-MaMa embeddings; supervision stays separate."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,time,re
import numpy as np
from union_core import catalog,feature_vector
R=Path(__file__).parent;P=R.parent/'pccs_candidate_quality_20260916';SOURCES=['baseline','reliable_points','residual005','residual010','residual025']
def load_rows(root,phase,arm):
    p=root/'runs'/phase/arm/'per_object.jsonl';receipt=json.loads((p.parent/'receipt.json').read_text());assert receipt['records_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[json.loads(s) for s in p.read_text().splitlines()];lookup={(r['video_id'],r['obj_id']):r for r in rows};assert len(rows)==len(lookup);return lookup
def unpack(z,k):return np.unpackbits(z[k])[:int(np.prod(z[k+'_shape']))].reshape(z[k+'_shape']).astype(np.uint8)
def geometry(mask,base,query):
    def single(m):
        ys,xs=np.where(m>0);area=float(m.mean())
        if not len(xs):return area,0.,.5,.5
        return area,float(len(xs)/((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))),float(xs.mean()/m.shape[1]),float(ys.mean()/m.shape[0])
    a,fill,x,y=single(mask);b,bfill,bx,by=single(base);inter=float(np.logical_and(mask,base).sum());union=float(np.logical_or(mask,base).sum())
    return [a,b,float(np.log((a+1e-6)/(b+1e-6))),fill,bfill,float(np.hypot(x-bx,y-by)),inter/max(union,1.),float(query.mean()),float(mask.any())]
def embed(matcher,qp,tp,qm,masks,mode):
    import torch
    import torch.nn.functional as F
    from aligned_model import preprocess,bbox,position,learned_scores
    h,g=matcher.heads['exo2ego'];device=matcher.device
    with torch.inference_mode():
        qi,qms,qs=preprocess(qp,[qm],mode,g[0],device);ti,tm,ts=preprocess(tp,masks,mode,g[1],device)
        batch={'SOURCE_img':qi,'SOURCE_mask':qms,'SOURCE_bbox':bbox(qms[0])[None],'SOURCE_img_size':qs,'GT_img':ti,'DEST_SAM_masks':tm[None],'DEST_SAM_bbox':torch.stack([bbox(m) for m in tm])[None],'DEST_img_size':ts}
        qd,qf=matcher.extractor.get_SOURCE_descriptors(batch);td,tf=matcher.extractor.get_DEST_descriptors(batch)
        qt=qf.flatten(2).transpose(1,2)+position(h.pos_embed_Q,g[0],qf.shape[-2:]);tt=tf.flatten(2).transpose(1,2)+position(h.pos_embed_T,g[1],tf.shape[-2:])
        aq=h.CROSS_context_attn(qd[:,:,:768],tt,residual=False);at=h.CROSS_context_attn(td[:,:,:768],qt,residual=False)
        q=F.normalize(h.mlp(torch.cat((aq,qd),2)),dim=2)[0,0];t=F.normalize(h.mlp(torch.cat((at,td),2)),dim=2)[0]
        score=F.cosine_similarity(q[None],t,dim=1);ref=learned_scores(h,qd,td,qf,tf,g);torch.testing.assert_close(score,ref,atol=1e-6,rtol=1e-6)
        return q.cpu().numpy(),t.cpu().numpy(),score.cpu().numpy()
def main():
    import torch,yaml
    from aligned_model import AlignedMatcher
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);a=p.parse_args();phase=a.phase
    allrows={s:load_rows(P if phase!='exo2exo' and s in SOURCES[:2] else R,phase,s) for s in SOURCES};keys=set(allrows['baseline'])
    assert all(set(x)==keys for x in allrows.values())
    manifest=json.loads((R/'manifest.json').read_text());spec=manifest['phases'][phase];ann=json.loads(Path(spec['annotation']).read_text());cfg=yaml.safe_load(Path(spec['config']).read_text());images=Path(cfg['data']['images'])
    target=R/'datasets'/phase;target.mkdir(parents=True,exist_ok=True);output=target/'records.jsonl'
    if output.exists():raise RuntimeError('Fresh union dataset required')
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;matcher=AlignedMatcher('cuda');maxerr=0.;start=time.monotonic()
    with output.open('w') as handle:
        for index,key in enumerate(sorted(keys)):
            records=[(s,allrows[s][key]) for s in SOURCES];base=records[0][1]
            # Residual pooling must not change the anchor branch.
            assert all(allrows[s][key]['mask_sha256']['anchor']==base['mask_sha256']['anchor'] for s in SOURCES[2:])
            cs,labels,fallback,err=catalog(records);maxerr=max(maxerr,err);pool={};qm=None
            for source,row in records:
                z=np.load(row['bank_file'])
                if source=='baseline':qm=unpack(z,'query')
                for e in ('visual','anchor','fusion'):
                    ident=row['mask_sha256'][e]
                    if ident not in pool:
                        m=unpack(z,e);assert hashlib.sha256(m.tobytes()).hexdigest()==ident;pool[ident]=m
            masks=[pool[c['id']] for c in cs];fallback_index=next(i for i,c in enumerate(cs) if c['id']==fallback)
            for c in cs:c['geometry']=geometry(pool[c['id']],pool[fallback],qm)
            rec=ann[key[0]];q=rec['prompt']['first_frame_image'];t=rec['video_path'];q=q[0] if isinstance(q,list) else q;t=t[0] if isinstance(t,list) else t
            data={};valid=bool(qm.any())
            for mode in ('canonical','native_interp'):
                qz,tz,scores=embed(matcher,images/q,images/t,qm,masks,mode) if valid else (np.zeros(768,np.float32),np.zeros((len(cs),768),np.float32),np.zeros(len(cs),np.float32))
                original=np.asarray([c['scores'][0 if mode=='canonical' else 1] for c in cs]);replay=float(np.max(np.abs(scores-original)));assert replay<1e-4,('union score parity',key,mode,replay);maxerr=max(maxerr,replay)
                data['q_'+mode]=qz;data['t_'+mode]=tz
            stem=hashlib.sha256((key[0]+'|'+key[1]).encode()).hexdigest()[:28];ep=target/(stem+'.npz');np.savez_compressed(ep,**data)
            take=re.search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',q).group()
            item={'video_id':key[0],'obj_id':key[1],'take_id':take,'candidates':cs,'fallback':fallback,'fallback_index':fallback_index,'query_valid':valid,'labels':[labels[c['id']] for c in cs],'baseline_metrics':base['metrics']['consensus'],'baseline_pccs':base['metrics']['pccs'],'baseline_oracle':base['metrics']['oracle'][0],'embedding_file':str(ep)}
            assert max(v[0] for v in item['labels'])+1e-7>=item['baseline_oracle']
            handle.write(json.dumps(item,allow_nan=False)+'\n');handle.flush()
            if (index+1)%10==0 or index+1==len(keys):
                state={'state':'running','phase':phase,'done':index+1,'total':len(keys),'eta_seconds':(time.monotonic()-start)/(index+1)*(len(keys)-index-1)};(R/('dataset_'+phase+'_status.json')).write_text(json.dumps(state))
    receipt={'state':'complete','objects':len(keys),'pairs':len(ann),'max_score_replay_error':maxerr,'records_sha256':hashlib.sha256(output.read_bytes()).hexdigest()};(target/'receipt.json').write_text(json.dumps(receipt,indent=2));(R/('dataset_'+phase+'_status.json')).write_text(json.dumps(receipt));print('UNION_DATASET_COMPLETE',phase,flush=True)
if __name__=='__main__':main()
