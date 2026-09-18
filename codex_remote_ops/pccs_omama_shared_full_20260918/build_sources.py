"""Derive native worker mechanically; the only inference change is saving its masks."""
from pathlib import Path
R=Path(__file__).parent;F=R.parent/'pccs_roi_full_exoego_20260917';B=R.parent/'pccs_omama_full_matched_20260918'
(R/'common.py').write_text((B/'common.py').read_text())
worker=(F/'worker.py').read_text()
needle="bankroot=R/'candidate_bank';bankroot.mkdir(exist_ok=True)"
assert worker.count(needle)==1
worker=worker.replace(needle,"bankroot=R/'candidate_bank'/args.phase/f'rank{args.rank}';bankroot.mkdir(parents=True,exist_ok=True)")
needle='                    batch_records.append(row)'
insert='''                    # Persist exactly the three native predictions before either selector runs.
                    arrays={'query':src.detach().cpu().numpy(),**{e:pred[0]['pred_masks_'+e][j].detach().cpu().numpy() for e in ('visual','anchor','fusion')}};packed={}
                    for e,v in arrays.items():
                        assert np.isin(v,[0,1]).all();packed[e]=np.packbits(v.astype(np.uint8));packed[e+'_shape']=np.asarray(v.shape)
                    bp=bankroot/(hashlib.sha256((vid+':'+str(key[1])).encode()).hexdigest()+'.npz');np.savez_compressed(bp,**packed)
                    row.update(bank_file=str(bp),bank_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),query_sha256=hashlib.sha256(arrays['query'].astype(np.uint8).tobytes()).hexdigest())
                    batch_records.append(row)'''
assert worker.count(needle)==1;worker=worker.replace(needle,insert)
needle="                f.write(''.join(json.dumps(row,allow_nan=False)+'\\n' for row in batch_records));f.flush()"
insert="""                for row in batch_records:row['choices']={'baseline':row['baseline'],'primary':row['integrated_primary'],'local20_matched':row['integrated_local20'],'frozen_cycle':row['integrated_frozen_cycle']}
                if bi==0: (out/'startup.json').write_text(json.dumps({'finite_metrics':all(np.isfinite(list(r['metrics'].values())).all() for r in batch_records),'gpu_peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'saved_mask_bank':True,'actual_routes_verified':True}))
"""+needle
assert worker.count(needle)==1;worker=worker.replace(needle,insert);(R/'worker.py').write_text(worker)
ref=(B/'reference.py').read_text().replace("R/'runs'/('rebuild_'+args.phase)","R/'runs'/args.phase").replace('same_current_seed1_candidates','same_shared_candidates')
ref=ref.replace("        scores,choice=score(*paths(wa)","        active_images=images;images=Path(yaml.safe_load((C/'full_runtime.yaml').read_text())['data']['images'])\n        scores,choice=score(*paths(wa)")
ref=ref.replace("        rows=verified_rows(","        images=active_images\n        rows=verified_rows(")
(R/'reference.py').write_text(ref)
evaluation=(B/'evaluate.py').read_text()
old="        spec=m['phases'][args.phase]['shards'][rank];assert sha(spec['prior'])==spec['prior_sha256'];prior=[json.loads(s) for s in Path(spec['prior']).read_text().splitlines()];banks=verified_rows(R/'runs'/('rebuild_'+args.phase)/f'rank{rank}'/'records.jsonl');new=verified_rows(R/'runs'/('reference_'+args.phase)/f'rank{rank}'/'records.jsonl');assert len(prior)==len(banks)==len(new)"
new="        banks=verified_rows(R/'runs'/args.phase/f'rank{rank}'/'records.jsonl');prior=banks;new=verified_rows(R/'runs'/('reference_'+args.phase)/f'rank{rank}'/'records.jsonl');assert len(prior)==len(banks)==len(new)"
assert old in evaluation;evaluation=evaluation.replace(old,new).replace('all_masks_identical_to_current_full1','all_methods_share_saved_candidate_masks').replace("args.phase=='pilot'","args.phase=='smoke1'")
evaluation=evaluation.replace("write(R/(args.phase+'_results.json'),result)","result['scope']='Fresh shared generation, not original full1 masks';write(R/(args.phase+'_results.json'),result)")
needle="    methods={k:measure(rows,v) for k,v in choices.items()};contrasts={}"
insert="""    import gzip
    with gzip.open(R/(args.phase+'_paired_compact.jsonl.gz'),'wt') as f:
        for i,r in enumerate(rows):f.write(json.dumps({**{k:r[k] for k in ('video_id','obj_id','take_id','baseline','candidate_seed','mask_sha256','metrics')},'choices':{k:v[i] for k,v in choices.items()}})+'\\n')
"""+needle
assert needle in evaluation;evaluation=evaluation.replace(needle,insert);(R/'evaluate.py').write_text(evaluation)
print('SHARED_SOURCES_READY')
