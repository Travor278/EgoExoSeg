"""Local verification against the published full1 scalar/mask-hash records."""
from pathlib import Path
import argparse,gzip,json,hashlib,sys
import numpy as np
R=Path(__file__).parent;F=R.parent/'pccs_roi_full_exoego_20260917'
sys.path.insert(0,str(F))
import importlib.util
s=importlib.util.spec_from_file_location('original_full_verifier',F/'verify_results.py');mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('pilot','full'),required=True);a=p.parse_args()
    with gzip.open(F/'full1_compact.jsonl.gz','rt') as f:original={(r['video_id'],r['obj_id']):r for r in map(json.loads,f)}
    references=[]
    for rank in range(4):
        d=R/'runs'/('reference_'+a.phase)/f'rank{rank}';path=d/'records.jsonl';receipt=json.loads((d/'receipt.json').read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['records_sha256'];rows=list(map(json.loads,path.read_text().splitlines()));assert len(rows)==receipt['objects'];references.extend(rows)
        w=json.loads((d/'witness.json').read_text());assert w['all_choices_reproduced'] and w['max_score_error']<1e-4
    keys=[(r['video_id'],r['obj_id']) for r in references];assert len(set(keys))==len(keys) and set(keys)<=set(original)
    plan=json.loads((R/'manifest.json').read_text());assert len(keys)==plan['phases'][a.phase]['objects'];assert len({k[0] for k in keys})==plan['phases'][a.phase]['pairs']
    if a.phase=='full':assert set(keys)==set(original)
    rows=[original[k] for k in keys];result=json.loads((R/(a.phase+'_results.json')).read_text())
    for r,x in zip(rows,references):
        assert r['mask_sha256']==x['mask_sha256'];assert all(np.isfinite(x['scores'][m]).all() for m in ('canonical','native_interp'))
    choices={k:[r['choices'][k] for r in rows] for k in ('baseline','primary','local20_matched','frozen_cycle')}
    choices.update({'omama_'+mode:[x['selected'][mode] for x in references] for mode in ('canonical','native_interp','consensus')})
    def check(rs,cs,v):
        frame,delta,ci=mod.aggregate(rs,cs);assert np.allclose(frame,v['frame'],atol=1e-12,rtol=0) and abs(delta-v['delta_pp'])<1e-9 and np.allclose(ci,v['ci95_pp'],atol=1e-9,rtol=0)
    for k,cs in choices.items():check(rows,cs,result['methods'][k])
    for k in ('primary','local20_matched'):check([{**r,'baseline':e} for r,e in zip(rows,choices['omama_consensus'])],choices[k],result['paired_comparisons'][k+' minus omama_consensus'])
    validation={'phase':a.phase,'pairs':len({k[0] for k in keys}),'objects':len(rows),'exact_key_coverage':True,'current_full1_mask_hash_identity':True,'frame_metrics_and_take_bootstrap':'passed','historical_score_and_selection_witnesses':'passed'}
    (R/(a.phase+'_validation.json')).write_text(json.dumps(validation,indent=2));print(json.dumps(validation))
if __name__=='__main__':main()
