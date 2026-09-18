"""Local paired aggregation and cross-run drift audit from scalar/hash-only records."""
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json
import numpy as np
R=Path(__file__).parent;F=R.parent/'pccs_roi_full_exoego_20260917'
import sys
sys.path.insert(0,str(F))
s=importlib.util.spec_from_file_location('original_frame_verifier',F/'verify_results.py');module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('smoke1','full1'),required=True);a=p.parse_args()
    data=R/(a.phase+'_paired_compact.jsonl.gz')
    with gzip.open(data,'rt') as f:rows=list(map(json.loads,f))
    spec=json.loads((R/'manifest.json').read_text())['phases'][a.phase];result=json.loads((R/(a.phase+'_results.json')).read_text());keys={(r['video_id'],r['obj_id']) for r in rows};assert len(keys)==len(rows)==spec['objects'];assert len({k[0] for k in keys})==spec['pairs']
    def check(rs,cs,v):
        frame,delta,ci=module.aggregate(rs,cs);assert np.allclose(frame,v['frame'],atol=1e-12,rtol=0) and abs(delta-v['delta_pp'])<1e-9 and np.allclose(ci,v['ci95_pp'],atol=1e-9,rtol=0)
    for k,v in result['methods'].items():check(rows,[r['choices'][k] for r in rows],v)
    for k,v in result['paired_comparisons'].items():
        method,base=k.split(' minus ');check([{**r,'baseline':r['choices'][base]} for r in rows],[r['choices'][method] for r in rows],v)
    for r in rows:
        assert all(e in r['mask_sha256'] for e in r['choices'].values())
        assert r['candidate_seed']==int.from_bytes(hashlib.sha256(('candidate-quality-v1:'+r['video_id']).encode()).digest()[:4],'little')%(2**31-1)
    validation={'phase':a.phase,'pairs':spec['pairs'],'objects':len(rows),'frame_and_take_bootstrap':'passed','same_run_paired_comparison':True,'raw_masks_remain_remote':True,'compact_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
    if a.phase=='full1':
        with gzip.open(F/'full1_compact.jsonl.gz','rt') as f:old={(r['video_id'],r['obj_id']):r for r in map(json.loads,f)}
        assert keys==set(old) and len({r['take_id'] for r in rows})==295
        validation['cross_run_diagnostic_only']={'objects_with_any_mask_drift':sum(r['mask_sha256']!=old[(r['video_id'],r['obj_id'])]['mask_sha256'] for r in rows),'objects_with_baseline_route_drift':sum(r['baseline']!=old[(r['video_id'],r['obj_id'])]['baseline'] for r in rows)}
    (R/(a.phase+'_validation.json')).write_text(json.dumps(validation,indent=2));print(json.dumps(validation))
if __name__=='__main__':main()
