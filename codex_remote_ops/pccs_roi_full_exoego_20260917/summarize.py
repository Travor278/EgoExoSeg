"""Frozen policies scored on complete full-test native evidence; pair-equal metrics."""
from collections import defaultdict
import os,json,hashlib,copy,time
from pathlib import Path
import numpy as np
from data_utils import R,read,score
from roi_policy import predict
from policies import calibrated_select
def main():
    import joblib
    phase=os.environ['PCCS_SUMMARY_PHASE'];rows=read(phase);m=json.loads((R/'manifest.json').read_text());sel=json.loads((R/'selection.json').read_text());assert hashlib.sha256((R/'selection.json').read_bytes()).hexdigest()==m['frozen_selection_sha256'];choices={'baseline':[r['baseline'] for r in rows]}
    for name,cfg in [('primary',sel['selected']),('frozen_cycle',m['frozen_cycle'])]+[(c['group']+'_matched',c) for c in sel['matched_controls']]:
        path=Path(cfg['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['sha256'];model=joblib.load(path)
        if cfg['family']=='roi':choices[name]=[predict(r,model,cfg['group'],cfg['threshold'],cfg['gate']) for r in rows]
        else:choices[name]=[calibrated_select(r,model,cfg['group'],cfg['threshold']) for r in rows]
        print('FULL_POLICY_SCORED',name,flush=True)
    assert choices['primary']==[r['integrated_primary'] for r in rows] and choices['local20_matched']==[r['integrated_local20'] for r in rows] and choices['frozen_cycle']==[r['integrated_frozen_cycle'] for r in rows]
    results={k:score(rows,v) for k,v in choices.items()};contrasts={}
    for a,b in [('primary','frozen_cycle'),('local20_matched','frozen_cycle'),('local20_matched','object20_matched'),('global_local20_matched','global_object20_matched')]:
        refs=[{**r,'baseline':e} for r,e in zip(rows,choices[b])];contrasts[a+' minus '+b]=score(refs,choices[a])
    counts={'pairs':len({r['video_id'] for r in rows}),'objects':len(rows),'takes':len({r['take_id'] for r in rows})};assert counts=={'pairs':46515,'objects':109253,'takes':295}
    result={'coverage':'exact','phase':phase,**counts,'methods':results,'paired_comparisons':contrasts,'selection_sha256':m['frozen_selection_sha256'],'expert_assets':m['assets'],'no_target_fitting':True,'full_actual_primary_local20_cycle_equals_replay':True,'baseline_zero_iou_objects':sum(r['metrics'][r['baseline']][0]==0 for r in rows),'historical_omama_comparison_note':'Prior corrected-weight full O-MaMa benchmark used the same experts but a different candidate run; do not interpret cross-run subtraction as paired comparison.'};(R/(phase+'_results.json')).write_text(json.dumps(result,indent=2))
    with (R/(phase+'_compact.jsonl')).open('w') as f:
        for i,r in enumerate(rows):f.write(json.dumps({**{k:r[k] for k in ('video_id','obj_id','take_id','baseline','candidate_seed','mask_sha256','metrics')},'choices':{k:v[i] for k,v in choices.items()}})+'\n')
    import gzip,shutil
    p=R/(phase+'_compact.jsonl')
    with p.open('rb') as src,gzip.open(str(p)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
    print('FULL_COMPLETE',json.dumps(counts),flush=True)
if __name__=='__main__':main()
