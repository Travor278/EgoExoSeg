"""Independent aggregation from downloadable compact records; no original images/masks needed."""
from pathlib import Path
from collections import defaultdict
import gzip,hashlib,json,os
import numpy as np
from data_utils import R,read

def aggregate(rows,choices):
    pairs=defaultdict(list)
    for r,e in zip(rows,choices):pairs[r['video_id']].append((r['take_id'],r['metrics'][e],r['metrics'][r['baseline']]))
    means=[];takes=defaultdict(list)
    for items in pairs.values():
        values=np.mean([i[1] for i in items],axis=0);base=np.mean([i[2] for i in items],axis=0);means.append(values);takes[items[0][0]].append(values[0]-base[0])
    sums=np.array([sum(v) for v in takes.values()]);counts=np.array([len(v) for v in takes.values()]);draw=np.random.default_rng(20260917).integers(0,len(counts),size=(10000,len(counts)));ci=np.quantile(sums[draw].sum(1)/counts[draw].sum(1),[.025,.975])*100
    return np.mean(means,axis=0),100*sums.sum()/counts.sum(),ci

def main():
    m=json.loads((R/'manifest.json').read_text());audit=json.loads((R/'weight_audit.json').read_text());old=R.parent/'pccs_roi_context_20260917';assert hashlib.sha256((R/'selection.json').read_bytes()).hexdigest()==m['frozen_selection_sha256']==hashlib.sha256((old/'selection.json').read_bytes()).hexdigest();assert audit['expert_assets_verified_by_bytes_and_sha256']==m['assets'];assert audit['previous_omama_full_used_identical_assets']
    checks={'weights_and_frozen_selection':'passed','phases':{}}
    if (R/'science_results.json').exists():
        results=json.loads((R/'science_results.json').read_text())
        for phase,v in results['phases'].items():
            rows=read(phase);prior={}
            for p in (old/'runs'/m['phases'][phase]['source_phase']).glob('rank*/records.jsonl'):prior.update({(r['video_id'],r['obj_id']):r for r in map(json.loads,p.read_text().splitlines())})
            for r in rows:assert r['mask_sha256']==prior[(r['video_id'],r['obj_id'])]['mask_sha256'] and r['baseline']==prior[(r['video_id'],r['obj_id'])]['baseline']
            for name,summary in v['methods'].items():
                f,delta,ci=aggregate(rows,[r['choices'][name] for r in rows]);assert np.allclose(f,summary['frame'],atol=1e-12,rtol=0) and abs(delta-summary['delta_pp'])<1e-9 and np.allclose(ci,summary['ci95_pp'],atol=1e-9,rtol=0)
            for mode in ('local15','local20'):
                for arm in ('far','rolled'):
                    refs=[{**r,'baseline':r['choices'][arm+'_'+mode]} for r in rows];_,delta,ci=aggregate(refs,[r['choices']['real_'+mode] for r in rows]);summary=v['real_minus_intervention']['real minus '+arm+' '+mode];assert abs(delta-summary['delta_pp'])<1e-9 and np.allclose(ci,summary['ci95_pp'],atol=1e-9,rtol=0)
            checks['phases'][phase]={'objects':len(rows),'paired_frame_metrics_and_bootstrap':'passed','same_frozen_candidates':'passed'}
    for seed in (1,2):
        phase='full'+str(seed);p=R/(phase+'_results.json');data=R/(phase+'_compact.jsonl.gz')
        if not p.exists() or not data.exists():continue
        with gzip.open(data,'rt') as f:rows=list(map(json.loads,f))
        result=json.loads(p.read_text());assert len(rows)==109253 and len({(r['video_id'],r['obj_id']) for r in rows})==109253 and len({r['video_id'] for r in rows})==46515 and len({r['take_id'] for r in rows})==295
        for r in rows:
            expected=int.from_bytes(hashlib.sha256(('candidate-quality-v'+str(seed)+':'+r['video_id']).encode()).digest()[:4],'little')%(2**31-1);assert r['candidate_seed']==expected
        for name,summary in result['methods'].items():
            frame,delta,ci=aggregate(rows,[r['choices'][name] for r in rows]);assert np.allclose(frame,summary['frame'],atol=1e-12,rtol=0) and abs(delta-summary['delta_pp'])<1e-9 and np.allclose(ci,summary['ci95_pp'],atol=1e-9,rtol=0)
        checks['phases'][phase]={'objects':len(rows),'pairs':46515,'takes':295,'candidate_seeds':'passed','paired_frame_metrics_and_bootstrap':'passed','compact_file_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'baseline_zero_iou_objects':sum(r['metrics'][r['baseline']][0]==0 for r in rows)}
    (R/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
if __name__=='__main__':main()
