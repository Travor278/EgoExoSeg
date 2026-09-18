import argparse
from common import *
from collections import defaultdict
import numpy as np
def aggregate(rows,choices):
    pairs=defaultdict(list)
    for r,e in zip(rows,choices):pairs[r['video_id']].append((r['take_id'],r['metrics'][e],r['metrics'][r['baseline']]))
    means=[];takes=defaultdict(list)
    for items in pairs.values():
        values=np.mean([i[1] for i in items],axis=0);base=np.mean([i[2] for i in items],axis=0);means.append(values);takes[items[0][0]].append(values[0]-base[0])
    sums=np.array([sum(v) for v in takes.values()]);counts=np.array([len(v) for v in takes.values()]);draw=np.random.default_rng(20260917).integers(0,len(counts),size=(10000,len(counts)));ci=np.quantile(sums[draw].sum(1)/counts[draw].sum(1),[.025,.975])*100
    return np.mean(means,axis=0),100*sums.sum()/counts.sum(),ci
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);args=p.parse_args();m=read(R/'manifest.json');rows=[];refs={}
    for rank in range(4):
        spec=m['phases'][args.phase]['shards'][rank];assert sha(spec['prior'])==spec['prior_sha256'];prior=[json.loads(s) for s in Path(spec['prior']).read_text().splitlines()];banks=verified_rows(R/'runs'/('rebuild_'+args.phase)/f'rank{rank}'/'records.jsonl');new=verified_rows(R/'runs'/('reference_'+args.phase)/f'rank{rank}'/'records.jsonl');assert len(prior)==len(banks)==len(new)
        lookup={(r['video_id'],r['obj_id']):r for r in banks}
        for r in prior:assert lookup[(r['video_id'],r['obj_id'])]['mask_sha256']==r['mask_sha256']
        rows.extend(prior);refs.update({(r['video_id'],r['obj_id']):r for r in new})
    assert len(refs)==len(rows)==m['phases'][args.phase]['objects'];assert len({r['video_id'] for r in rows})==m['phases'][args.phase]['pairs']
    for r in rows:assert refs[(r['video_id'],r['obj_id'])]['mask_sha256']==r['mask_sha256']
    choices={k:[r['choices'][k] for r in rows] for k in ('baseline','primary','local20_matched','frozen_cycle')}
    choices.update({'omama_'+mode:[refs[(r['video_id'],r['obj_id'])]['selected'][mode] for r in rows] for mode in ('canonical','native_interp','consensus')})
    def measure(rs,cs):
        frame,delta,ci=aggregate(rs,cs);return {'frame':frame.tolist(),'delta_pp':delta,'ci95_pp':ci.tolist()}
    methods={k:measure(rows,v) for k,v in choices.items()};contrasts={}
    for k in ('primary','local20_matched'):
        contrast=[{**r,'baseline':e} for r,e in zip(rows,choices['omama_consensus'])];contrasts[k+' minus omama_consensus']=measure(contrast,choices[k])
    result={'phase':args.phase,'coverage':'exact','pairs':len({r['video_id'] for r in rows}),'objects':len(rows),'takes':len({r['take_id'] for r in rows}),'all_masks_identical_to_current_full1':True,'methods':methods,'paired_comparisons':contrasts,'pilot_is_not_full_benchmark':args.phase=='pilot'};write(R/(args.phase+'_results.json'),result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
