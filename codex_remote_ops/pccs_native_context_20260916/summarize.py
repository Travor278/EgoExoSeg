from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
from configs import CONFIGS,ELIGIBLE
R=Path(__file__).parent
def read_phase(phase):
    allrows=[];checks=[]
    for rank in range(4):
        p=R/'runs'/phase/f'rank{rank}'/'records.jsonl';rec=json.loads((p.parent/'receipt.json').read_text());assert rec['coverage']=='exact' and rec['records_sha256']==hashlib.sha256(p.read_bytes()).hexdigest();allrows.extend(map(json.loads,p.read_text().splitlines()));checks.append(rec)
    spec=json.loads((R/'manifest.json').read_text())['phases'][phase];assert len(allrows)==spec['objects'];keys={(r['video_id'],r['obj_id']) for r in allrows};assert len(keys)==len(allrows)
    pairs=defaultdict(list)
    for r in allrows:pairs[r['video_id']].append(r)
    assert len(pairs)==spec['pairs'];methods={}
    for name in CONFIGS:
        frame=np.mean([np.mean([r['methods'][name]['metrics'] for r in group],axis=0) for group in pairs.values()],axis=0);changes=[r for r in allrows if r['methods'][name]['expert']!=r['methods']['baseline']['expert']]
        groups=defaultdict(list)
        for group in pairs.values():groups[group[0]['take_id']].append(float(np.mean([r['methods'][name]['metrics'][0]-r['methods']['baseline']['metrics'][0] for r in group])))
        sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()]);draw=np.random.default_rng(20260916).integers(0,len(counts),size=(10000,len(counts)));ci=np.quantile(sums[draw].sum(1)/counts[draw].sum(1),[.025,.975])*100
        methods[name]={'frame':frame.tolist(),'delta_pp':float(sums.sum()/counts.sum()*100),'ci95_pp':ci.tolist(),'changed':len(changes),'improved':sum(r['methods'][name]['metrics'][0]>r['methods']['baseline']['metrics'][0]+1e-8 for r in changes),'harmed':sum(r['methods'][name]['metrics'][0]<r['methods']['baseline']['metrics'][0]-1e-8 for r in changes)}
    result={'phase':phase,'coverage':'exact','pairs':len(pairs),'objects':len(allrows),'takes':len({r['take_id'] for r in allrows}),'methods':methods,'capture_on_off_pairs':sum(v['capture_on_off_pairs'] for v in checks),'all_prior_candidates_exact':True,'lambda_zero_parity':True,'fusion_accepted_objects':sum(r['original_fusion_accepted'] for r in allrows),'fusion_recheck_objects':sum(r['original_fusion_accepted'] and r['fusion_context_recheck'] for r in allrows),'reliable_objects_by_expert':{e:sum(r['context'][e]['reliability']>=.2 for r in allrows) for e in ('visual','anchor','fusion')},'pretrained_omama_used':False}
    assert methods['zero']['changed']==0 and methods['zero']['delta_pp']==0
    (R/(phase+'_results.json')).write_text(json.dumps(result,indent=2));return result
def select():
    a=json.loads((R/'screen_results.json').read_text());b=json.loads((R/'calibration_results.json').read_text());eligible=[n for n in ELIGIBLE if a['methods'][n]['delta_pp']>0 and b['methods'][n]['delta_pp']>0];winner=max(eligible,key=lambda n:b['methods'][n]['frame'][0]) if eligible else 'baseline'
    result={'selected':winner,'config':CONFIGS[winner],'eligible':eligible,'rule':'positive native-PCCS gain in both TRAIN screen/cal; maximum calibration; fixed before target; wrong-context excluded','pretrained_omama_used':False};(R/'selection.json').write_text(json.dumps(result,indent=2));return result
