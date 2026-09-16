from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib
import numpy as np
R=Path(__file__).parent;ARMS=('baseline','weighted_pool','reliable_points','both');METHODS=('pccs','consensus','oracle','visual','anchor','fusion')
def read_phase(phase,arms=ARMS):
    plan=json.loads((R/'manifest.json').read_text());spec=plan['phases'][phase];ann=json.loads(Path(spec['annotation']).read_text());expected={(k,str(o)) for k,r in ann.items() for o in r['objects']};data={};summaries={};frames={}
    for arm in arms:
        folder=R/'runs'/phase/arm;p=folder/'per_object.jsonl';receipt=json.loads((folder/'receipt.json').read_text());assert receipt['coverage']=='exact' and receipt['records_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
        rows=[json.loads(s) for s in p.read_text().splitlines()];lookup={(r['video_id'],r['obj_id']):r for r in rows};assert len(rows)==len(expected) and set(lookup)==expected
        data[arm]=lookup;fp=defaultdict(list)
        for row in rows:fp[row['video_id']].append(row)
        frames[arm]={method:{vid:np.mean([r['metrics'][method] for r in group],axis=0) for vid,group in fp.items()} for method in METHODS}
        summaries[arm]={'pairs':len(fp),'objects':len(rows),'frame':{m:np.mean(list(frames[arm][m].values()),axis=0).tolist() for m in METHODS},'object':{m:np.mean([r['metrics'][m] for r in rows],axis=0).tolist() for m in METHODS},'multipoint_objects':sum(r['point_count']>1 for r in rows)}
    checks={}
    if set(arms)==set(ARMS):
        for a,b,e in [('baseline','weighted_pool','anchor'),('baseline','reliable_points','visual'),('weighted_pool','both','visual'),('reliable_points','both','anchor')]:
            mismatches=[k for k in expected if data[a][k]['mask_sha256'][e]!=data[b][k]['mask_sha256'][e]];checks[f'{a}_vs_{b}_{e}_unchanged']={'mismatches':len(mismatches),'examples':mismatches[:4]}
        assert all(c['mismatches']==0 for c in checks.values()),('Cross-arm isolation failed',checks)
    comparisons={}
    for arm in arms:
        if arm=='baseline':continue
        comparisons[arm]={}
        for method in ('pccs','consensus','oracle'):
            groups=defaultdict(list)
            for vid,v in frames[arm][method].items():
                q=ann[vid]['prompt']['first_frame_image'];q=q[0] if isinstance(q,list) else q;take=q.split('/')[0];groups[take].append(float(v[0]-frames['baseline'][method][vid][0]))
            ds=np.array([sum(v) for v in groups.values()]);ns=np.array([len(v) for v in groups.values()]);rng=np.random.default_rng(20260916);draw=rng.integers(0,len(ds),size=(5000,len(ds)));boot=ds[draw].sum(1)/ns[draw].sum(1)
            comparisons[arm][method]={'delta_iou_pp':float(100*ds.sum()/ns.sum()),'take_bootstrap_95ci_pp':(100*np.quantile(boot,[.025,.975])).tolist()}
    out={'phase':phase,'coverage':'exact','arms':summaries,'comparisons':comparisons,'isolation_checks':checks,'metric_order':['IoU','Dice','ContA','LocE'],'scope':'official TRAIN screening/calibration, not unseen evaluation' if phase!='exo2exo' else 'preselected Exo2Exo transfer confirmation; benchmark previously observed'}
    (R/(phase+'_summary.json')).write_text(json.dumps(out,indent=2));return out
def decide():
    screen=json.loads((R/'screen_summary.json').read_text());cal=json.loads((R/'calibration_summary.json').read_text());eligible=[]
    for arm in ARMS[1:]:
        if screen['comparisons'][arm]['oracle']['delta_iou_pp']>0 and cal['comparisons'][arm]['oracle']['delta_iou_pp']>0 and cal['comparisons'][arm]['consensus']['delta_iou_pp']>0:eligible.append(arm)
    selected=max(eligible,key=lambda a:cal['arms'][a]['frame']['consensus'][0]) if eligible else 'baseline'
    decision={'selected':selected,'eligible':eligible,'rule':'positive oracle on screen+calibration and positive calibration consensus; maximize calibration consensus','test_labels_used':False,'new_expert_training':False}
    (R/'selection.json').write_text(json.dumps(decision,indent=2));return decision
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase');x=p.parse_args()
    if x.phase=='decide':print(json.dumps(decide()))
    else:print(json.dumps(read_phase(x.phase)))
