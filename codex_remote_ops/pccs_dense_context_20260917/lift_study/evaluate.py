from pathlib import Path
import sys,json,hashlib,copy,time
R=Path(__file__).parent;P=R.parent;sys.path[:0]=[str(P/'local_deps'),str(P)]
import joblib
from data_utils import read,score
from lift_policy import scores,choose
frozen_bytes=(R/'selection.json').read_bytes();frozen=json.loads(frozen_bytes);cfg=frozen['selected'];rows=read('exo2exo');results={};predictions={}
def evaluate(c,wrong=False):
    path=Path(c['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256'];model=joblib.load(path);chosen=[]
    for row in rows:
        r=copy.deepcopy(row) if wrong else row
        if wrong:
            for e in r['evidence']:
                for k in list(r['evidence'][e]):
                    if k.startswith('ctx'):r['evidence'][e][k]=r['evidence'][e][k.replace('ctx','wrong',1)]
        selected=choose(r,scores(r,model,c['group']),c['gate'],c['threshold']);chosen.append(selected)
        counter=copy.deepcopy(r);counter['metrics']=None;assert choose(counter,scores(counter,model,c['group']),c['gate'],c['threshold'])==selected
    return chosen
for name,c,wrong in [('primary',cfg,False),('wrong_context',cfg,True)]+[(c['group']+'_matched',c,False) for c in frozen['matched_controls']]:
    pred=evaluate(c,wrong);predictions[name]=pred;results[name]=score(rows,pred)
control=next(c for c in json.loads((R/'search.json').read_text())['candidates'] if c['name']=='dominance_cycle_control_ridge10_t0.03');assert evaluate(control)==[r['integrated_primary'] for r in rows],'R1 bridge decision control mismatch'
refs=copy.deepcopy(rows)
for r in refs:r['metrics']['omama_reference']=r['frozen_omama_reference'];r['mask_sha256']['omama_reference']=r['mask_sha256'][r['frozen_omama_expert']];r['baseline']='omama_reference'
summary={'selected':cfg,'methods':results,'primary_vs_omama':score(refs,predictions['primary']),'coverage':'exact','r1_integrated_control_exact':True,'gt_label_counterfactual':True,'cached_policy_replay':True,'selection_sha256':hashlib.sha256(frozen_bytes).hexdigest(),'evaluated_at':time.time()}
(R/'target_results.json').write_text(json.dumps(summary,indent=2));(R/'predictions.jsonl').write_text(''.join(json.dumps({'video_id':r['video_id'],'obj_id':r['obj_id'],'selected':{n:pred[i] for n,pred in predictions.items()}})+'\n' for i,r in enumerate(rows)));print(json.dumps({n:(v['frame'][0]*100,v['delta_pp'],v['ci95_pp']) for n,v in results.items()},indent=2));print('VS_OMAMA',summary['primary_vs_omama']['delta_pp'],summary['primary_vs_omama']['ci95_pp'])
