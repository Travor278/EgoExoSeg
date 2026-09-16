"""Validate the second frozen policy at the actual native PCCS bridge on cached masks."""
from types import SimpleNamespace
import json,hashlib,copy
import numpy as np,torch,joblib
from data_utils import R,read,score
from native_bridge import native_select
from gate_policy import scores,choose
U=R.parent/'pccs_candidate_union_20260916';rows=read('exo2exo');frozen=json.loads((R/'gate_selection_deployed.json').read_text());cfg=frozen['selected'];model=joblib.load(cfg['checkpoint']);old={(r['video_id'],r['obj_id']):r for r in map(json.loads,(U/'runs/exo2exo/baseline/per_object.jsonl').read_text().splitlines())};metric=SimpleNamespace(native_dense_config=cfg);chosen=[];wrong=[]
assert hashlib.sha256(open(cfg['checkpoint'],'rb').read()).hexdigest()==cfg['sha256']
assert native_select(SimpleNamespace(native_dense_config={'strength':0}),{},None,0,'fusion')=='fusion'
for r in rows:
    p=r.get('bank_file') or old[(r['video_id'],r['obj_id'])]['bank_file'];z=np.load(p)
    def unpack(e):return np.unpackbits(z[e])[:int(np.prod(z[e+'_shape']))].reshape(z[e+'_shape']).astype(np.uint8)
    pred={'dense_context':[r['evidence']],**{'pred_masks_'+e:torch.from_numpy(unpack(e))[None] for e in ('visual','anchor','fusion')}};source=torch.from_numpy(unpack('query'))
    actual=native_select(metric,pred,source,0,r['baseline']);replay=choose(r,scores(r,model,cfg['group']),cfg['gate'],cfg['threshold']);assert actual==replay
    corrupted=copy.deepcopy(r);corrupted['metrics']=None;assert choose(corrupted,scores(corrupted,model,cfg['group']),cfg['gate'],cfg['threshold'])==actual
    chosen.append(actual);w=copy.deepcopy(r)
    for e in w['evidence']:
        for k in list(w['evidence'][e]):
            if k.startswith('ctx'):w['evidence'][e][k]=w['evidence'][e][k.replace('ctx','wrong',1)]
    wrong.append(choose(w,scores(w,model,cfg['group']),cfg['gate'],cfg['threshold']))
results={'primary':score(rows,chosen),'wrong_context':score(rows,wrong),'r1_dominance_same_model':score(rows,[choose(r,scores(r,model,cfg['group']),'dominance',cfg['threshold']) for r in rows])}
for c in frozen['matched_controls']:
    m=joblib.load(c['checkpoint']);assert hashlib.sha256(open(c['checkpoint'],'rb').read()).hexdigest()==c['sha256'];pred=[choose(r,scores(r,m,c['group']),c['gate'],c['threshold']) for r in rows];results[c['group']+'_matched']=score(rows,pred)
refs=copy.deepcopy(rows)
for r in refs:r['metrics']['omama_reference']=r['frozen_omama_reference'];r['mask_sha256']['omama_reference']=r['mask_sha256'][r['frozen_omama_expert']];r['baseline']='omama_reference'
summary={'selected':cfg,'methods':results,'primary_vs_omama':score(refs,chosen),'native_bridge_replay_exact':True,'gt_label_counterfactual':True,'strength_zero_parity':True,'coverage':'exact','historical_drift_objects':sum(r.get('historical_drift',False) for r in rows),'source_selection_frozen':True}
(R/'gate_target_results.json').write_text(json.dumps(summary,indent=2));(R/'gate_predictions.jsonl').write_text(''.join(json.dumps({'video_id':r['video_id'],'obj_id':r['obj_id'],'selected':e})+'\n' for r,e in zip(rows,chosen)));print('GATE_TARGET_COMPLETE',results['primary']['delta_pp'],flush=True)
