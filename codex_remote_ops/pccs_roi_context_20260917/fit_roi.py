"""Source-only native PCCS calibration with matched feature and gate controls."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
from data_utils import R,read,score
from roi_policy import GROUPS,GATES,features,predict
from policies import EXPERTS

def main():
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    tr,cal=read('train'),read('calibration');assert {r['take_id'] for r in tr}.isdisjoint({r['take_id'] for r in cal});fitted=R/'fitted';fitted.mkdir();results=[]
    folds=json.loads((R.parent/'pccs_dense_context_20260917/folds.json').read_text());splits=[]
    for fold in folds:
        fit_ids=[i for i,r in enumerate(tr) if r['take_id'] in fold['fit_takes']];val_ids=[i for i,r in enumerate(tr) if r['take_id'] in fold['held_takes']];assert len(fit_ids)+len(val_ids)==len(tr);splits.append((fit_ids,val_ids))
    def create(kind):
        if kind=='tree':return Pipeline([('model',HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=3,min_samples_leaf=25,l2_regularization=20,random_state=20260917))])
        return Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10 if kind=='ridge10' else 100))])
    def fit(m,rows,group):
        counts=Counter(r['video_id'] for r in rows);x=[];y=[];w=[]
        for r in rows:
            options=[e for e in EXPERTS if e!=r['baseline'] and r['source_valid'] and r['evidence'][e]['quality_valid'] and r['mask_sha256'][e]!=r['mask_sha256'][r['baseline']]]
            for e in options:x.append(features(r,e,group));y.append(r['metrics'][e][0]-r['metrics'][r['baseline']][0]);w.append(1/(counts[r['video_id']]*len(options)))
        x=np.asarray(x);y=np.asarray(y);w=np.asarray(w);assert np.isfinite(x).all() and np.isfinite(y).all();m.fit(x,y,model__sample_weight=w/w.mean());return m
    for group in GROUPS:
        for kind in ('ridge10','ridge100','tree'):
            oof={(g,t):[None]*len(tr) for g in GATES for t in (.03,.05)}
            for fit_ids,val_ids in splits:
                m=fit(create(kind),[tr[i] for i in fit_ids],group)
                for (g,t),choices in oof.items():
                    for i in val_ids:choices[i]=predict(tr[i],m,group,t,g)
            m=fit(create(kind),tr,group);path=fitted/(group+'_'+kind+'.joblib');joblib.dump(m,path);sha=hashlib.sha256(path.read_bytes()).hexdigest()
            for (gate,t),choices in oof.items():
                assert all(x is not None for x in choices)
                cfg={'name':f'{group}_{kind}_{gate}_t{t}','family':'roi','group':group,'kind':kind,'gate':gate,'threshold':t,'checkpoint':str(path),'sha256':sha,'train_oof':score(tr,choices),'calibration':score(cal,[predict(r,m,group,t,gate) for r in cal])};results.append(cfg)
            (R/'search.json').write_text(json.dumps({'candidates':results},indent=2));print('ROI_CALIBRATED',group,kind,flush=True)
    old=json.loads((R.parent/'pccs_dense_context_20260917/search.json').read_text())['candidates']
    for c in results:
        if c['group']=='global' and c['gate']=='dominance':
            ref=next(x for x in old if x['name']==f"cycle_{c['kind']}_t{c['threshold']}");assert abs(c['train_oof']['delta_pp']-ref['train_oof']['delta_pp'])<1e-7;assert abs(c['calibration']['delta_pp']-ref['calibration']['delta_pp'])<1e-7
    eligible=[c for c in results if c['train_oof']['delta_pp']>0 and c['calibration']['delta_pp']>0];selected=max(eligible,key=lambda c:c['calibration']['frame'][0]) if eligible else {'family':'baseline','name':'baseline'}
    matched=[c for c in results if c['kind']==selected.get('kind') and c['gate']==selected.get('gate') and c['threshold']==selected.get('threshold')]
    selection={'selected':selected,'matched_controls':matched,'eligible':[c['name'] for c in eligible],'rule':'Positive TRAIN take-OOF and disjoint TRAIN calibration; maximize calibration; tie chooses first predeclared config. No target fitting.','gate_contract':'Within each gate all feature groups have exactly the same admission rule, independent of ROI evidence','pretrained_omama_used':False}
    (R/'folds.json').write_text(json.dumps(folds,indent=2));(R/'selection.json').write_text(json.dumps(selection,indent=2));print('ROI_SELECTION_FROZEN',selected['name'],flush=True)
if __name__=='__main__':main()
