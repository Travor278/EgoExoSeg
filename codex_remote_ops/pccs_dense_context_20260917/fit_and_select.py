"""Take-held-out confidence calibration inside the native PCCS challenger mechanism."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
from data_utils import R,read,score
from policies import EXPERTS,GROUPS,HAND,features,hand_select,calibrated_select

def main():
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    tr,cal=read('train'),read('calibration');assert {r['take_id'] for r in tr}.isdisjoint({r['take_id'] for r in cal});results=[];root=R/'fitted';root.mkdir(exist_ok=True)
    def record(cfg,oof,cp):
        x={**cfg,'train_oof':score(tr,oof),'calibration':score(cal,cp)};results.append(x);(R/'search.json').write_text(json.dumps({'candidates':results},indent=2));return x
    for cfg in HAND:record({'name':cfg['name'],'family':'hand','config':cfg},[hand_select(r,cfg) for r in tr],[hand_select(r,cfg) for r in cal])
    def model(kind):
        if kind=='tree':return Pipeline([('model',HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=3,min_samples_leaf=25,l2_regularization=20,random_state=20260917))])
        return Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10 if kind=='ridge10' else 100))])
    def fit(m,rows,group):
        counts=Counter(r['video_id'] for r in rows);x=[];y=[];w=[]
        for r in rows:
            options=[e for e in EXPERTS if e!=r['baseline'] and r['source_valid'] and r['evidence'][e]['quality_valid'] and r['mask_sha256'][e]!=r['mask_sha256'][r['baseline']]]
            for e in options:x.append(features(r,e,group));y.append(r['metrics'][e][0]-r['metrics'][r['baseline']][0]);w.append(1/(counts[r['video_id']]*len(options)))
        x=np.asarray(x);y=np.asarray(y);w=np.asarray(w);assert len(y)>20 and np.isfinite(x).all() and np.isfinite(y).all();m.fit(x,y,model__sample_weight=w/w.mean());return m
    splits=list(GroupKFold(n_splits=4).split(np.arange(len(tr)),groups=[r['take_id'] for r in tr]));fold_receipts=[]
    for fi,(fit_ids,val_ids) in enumerate(splits):
        a={tr[i]['take_id'] for i in fit_ids};b={tr[i]['take_id'] for i in val_ids};assert a.isdisjoint(b);fold_receipts.append({'fold':fi,'fit_takes':sorted(a),'held_takes':sorted(b)})
    for group in GROUPS:
        for kind in ('ridge10','ridge100','tree'):
            oof={t:[None]*len(tr) for t in (.03,.05)}
            for fit_ids,val_ids in splits:
                m=fit(model(kind),[tr[i] for i in fit_ids],group)
                for t in oof:
                    for i in val_ids:oof[t][i]=calibrated_select(tr[i],m,group,t)
            m=fit(model(kind),tr,group);path=root/(group+'_'+kind+'.joblib');joblib.dump(m,path);sha=hashlib.sha256(path.read_bytes()).hexdigest()
            for t,choices in oof.items():
                assert all(c is not None for c in choices)
                record({'name':f'{group}_{kind}_t{t}','family':'calibrated','group':group,'kind':kind,'threshold':t,'checkpoint':str(path),'sha256':sha},choices,[calibrated_select(r,m,group,t) for r in cal])
            print('CALIBRATION_FAMILY_COMPLETE',group,kind,flush=True)
    eligible=[r for r in results if r['train_oof']['delta_pp']>0 and r['calibration']['delta_pp']>0]
    selected=max(eligible,key=lambda r:r['calibration']['frame'][0]) if eligible else {'name':'baseline','family':'baseline'}
    controls=[]
    if selected['family']=='calibrated':
        controls=[r for r in results if r['family']=='calibrated' and r['kind']==selected['kind'] and r['threshold']==selected['threshold']]
    selection={'selected':selected,'matched_controls':controls,'eligible':[r['name'] for r in eligible],'rule':'positive take-OOF training and disjoint calibration mean gain; maximize calibration; no target fitting','target_reference':'Frozen O-MaMa consensus on identical original candidates, read only','pretrained_omama_used':False}
    (R/'folds.json').write_text(json.dumps(fold_receipts,indent=2));(R/'selection.json').write_text(json.dumps(selection,indent=2));print('FROZEN_SELECTION',selected['name'],flush=True)
if __name__=='__main__':main()
