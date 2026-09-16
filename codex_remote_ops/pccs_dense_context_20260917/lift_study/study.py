from pathlib import Path
import os,sys,json,hashlib,time
R=Path(__file__).parent;P=R.parent;sys.path[:0]=[str(P/'local_deps'),str(P)]
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,joblib,sklearn,scipy
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from data_utils import read,score
from policies import EXPERTS
from lift_policy import GATES,GROUPS,features,scores,choose,test
test()
assert not (R/'selection.json').exists(),'Do not overwrite frozen follow-up'
tr,cal=read('train'),read('calibration');assert {r['take_id'] for r in tr}.isdisjoint({r['take_id'] for r in cal});results=[];(R/'fitted').mkdir(exist_ok=True)
manifest={'declared_at':time.time(),'source_only':True,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'data_hashes':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for phase in ('train','calibration') for p in (P/'runs'/phase).glob('rank*/records.jsonl')},'target_loaded_before_selection':False,'fold_source_sha256':hashlib.sha256((P/'folds.json').read_bytes()).hexdigest()}
(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
def model(kind):
    if kind=='tree':return Pipeline([('model',HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=3,min_samples_leaf=25,l2_regularization=20,random_state=20260917))])
    return Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10 if kind=='ridge10' else 100))])
def fit(m,rows,group):
    counts=Counter(r['video_id'] for r in rows);x=[];y=[];w=[]
    for r in rows:
        options=[e for e in EXPERTS if e!=r['baseline'] and r['source_valid'] and r['evidence'][e]['quality_valid'] and r['mask_sha256'][e]!=r['mask_sha256'][r['baseline']]]
        for e in options:x.append(features(r,e,group));y.append(r['metrics'][e][0]-r['metrics'][r['baseline']][0]);w.append(1/(counts[r['video_id']]*len(options)))
    w=np.asarray(w);m.fit(np.asarray(x),np.asarray(y),model__sample_weight=w/w.mean());return m
frozen_folds=json.loads((P/'folds.json').read_text());folds=[]
for fold in frozen_folds:
    held=set(fold['held_takes']);fi=np.asarray([i for i,r in enumerate(tr) if r['take_id'] not in held]);vi=np.asarray([i for i,r in enumerate(tr) if r['take_id'] in held]);assert {tr[i]['take_id'] for i in fi}==set(fold['fit_takes']);folds.append((fi,vi))
for group in GROUPS:
    for kind in ('ridge10','ridge100','tree'):
        oof=[None]*len(tr)
        for fi,vi in folds:
            assert {tr[i]['take_id'] for i in fi}.isdisjoint({tr[i]['take_id'] for i in vi})
            m=fit(model(kind),[tr[i] for i in fi],group)
            for i in vi:oof[i]=scores(tr[i],m,group)
        m=fit(model(kind),tr,group);path=R/'fitted'/f'{group}_{kind}.joblib';joblib.dump(m,path);cp=[scores(r,m,group) for r in cal]
        for gate in GATES:
            for t in (.03,.05):
                cfg={'name':f'{gate}_{group}_{kind}_t{t}','gate':gate,'group':group,'kind':kind,'threshold':t,'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                cfg['train_oof']=score(tr,[choose(r,p,gate,t) for r,p in zip(tr,oof)]);cfg['calibration']=score(cal,[choose(r,p,gate,t) for r,p in zip(cal,cp)]);results.append(cfg)
        (R/'search.json').write_text(json.dumps({'candidates':results},indent=2));(R/'status.json').write_text(json.dumps({'state':'running','group':group,'kind':kind,'updated_at':time.time()}));print(group,kind,flush=True)
old={r['name']:r for r in json.loads((P/'search.json').read_text())['candidates'] if r['family']=='calibrated'};maxerr=0.
for r in results:
    if r['gate']=='dominance' and r['group']=='cycle_control':
        prior=old[r['name'].removeprefix('dominance_').replace('cycle_control','cycle')]
        for phase in ('train_oof','calibration'):maxerr=max(maxerr,abs(r[phase]['delta_pp']-prior[phase]['delta_pp']))
assert maxerr<1e-6,('Original protocol did not reproduce',maxerr)
eligible=[r for r in results if r['train_oof']['delta_pp']>0 and r['calibration']['delta_pp']>0];best=max(eligible,key=lambda r:r['calibration']['frame'][0]) if eligible else {'name':'baseline'}
controls=[r for r in results if r['kind']==best.get('kind') and r['threshold']==best.get('threshold') and r['gate']==best.get('gate')]
(R/'selection.json').write_text(json.dumps({'selected':best,'matched_controls':controls,'eligible':[r['name'] for r in eligible],'r1_reproduction_max_delta_error_pp':maxerr,'frozen_at':time.time(),'target_loaded_before_selection':False},indent=2));(R/'status.json').write_text(json.dumps({'state':'complete','selected':best['name']}));print('FROZEN',best['name'],best.get('train_oof',{}).get('delta_pp'),best.get('calibration',{}).get('delta_pp'),flush=True)
