from pathlib import Path
from collections import defaultdict
import hashlib,json,sys
import numpy as np,joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from gain_features import vector
R=Path(__file__).parent;G=R/'gate_training'
records=[json.loads(s) for s in (G/'training_records.jsonl').read_text().splitlines()]
plan=json.loads((G/'plan.json').read_text());fit=[r for r in records if r['split']=='fit'];cal=[r for r in records if r['split']=='calibration']
assert {r['video_id'] for r in fit}.isdisjoint({r['video_id'] for r in cal})
X=[];y=[];weights=[];columns=None
for r in fit:
 allowed=[i for i in range(1,len(r['inputs']['names'])) if r['query_nonempty'] and r['nonempty'][i]]
 for i in allowed:
  x,cols=vector(r['inputs'],i);columns=cols if columns is None else columns;assert cols==columns
  X.append(x);y.append(r['labels'][i][0]-r['labels'][0][0]);weights.append(1./(r['inputs']['values']['pair_num_objects']*len(allowed)))
X=np.asarray(X);y=np.asarray(y);weights=np.asarray(weights);weights/=weights.mean()
assert len(X)>100 and np.isfinite(y).all()
# Regression labels never enter vector(): its only input is the allowlisted prediction dictionary.
models={f'ridge_{alpha:g}':Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('regressor',Ridge(alpha=alpha))]) for alpha in plan['models']['ridge']}
models['hist_gradient_boosting']=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('regressor',HistGradientBoostingRegressor(random_state=42,**plan['models']['hist_gradient_boosting']))])
def summary(selected):
 groups=defaultdict(list);changes=0
 for r,i in zip(cal,selected):groups[r['video_id']].append(r['labels'][i][0]);changes+=i!=0
 return {'frame_iou':float(np.mean([np.mean(v) for v in groups.values()])),'changed_objects':changes,'objects':len(cal),'pairs':len(groups)}
base=summary([0]*len(cal));rows=[];best={'model':'baseline','threshold':None,**base};best_model=None
for name,model in models.items():
 model.fit(X,y,regressor__sample_weight=weights)
 predictions=[]
 for r in cal:
  allowed=[i for i in range(1,len(r['inputs']['names'])) if r['query_nonempty'] and r['nonempty'][i]]
  preds=model.predict(np.asarray([vector(r['inputs'],i)[0] for i in allowed])) if allowed else []
  assert np.isfinite(preds).all();predictions.append(dict(zip(allowed,preds)))
 for threshold in plan['thresholds']:
  selected=[]
  for preds in predictions:
   i=max(preds,key=preds.get) if preds else 0;selected.append(i if i and preds[i]>threshold else 0)
  result={'model':name,'threshold':threshold,**summary(selected)};result['delta_pp']=100*(result['frame_iou']-base['frame_iou']);rows.append(result)
  if result['frame_iou']>best['frame_iou']+1e-8 or (abs(result['frame_iou']-best['frame_iou'])<1e-8 and result['changed_objects']<best['changed_objects']):best=result;best_model=model
 if best_model is model:joblib.dump(model,G/'selected_gate.joblib')
out={'state':'complete','fit_pairs':len({r['video_id'] for r in fit}),'calibration_pairs':len({r['video_id'] for r in cal}),
     'regression_examples':len(X),'features':columns,'baseline':base,'calibration_search':rows,'selected':best,
     'calibration_is_from_official_train':True,'upstream_experts_saw_train':True,'official_test_not_used':True,
     'next_step':'Freeze selected gate and evaluate on unused test takes; calibration gains are not test gains.'}
if best_model is not None:
 out['checkpoint_sha256']=hashlib.sha256((G/'selected_gate.joblib').read_bytes()).hexdigest()
 out['checkpoint']=str(G/'selected_gate.joblib')
(G/'fit_results.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({k:out[k] for k in ('fit_pairs','calibration_pairs','regression_examples','baseline','selected','next_step')},indent=2))
