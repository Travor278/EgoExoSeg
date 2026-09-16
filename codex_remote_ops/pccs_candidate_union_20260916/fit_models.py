"""Stages3/4: TRAIN-only quality regression and lightweight O-MaMa metric adaptation."""
from pathlib import Path
from collections import defaultdict
import copy,json,hashlib,random
import numpy as np
from union_core import feature_vector,route
R=Path(__file__).parent
def read(phase):
    p=R/'datasets'/phase/'records.jsonl';rec=json.loads((p.parent/'receipt.json').read_text());assert rec['records_sha256']==hashlib.sha256(p.read_bytes()).hexdigest();return [json.loads(s) for s in p.read_text().splitlines()]
def score(rows,selected):
    pairs=defaultdict(list);changed=0;improved=0;harmed=0
    for r,index in zip(rows,selected):
        value=r['labels'][index][0];base=r['baseline_metrics'][0];pairs[r['video_id']].append(value);changed+=index!=r['fallback_index'];improved+=value>base+1e-8;harmed+=value<base-1e-8
    return {'frame_iou':float(np.mean([np.mean(v) for v in pairs.values()])),'changed':changed,'improved':improved,'harmed':harmed,'pairs':len(pairs),'objects':len(rows)}
def select_regressor(model,rows,threshold):
    selected=[]
    for r in rows:
        allowed=[i for i,c in enumerate(r['candidates']) if i!=r['fallback_index'] and not c['empty'] and r['query_valid']]
        if not allowed:selected.append(r['fallback_index']);continue
        x=np.asarray([feature_vector(r['candidates'],r['fallback'],r['candidates'][i]['id']) for i in allowed]);pred=model.predict(x);assert np.isfinite(pred).all();k=int(np.argmax(pred));selected.append(allowed[k] if pred[k]>threshold else r['fallback_index'])
    return selected
def main():
    import joblib,torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    screen,cal=read('screen'),read('calibration');base={p:score(rows,[r['fallback_index'] for r in rows]) for p,rows in [('screen',screen),('calibration',cal)]}
    results=[];models_dir=R/'fitted';models_dir.mkdir(exist_ok=True)
    def record(name,family,parameter,pred_screen,pred_cal,checkpoint=None):
        values={'name':name,'family':family,'parameter':parameter,'screen':score(screen,pred_screen),'calibration':score(cal,pred_cal),'checkpoint':str(checkpoint) if checkpoint else None}
        for phase in base:values[phase]['delta_pp']=100*(values[phase]['frame_iou']-base[phase]['frame_iou'])
        if checkpoint:values['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        results.append(values);(R/'model_search.json').write_text(json.dumps({'baseline':base,'candidates':results},indent=2));return values
    # Fixed union reranking stage; the first comparison preserves the originally fixed0.05.
    for margin in (.05,.10):
        chosen=[]
        for rows in (screen,cal):chosen.append([next(i for i,c in enumerate(r['candidates']) if c['id']==route(r['candidates'],r['fallback'],margin)) if r['query_valid'] else r['fallback_index'] for r in rows])
        record('union_margin_'+str(margin),'fixed_union',margin,*chosen)
    # Stage3 explicit prediction-only feature matrix; label differences are separate.
    X=[];y=[];w=[];counts=defaultdict(int)
    for r in screen:counts[r['video_id']]+=1
    for r in screen:
        valid=[i for i,c in enumerate(r['candidates']) if i!=r['fallback_index'] and not c['empty'] and r['query_valid']]
        for i in valid:X.append(feature_vector(r['candidates'],r['fallback'],r['candidates'][i]['id']));y.append(r['labels'][i][0]-r['baseline_metrics'][0]);w.append(1/(counts[r['video_id']]*len(valid)))
    X=np.asarray(X);y=np.asarray(y);w=np.asarray(w);w/=w.mean();assert X.shape[1]==29 and np.isfinite(X).all()
    candidates={'ridge10':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10))]),'ridge100':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=100))]),'tree':Pipeline([('model',HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=3,min_samples_leaf=25,l2_regularization=10,random_state=42))])}
    for name,model in candidates.items():
        model.fit(X,y,model__sample_weight=w);path=models_dir/(name+'.joblib');joblib.dump(model,path)
        for threshold in (.01,.03,.05):record(name+'_t'+str(threshold),'quality_ranker',threshold,select_regressor(model,screen,threshold),select_regressor(model,cal,threshold),path)
    print('STAGE3_QUALITY_RANKERS_COMPLETE',flush=True)
    # Stage4 learns a shared rank8 residual metric projection after frozen O-MaMa.
    from metric_adapter import Adapter
    torch.manual_seed(20260916);random.seed(20260916);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');adapter=Adapter().to(device)
    def tensors(rows):
        out=[]
        for r in rows:
            z=np.load(r['embedding_file']);out.append({mode:(torch.tensor(z['q_'+mode],device=device),torch.tensor(z['t_'+mode],device=device)) for mode in ('canonical','native_interp')})
        return out
    st,ct=tensors(screen),tensors(cal)
    with torch.no_grad():
        q,t=st[0]['canonical'];original=F.cosine_similarity(q[None],t,dim=-1);zero=F.cosine_similarity(adapter(q)[None],adapter(t),dim=-1);torch.testing.assert_close(original,zero,atol=1e-6,rtol=1e-6)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-3,weight_decay=.01)
    def adapted_choices(rows,tensor_rows,margin):
        selected=[]
        with torch.no_grad():
            for r,ts in zip(rows,tensor_rows):
                cs=copy.deepcopy(r['candidates'])
                for mi,mode in enumerate(('canonical','native_interp')):
                    q,t=ts[mode];values=F.cosine_similarity(adapter(q)[None],adapter(t),dim=-1).cpu().tolist()
                    for c,v in zip(cs,values):c['scores'][mi]=v
                ident=route(cs,r['fallback'],margin) if r['query_valid'] else r['fallback'];selected.append(next(i for i,c in enumerate(cs) if c['id']==ident))
        return selected
    train_ids=[i for i,r in enumerate(screen) if r['query_valid'] and sum(not c['empty'] for c in r['candidates'])>=2]
    losses=[]
    for epoch in range(1,41):
        random.shuffle(train_ids);adapter.train()
        for start in range(0,len(train_ids),16):
            optimizer.zero_grad();terms=[]
            for i in train_ids[start:start+16]:
                r=screen[i];valid=torch.tensor([not c['empty'] for c in r['candidates']],device=device);teacher=torch.softmax(torch.tensor([v[0] for v in r['labels']],device=device)[valid]/.1,dim=0)
                for mode in ('canonical','native_interp'):
                    q,t=st[i][mode];s=F.cosine_similarity(adapter(q)[None],adapter(t),dim=-1)[valid];terms.append(-(teacher*F.log_softmax(s/.07,dim=0)).sum()/counts[r['video_id']])
            loss=torch.stack(terms).mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        if epoch in (5,15,40):
            adapter.eval();path=models_dir/f'omama_adapter_e{epoch}.pt';torch.save(adapter.state_dict(),path)
            for margin in (.02,.05,.10):record(f'omama_adapter_e{epoch}_m{margin}','omama_metric_adapter',margin,adapted_choices(screen,st,margin),adapted_choices(cal,ct,margin),path)
            print('STAGE4_CHECKPOINT',epoch,'finite_loss',losses[-1],flush=True)
    eligible=[r for r in results if r['screen']['delta_pp']>0 and r['calibration']['delta_pp']>0]
    best=max(eligible,key=lambda r:r['calibration']['frame_iou']) if eligible else {'name':'baseline','family':'baseline','parameter':None,'checkpoint':None}
    selection={'selected':best,'eligible_names':[r['name'] for r in eligible],'baseline':base,'rule':'positive final frameIoU on screen and calibration; maximize calibration; no target labels used','adapter_rank':8,'adapter_epochs':[5,15,40],'V2SAM_prompt_projection_finetuned':False,'test_fitting':False}
    (R/'selection.json').write_text(json.dumps(selection,indent=2));(R/'adapter_training_receipt.json').write_text(json.dumps({'steps':len(losses),'losses_finite':bool(np.isfinite(losses).all()),'last_loss':losses[-1],'zero_adapter_parity':'passed'},indent=2));print('SELECTION_FROZEN',best['name'],flush=True)
if __name__=='__main__':main()
