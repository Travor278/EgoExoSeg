from pathlib import Path
from collections import defaultdict
import copy,hashlib,json
import numpy as np
from union_core import route
from fit_models import read,select_regressor
R=Path(__file__).parent
def main():
    import torch,joblib
    from torch.nn import functional as F
    from metric_adapter import Adapter
    frozen=json.loads((R/'selection.json').read_text());choice=frozen['selected'];rows=read('exo2exo');path=Path(choice['checkpoint']) if choice.get('checkpoint') else None
    if path:assert hashlib.sha256(path.read_bytes()).hexdigest()==choice['checkpoint_sha256']
    if choice['family']=='fixed_union':
        selected=[next(i for i,c in enumerate(r['candidates']) if c['id']==route(r['candidates'],r['fallback'],choice['parameter'])) if r['query_valid'] else r['fallback_index'] for r in rows]
    elif choice['family']=='quality_ranker':selected=select_regressor(joblib.load(path),rows,choice['parameter'])
    elif choice['family']=='omama_metric_adapter':
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=Adapter().to(device);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);model.eval();selected=[]
        with torch.no_grad():
            for r in rows:
                z=np.load(r['embedding_file']);cs=copy.deepcopy(r['candidates'])
                for mi,mode in enumerate(('canonical','native_interp')):
                    q=torch.tensor(z['q_'+mode],device=device);t=torch.tensor(z['t_'+mode],device=device);s=F.cosine_similarity(model(q)[None],model(t),dim=-1).tolist()
                    for c,v in zip(cs,s):c['scores'][mi]=v
                ident=route(cs,r['fallback'],choice['parameter']) if r['query_valid'] else r['fallback'];selected.append(next(i for i,c in enumerate(cs) if c['id']==ident))
    else:raise ValueError('No non-baseline method was preselected')
    # The selected method is fixed before reading labels for scoring.
    data=defaultdict(list);per_object=[]
    for r,i in zip(rows,selected):
        values={'original_pccs':r['baseline_pccs'],'original_consensus':r['baseline_metrics'],'selected_union':r['labels'][i]}
        data[r['video_id']].append({'take':r['take_id'],'metrics':values});per_object.append({'video_id':r['video_id'],'obj_id':r['obj_id'],'take_id':r['take_id'],'selected_hash':r['candidates'][i]['id'],'fallback_hash':r['fallback'],'metrics':values})
    results={m:{'frame':np.mean([np.mean([x['metrics'][m] for x in group],axis=0) for group in data.values()],axis=0).tolist(),'object':np.mean([r['metrics'][m] for r in per_object],axis=0).tolist()} for m in ('original_pccs','original_consensus','selected_union')}
    groups=defaultdict(list)
    for vid,g in data.items():groups[g[0]['take']].append(float(np.mean([x['metrics']['selected_union'][0]-x['metrics']['original_consensus'][0] for x in g])))
    sums=np.asarray([sum(v) for v in groups.values()]);counts=np.asarray([len(v) for v in groups.values()]);draw=np.random.default_rng(20260916).integers(0,len(groups),size=(10000,len(groups)));ci=np.quantile(sums[draw].sum(1)/counts[draw].sum(1),[.025,.975])*100
    expected=json.loads((R/'manifest.json').read_text())['phases']['exo2exo'];assert len(data)==expected['pairs'] and len(rows)==expected['objects']
    result={'coverage':'exact','pairs':len(data),'objects':len(rows),'takes':len(groups),'frozen_choice':choice,'methods':results,'primary_delta_vs_original_consensus_pp':float(100*sums.sum()/counts.sum()),'primary_95ci_pp':ci.tolist(),'original_masks_preserved':True,'test_fitting':False,'metric_order':['IoU','Dice','ContA','LocE']}
    (R/'exo2exo_results.json').write_text(json.dumps(result,indent=2));(R/'exo2exo_per_object.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in per_object));print('EXO2EXO_CONFIRMATION_COMPLETE',flush=True)
if __name__=='__main__':main()
