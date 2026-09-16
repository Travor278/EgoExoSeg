"""Independent numeric verification; no images or target-driven re-selection."""
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib
import numpy as np
R=Path(__file__).parent
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
result=load(R/'exo2exo_results.json');selected=rows(R/'exo2exo_per_object.jsonl');ds=rows(R/'datasets/exo2exo/records.jsonl');receipt=load(R/'datasets/exo2exo/receipt.json')
assert receipt['records_sha256']==hashlib.sha256((R/'datasets/exo2exo/records.jsonl').read_bytes()).hexdigest()
ident=lambda r:(r['video_id'],r['obj_id'])
d={ident(r):r for r in ds};pred={ident(r):r for r in selected};assert len(d)==len(ds)==len(pred)==1094
arms={};anchor_checks={}
for arm in ('baseline','reliable_points','residual005','residual010','residual025'):
    p=R/'runs/exo2exo'/arm/'per_object.jsonl';rec=load(p.parent/'receipt.json');assert rec['coverage']=='exact' and rec['records_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    a={ident(r):r for r in rows(p)};assert a.keys()==d.keys();arms[arm]=a
base=arms['baseline']
for arm in ('residual005','residual010','residual025'):
    mismatch=sum(base[k]['mask_sha256']['anchor']!=arms[arm][k]['mask_sha256']['anchor'] for k in base);assert mismatch==0;anchor_checks[arm]=mismatch
changed=improved=harmed=0;provenance=Counter();old_oracle=[];new_oracle=[];take_deltas=defaultdict(list)
for k,r in d.items():
    p=pred[k];candidates={c['id']:i for i,c in enumerate(r['candidates'])};assert set(base[k]['mask_sha256'].values())<=candidates.keys()
    assert p['fallback_hash']==base[k]['mask_sha256'][base[k]['consensus_expert']]
    index=candidates[p['selected_hash']];assert np.allclose(p['metrics']['selected_union'],r['labels'][index],atol=0,rtol=0)
    assert p['metrics']['original_consensus']==base[k]['metrics']['consensus'] and p['metrics']['original_pccs']==base[k]['metrics']['pccs']
    a=p['metrics']['selected_union'][0];b=p['metrics']['original_consensus'][0];changed+=p['selected_hash']!=p['fallback_hash'];improved+=a>b+1e-8;harmed+=a<b-1e-8
    if p['selected_hash']!=p['fallback_hash']:provenance.update(r['candidates'][index]['provenance'])
    old_oracle.append(base[k]['metrics']['oracle'][0]);new_oracle.append(max(v[0] for v in r['labels']));assert new_oracle[-1]+1e-7>=old_oracle[-1]
    take_deltas[r['take_id']].append(a-b)
errors=[]
for method in result['methods']:
    bypair=defaultdict(list)
    for p in selected:bypair[p['video_id']].append(p['metrics'][method])
    frame=np.mean([np.mean(v,axis=0) for v in bypair.values()],axis=0);err=float(np.max(np.abs(frame-result['methods'][method]['frame'])));assert err<1e-12;errors.append(err)
sums=np.array([sum(v) for v in take_deltas.values()]);counts=np.array([len(v) for v in take_deltas.values()]);draw=np.random.default_rng(20260916).integers(0,len(counts),size=(10000,len(counts)));ci=np.quantile(sums[draw].sum(1)/counts[draw].sum(1),[.025,.975])*100
np.testing.assert_allclose(ci,result['primary_95ci_pp'],atol=1e-12,rtol=0)
frozen=load(R/'selection.json');assert frozen['selected']==result['frozen_choice'];assert hashlib.sha256((R/'fitted/ridge10.joblib').read_bytes()).hexdigest()==frozen['selected']['checkpoint_sha256']
audit={'coverage':'exact','pairs':len({p['video_id'] for p in selected}),'objects':len(selected),'takes':len(take_deltas),'anchor_mismatches':anchor_checks,'all_original_candidates_preserved':True,'changed':changed,'improved':improved,'harmed':harmed,'selected_provenance_multilabel':dict(provenance),'baseline_oracle_iou':float(np.mean(old_oracle)*100),'union_oracle_iou':float(np.mean(new_oracle)*100),'max_metric_recompute_error':max(errors),'bootstrap_recomputed':True,'original_pccs_zero_iou_objects':sum(p['metrics']['original_pccs'][0]==0 for p in selected),'frozen_checkpoint_sha_verified':True,'max_embedding_score_error':receipt['max_score_replay_error']}
(R/'final_validation.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
