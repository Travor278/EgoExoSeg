"""Post hoc, fixed-cohort description; GT oracle never used for policy selection."""
from collections import defaultdict
from pathlib import Path
import gzip,json,sys
import numpy as np

R=Path(__file__).parent
seed=int(sys.argv[1]) if len(sys.argv)>1 else 1
with gzip.open(R/f'full{seed}_compact.jsonl.gz','rt') as f:
    rows=list(map(json.loads,f))
old=R.parent/'pccs_roi_context_20260917/exo2ego_predictions.jsonl'
ids={json.loads(s)['video_id'] for s in old.read_text().splitlines()}
assert len(ids)==512 and ids<={r['video_id'] for r in rows}

def describe(rs):
    pairs=defaultdict(list)
    for r in rs:
        b=r['metrics'][r['baseline']][0]
        p=r['metrics'][r['choices']['primary']][0]
        v=r['metrics'][r['choices']['local20_matched']][0]
        oracle=max(x[0] for x in r['metrics'].values())
        pairs[r['video_id']].append([b,p,v,oracle,max(0,v-b),min(0,v-b)])
    means=np.mean([np.mean(v,axis=0) for v in pairs.values()],axis=0)*100
    return dict(pairs=len(pairs),objects=len(rs),takes=len({r['take_id'] for r in rs}),
                baseline=means[0],primary=means[1],roi2=means[2],oracle=means[3],
                primary_gain_pp=means[1]-means[0],roi2_gain_pp=means[2]-means[0],
                headroom_pp=means[3]-means[0],roi2_recovery_fraction=(means[2]-means[0])/(means[3]-means[0]),
                roi2_positive_pp=means[4],roi2_negative_pp=means[5])

result={'seed':seed,'scope':'Same full-run candidates, pair-equal aggregation; fixed prior 512 pair membership, not prior-run scores.',
        'limitation':'Post hoc descriptive oracle and subgroup analysis; no fitting/threshold selection; subgroup difference does not identify causal hardness.',
        'all':describe(rows),'prior512_membership':describe([r for r in rows if r['video_id'] in ids]),
        'remaining':describe([r for r in rows if r['video_id'] not in ids])}
assert result['remaining']['pairs']==46003
(R/f'full{seed}_diagnostic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
