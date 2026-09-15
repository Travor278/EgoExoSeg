from pathlib import Path
import json
from summarize import aggregate
R=Path(__file__).parent;H=R.parent/'pccs_gain_20260915/gate_training/holdout'
rows=[json.loads(s) for s in (H/'per_object.jsonl').read_text().splitlines()]
ref=json.loads((H/'results.json').read_text())['methods']
sharded=[r for rank in range(4) for r in rows[rank::4]]
result,_,seen=aggregate(sharded);assert len(seen)==965;json.dumps(result,allow_nan=False)
for m in ref:
 for level in ('frame','object'):
  for k,v in ref[m][level].items():assert abs(result[m][level][k]-v)<1e-12,(m,level,k)
 for k in ('changed','improved','harmed'):assert result[m][k]==ref[m][k]
print('PASS: four-way aggregation matches all prior965-object metrics/counts; JSON-safe')
