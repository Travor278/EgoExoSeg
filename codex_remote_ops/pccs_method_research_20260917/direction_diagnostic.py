"""Ground-truth oracle for post hoc diagnostics only, never inference or tuning."""
from pathlib import Path
from collections import defaultdict
import json
R=Path(__file__).parent;P=R.parent/'pccs_roi_context_20260917';out={}
for phase in ('exo2exo','exo2ego'):
    pred={(r['video_id'],r['obj_id']):r['choices']['local20_matched'] for r in map(json.loads,(P/(phase+'_predictions.jsonl')).read_text().splitlines())};rows=[json.loads(s) for p in (P/'runs'/phase).glob('rank*/records.jsonl') for s in p.read_text().splitlines()];pairs=defaultdict(list);switch=good=bad=better=0
    for r in rows:
        b=r['metrics'][r['baseline']][0];e=pred[(r['video_id'],r['obj_id'])];v=r['metrics'][e][0];o=max(x[0] for x in r['metrics'].values());pairs[r['video_id']].append((b,v,o,max(0,v-b),min(0,v-b)));switch+=e!=r['baseline'];good+=v>b+1e-8;bad+=v<b-1e-8;better+=o>b+1e-8
    means=[sum(sum(x[i] for x in rs)/len(rs) for rs in pairs.values())/len(pairs)*100 for i in range(5)]
    out[phase]={'pairs':len(pairs),'objects':len(rows),'baseline':means[0],'roi2':means[1],'oracle':means[2],'headroom':means[2]-means[0],'positive_contribution':means[3],'negative_contribution':means[4],'switches':switch,'improved':good,'harmed':bad,'objects_with_better_candidate':better}
(R/'direction_diagnostic.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
