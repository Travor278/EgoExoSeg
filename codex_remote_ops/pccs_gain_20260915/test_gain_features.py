from pathlib import Path
import copy,json,numpy as np
from gain_features import prediction_inputs,vector
R=Path(__file__).parent
raw=[json.loads(s) for s in (R/'runs/confirmation/exo2ego/per_object.jsonl').read_text().splitlines()]
scored=[json.loads(s) for s in (R/'confirmation_per_object.jsonl').read_text().splitlines()]
byid={(r['video_id'],r['obj_id']):r for r in raw};checked=0
for s in scored[:64]:
 row=byid[s['video_id'],s['obj_id']];original=prediction_inputs(row,s['scores'],s['candidate_names'])
 corrupted=copy.deepcopy(row)
 for k in corrupted:
  if k in ('IoU','Dice','shape_acc','location_score') or k.startswith(('iou_','dice_','shape_acc_','location_score_','oracle_')):corrupted[k]=987654.321
 after=prediction_inputs(corrupted,s['scores'],s['candidate_names']);assert original==after
 for i in range(1,len(s['candidate_names'])):
  x,names=vector(original,i);y,n2=vector(after,i);assert names==n2 and len(x)==65
  np.testing.assert_array_equal(x,y);assert np.isfinite(x).sum()>40;checked+=1
print(f'PASS: {checked} real-candidate feature vectors invariant to target/oracle metric corruption;65 allowlisted features')
