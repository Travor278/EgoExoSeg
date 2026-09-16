"""Describe actual zero-imputation behavior; no policy or target selection changes."""
import json
from data_utils import R,read,score
out={'scope':'Implementation audit: absent ROI is zero-imputed plus validity; original measurements may still trigger switching. Forced fallback is a post hoc diagnostic only.','phases':{}}
for phase in ('exo2exo','exo2ego'):
    p=R/(phase+'_results.json')
    if not p.exists():continue
    rows=read(phase);preds={(r['video_id'],r['obj_id']):r['choices'] for r in map(json.loads,(R/(phase+'_predictions.jsonl')).read_text().splitlines())};result={}
    for name,mode in [('primary','local15'),('local20_matched','local20')]:
        original=[preds[(r['video_id'],r['obj_id'])][name] for r in rows];missing=[i for i,(r,e) in enumerate(zip(rows,original)) if e!=r['baseline'] and not r['evidence'][e][mode+'_valid']]
        forced=[r['baseline'] if i in missing else e for i,(r,e) in enumerate(zip(rows,original))];a=score(rows,original);b=score(rows,forced)
        result[name]={'changed_to_missing_roi_objects':len(missing),'all_candidate_roi_missing_objects':sum(not any(v[mode+'_valid'] for v in r['evidence'].values()) for r in rows),'forced_fallback_iou':b['frame'][0],'original_minus_forced_pp':(a['frame'][0]-b['frame'][0])*100}
    out['phases'][phase]=result
(R/'missing_view_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
