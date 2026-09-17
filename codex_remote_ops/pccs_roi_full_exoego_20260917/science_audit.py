from pathlib import Path
import json,hashlib
import numpy as np
from data_utils import R,read
out={'scope':'Intervention strength and real-view reproduction; all objects kept including ineffective edits','phases':{}}
for phase in ('science_exo2exo','science_exo2ego'):
    if not all((R/'runs'/phase/f'rank{k}/receipt.json').exists() for k in range(4)):continue
    rows=read(phase);checks={}
    for mode in ('local15','local20'):
        for arm in ('far','rolled'):
            records=[r['metadata']['source_interventions'].get(mode+'/'+arm) for r in rows];valid=[v for v in records if v is not None]
            checks[mode+'/'+arm]={'objects':len(rows),'source_crop_missing':len(rows)-len(valid),'zero_pixel_edit_objects':sum(v.get('changed_pixel_fraction',0)==0 for v in valid),'mean_changed_pixel_fraction':float(np.mean([v.get('changed_pixel_fraction',0) for v in valid])) if valid else 0.,'mean_editable_fraction':float(np.mean([v['editable_fraction'] for v in valid])) if valid else 0.}
            if arm=='far':checks[mode+'/'+arm]['donor_overlaps_source_mask_objects']=sum(v.get('donor_foreground_fraction',0)>0 for v in valid)
    out['phases'][phase]={'real_max_feature_error':max(r['real_max_feature_error'] for r in rows),'arms':checks}
(R/'science_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
