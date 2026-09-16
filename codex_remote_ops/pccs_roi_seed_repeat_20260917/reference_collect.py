import json,hashlib
from data_utils import R,read
rows={(r['video_id'],r['obj_id']):r for r in read('exo2exo')};seen=set();updates=[];witnesses=[];reused=0
for rank in range(4):
    p=R/'runs/reference'/f'rank{rank}'/'references.jsonl';receipt=json.loads((p.parent/'receipt.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['records_sha256'];witnesses.append(receipt['witness'])
    for r in map(json.loads,p.read_text().splitlines()):
        key=(r['video_id'],r['obj_id']);assert key not in seen;seen.add(key);row=rows[key];assert r['mask_sha256']==row['mask_sha256'] and r['metrics']==row['metrics'][r['expert']]
        if r['reused_identical_bank']:assert row['frozen_omama_reference']==r['metrics'] and row['frozen_omama_expert']==r['expert'];reused+=1
        else:updates.append(r)
assert seen==set(rows)
(R/'reference_updates.json').write_text(json.dumps({'updates':updates,'count':len(updates),'reused_identical_bank':reused,'witnesses':witnesses,'same_current_bank':True,'external_reference_only':True},indent=2));print('REFERENCE_VERIFIED',len(updates),reused)
