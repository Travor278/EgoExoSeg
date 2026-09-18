"""Validate downloaded first-record witnesses and seed2 smoke receipts."""
from pathlib import Path
import hashlib,json
R=Path(__file__).parent
a=json.loads((R/'startup_validation_seed2.json').read_text(encoding='utf-8-sig'))
assert len(a)==4 and {x['rank'] for x in a}==set(range(4))
expected=json.loads((R/'manifest.json').read_text())['encoder_contract']
assert not (R/'gpu_preflight_seed2.txt').read_text().strip()
for x in a:
    assert x['finite_metrics'] and x['seed_namespace']=='candidate-quality-v2:'
    assert set(x['expert_masks'])=={'visual','anchor','fusion'}
    assert x['candidate_seed']==int.from_bytes(hashlib.sha256(('candidate-quality-v2:'+x['video_id']).encode()).digest()[:4],'little')%(2**31-1)
    assert all(x[k] in x['expert_masks'] for k in ('integrated_primary','integrated_local20','integrated_frozen_cycle'))
    assert 0 < x['gpu_peak_GiB'] < 80
    e=json.loads((R/f"runs/full2/rank{x['rank']}/encoder_receipt.json").read_text())
    assert all(e[k]==expected[k] for k in e)
    s=json.loads((R/f"runs/smoke2/rank{x['rank']}/receipt.json").read_text())
    assert s['coverage']=='exact' and s['capture_on_off_pairs']==s['pairs']
print('SEED2_STARTUP_PASSED',json.dumps([{k:x[k] for k in ('rank','finite_metrics','gpu_peak_GiB')} for x in a]))
