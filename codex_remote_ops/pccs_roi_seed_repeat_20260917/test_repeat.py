import hashlib,importlib.util
from pathlib import Path
from randomness import stable_pair_seed
R=Path(__file__).parent;P=R.parent/'pccs_roi_context_20260917'
spec=importlib.util.spec_from_file_location('seed1',P/'randomness.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
keys=['repeat-check-a','repeat-check-b','same-frame-object']
for k in keys:
    expected=int.from_bytes(hashlib.sha256(('candidate-quality-v2:'+k).encode()).digest()[:4],'little')%(2**31-1)
    assert stable_pair_seed(k)==expected and expected!=old.stable_pair_seed(k)
for n in ('roi_views.py','roi_evidence.py','roi_policy.py','native_bridge.py','dense_context.py','policies.py','encoder_contract.py'):assert (R/n).read_bytes()==(P/n).read_bytes()
print('REPEAT_NAMESPACE_AND_FROZEN_METHOD_PARITY_PASS')
