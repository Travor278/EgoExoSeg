from pathlib import Path
import json,time
R=Path(__file__).parent;p=R/'manifest.json';a=json.loads(p.read_text())
assert not (R/'runs/smokeH100').exists() and not (R/'runs/full').exists()
a['stages']['smokeH100']=dict(a['stages']['smoke4090'])
a['preflight_4090']={'state':'blocked_by_existing_gpu_process','candidate_generation_started':False,'receipt':'runs/smoke4090/preflight_gpu_processes.txt','time':time.time()}
p.write_text(json.dumps(a,indent=2));print('H100_READY')
