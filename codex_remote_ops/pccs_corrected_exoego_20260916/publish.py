from pathlib import Path
import hashlib,json,re
R=Path(__file__).parent;O=R.parent;D=O/'publish_egoexoseg_20260916';receipts=[]
def copy(src):
 data=src.read_bytes();text=data.decode('utf-8-sig')
 for pattern in (r'gh[pousr]_[A-Za-z0-9]{20,}',r'hf_[A-Za-z0-9]{20,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',r'Bearer\s+[A-Za-z0-9_.-]{24,}'):
  if re.search(pattern,text):raise RuntimeError('Potential secret in '+str(src))
 dest=D/'codex_remote_ops'/src.relative_to(O);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);receipts.append({'file':dest.relative_to(D).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
for name in ('prepare.py','score.py','cache.py','summarize.py','run_stage.py','launch_h100.sh','PLAN.md','EXPLORATION_PLAN.md','manifest.json','job_receipt.json','status.json','finalize.py','publish.py','final_validation.json','resource_release.json'):
 if (R/name).exists():copy(R/name)
for pattern in ('runs/full/*.json','runs/full/*.md','runs/full/*.txt','runs/full/*.log','runs/smokeH100/*.json','runs/smokeH100/*.txt'):
 for p in R.glob(pattern):copy(p)
for name in ('FINAL_REPORT.md','render_master_report.py'):copy(O/'pccs_gain_full_20260915'/name)
(D/'codex_remote_ops'/R.name/'EXPORT_MANIFEST.json').write_text(json.dumps({'files':receipts},indent=2),encoding='utf8');print('PUBLISHED_LOCAL_EXPORT',len(receipts))
