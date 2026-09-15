from pathlib import Path
import ast,hashlib,json,subprocess
from ast_identity import identity
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914/code';BASE='0e3bc33dec3e202ffbb86cec01038e60b18c162a'
tree=ast.parse((R/'remote_audit.py').read_text(encoding='utf8'));assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='paths' for t in n.targets));paths=ast.literal_eval(assignment.value)
out={'revision':BASE,'files':{},'file_ast_sha256':{},'metric_functions':{}}
for path in paths:
 data=subprocess.check_output(['git','-C',str(P),'show',BASE+':'+path]);out['files'][path]=hashlib.sha256(data).hexdigest();out['file_ast_sha256'][path]=identity(data.decode('utf8'))
data=subprocess.check_output(['git','-C',str(P),'show',BASE+':projects/v2sam_pccs/evaluation/pccs_metric.py']).decode('utf8');tree=ast.parse(data)
for name in ('compute_metrics','compute_boundary_f_measure','compute_location_score','_seg2bmap'):
 out['metric_functions'][name]=identity(data,name)
diary=Path('D:/Code/Work/EgoExoSeg/docs/V2SAM_EGO2EXO_EXPERIMENT_DIARY_AND_PLAN_20260820.md');out['diary_source']={'path':str(diary),'sha256':hashlib.sha256(diary.read_bytes()).hexdigest()}
(R/'upstream_identity.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('UPSTREAM_IDENTITIES_BUILT')
