"""Read-only audit of completed experiments; never launches inference or fits parameters."""
from pathlib import Path
from collections import defaultdict,Counter
import ast,hashlib,json,math,os,subprocess,time
from ast_identity import identity as code_identity
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914';F=R.parent/'pccs_gain_full_20260915';X=R.parent/'pccs_exoexo_transfer_20260916';BASE='0e3bc33dec3e202ffbb86cec01038e60b18c162a'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):(R/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def astfn(src,name):
 tree=ast.parse(src)
 return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name),include_attributes=False)
def audit_records(root):
 plan=json.loads((root/'manifest.json').read_text());spec=plan['stages']['full'];anno=json.loads(Path(spec['annotation']).read_text());report=json.loads((root/'runs/full/results.json').read_text())
 expected={(str(v),str(o)) for v,rec in anno.items() for o in rec['objects']};expected_perpair={str(k):len(v['objects']) for k,v in anno.items()}
 def scan(paths,candidate=False):
  seen=set();pairs=defaultdict(list);zero=Counter();totals=defaultdict(list)
  for path in paths:
   with path.open() as handle:
    for line in handle:
     row=json.loads(line);ident=(row['video_id'],row['obj_id']);assert ident not in seen;seen.add(ident)
     ms={'baseline':[row[k] for k in ('IoU','Dice','shape_acc','location_score')]} if candidate else row['metrics']
     pairs[row['video_id']].append(ms)
     for m,vals in ms.items():
      assert len(vals)==4 and all(math.isfinite(float(v)) for v in vals)
      totals[m].append(vals);zero[m]+=vals[0]==0
  assert seen==expected
  assert {k:len(v) for k,v in pairs.items()}==expected_perpair
  result={}
  for m,values in totals.items():
   obj=[math.fsum(v[j] for v in values)/len(values) for j in range(4)]
   frame=[math.fsum(math.fsum(item[m][j] for item in p)/len(p) for p in pairs.values())/len(pairs) for j in range(4)]
   expected_report=report['methods'][m]
   error=max(abs(a-b) for a,b in zip(obj+frame,list(expected_report['object'].values())+list(expected_report['frame'].values())))
   assert error<1e-10,(root,m,error)
   result[m]={'object':obj,'frame':frame,'zero_iou_objects_included':zero[m],'max_report_error':error}
  return {'pairs':len(pairs),'objects':len(seen),'nonempty_pairs':all(expected_perpair.values()),'source_files':[{'path':str(p),'sha256':sha(p)} for p in paths],'metrics':result}
 before=scan([root/'runs/full/candidates/exo2ego/per_object.jsonl'],True)
 after=scan(sorted((root/'runs/full/scored').glob('rank[0-9].jsonl')))
 assert before['metrics']['baseline']==after['metrics']['baseline']
 return {'annotation_sha256':sha(spec['annotation']),'expected_annotation_sha256':spec['annotation_sha256'],'aggregation':'independent math.fsum; objects within pair then equal pair mean; no positive-IoU filter','raw_candidates':before,'scored':after}
status={'state':'running','started_at':time.time()};save('status.json',status)
try:
 out={'upstream_revision':BASE,'sources':{},'records':{},'runtime':{}}
 paths=['projects/v2sam_pccs/models/pccs_fusion_expert.py','projects/v2sam_pccs/models/pccs_visual_anchor_expert.py','projects/v2sam_pccs/models/pccs_components/region_sampler.py','projects/v2sam_pccs/models/pccs_components/visual_prompt_matcher.py','projects/v2sam_pccs/models/pccs_components/forward_correspondence.py','projects/v2sam_pccs/models/pccs_components/backward_correspondence.py','projects/v2sam_pccs/models/sam2_train.py','projects/v2sam_pccs/models/pccs_components/fusion_encoder.py','projects/v2sam_pccs/datasets/adapter.py','projects/v2sam_fusion/datasets/ReObjectRelator_Dataset.py','projects/v2sam_fusion/datasets/collect_fns.py']
 identity=json.loads((R/'upstream_identity.json').read_text());assert identity['revision']==BASE
 for path in paths:
  out['sources'][path]={'upstream_sha256':identity['files'][path],'full_sha256':sha(F/'code'/path),'exoexo_sha256':sha(X/'code'/path)}
  out['sources'][path]['ast_sha256']={root.name:code_identity((root/'code'/path).read_text()) for root in (F,X)}
  out['sources'][path]['upstream_ast_sha256']=identity['file_ast_sha256'][path]
  save('read_only_audit.json',out)
  assert all(v==identity['file_ast_sha256'][path] for v in out['sources'][path]['ast_sha256'].values()),path
 metric='projects/v2sam_pccs/evaluation/pccs_metric.py'
 out['metric_function_ast_equal']={name:all(identity['metric_functions'][name]==code_identity((root/'code'/metric).read_text(),name) for root in (F,X)) for name in ('compute_metrics','compute_boundary_f_measure','compute_location_score','_seg2bmap')}
 assert all(out['metric_function_ast_equal'].values())
 out['runtime']['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True)
 out['runtime']['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True)
 for root in (F,X):
  out['records'][root.name]=audit_records(root);save('read_only_audit.json',out);print('RECORD_AUDIT_OK',root.name,flush=True)
 import yaml
 cfg=yaml.safe_load((F/'full_runtime.yaml').read_text());weights={'visual':cfg['models']['experts']['exo2ego']['visual'],'fusion':cfg['models']['experts']['exo2ego']['fusion'],'dinov3':cfg['models']['dinov3']['weights']}
 out['weights']={name:{'path':p,'bytes':Path(p).stat().st_size,'sha256':sha(p)} for name,p in weights.items()}
 out['runtime']['python']=subprocess.check_output([os.sys.executable,'--version'],text=True).strip()
 save('read_only_audit.json',out);save('status.json',{'state':'complete','updated_at':time.time()});print('READ_ONLY_AUDIT_COMPLETE',flush=True)
except Exception as e:save('status.json',{'state':'failed','error':repr(e),'updated_at':time.time()});raise
