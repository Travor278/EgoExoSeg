from pathlib import Path
import hashlib,json,os,time
from huggingface_hub import HfApi
R=Path(__file__).parent;P=R.parent
expected={'vp_exo2ego_full.pth':'fd2e1b8f342c4468b45c962c61d4c914b9f6d71b6c6085014fb9a3ed2924b65f','fusion_exo2ego_full.pth':'8373b17180c198898fe9f9e39d09fe72991150ce3053b207b19e81986c60cab1'}
out={'started_at':time.time(),'expected_from_diary':expected,'local_candidates':[]}
def save():(R/'corrected_weights.json').write_text(json.dumps(out,indent=2))
try:
 info=HfApi().model_info('Travor278/V2-SAM',files_metadata=True)
 out['huggingface']={'revision':info.sha,'files':[]}
 for f in info.siblings:
  if f.rfilename in expected:
   lfs=f.lfs;oid=lfs.sha256 if hasattr(lfs,'sha256') else lfs.get('sha256') if isinstance(lfs,dict) else None
   out['huggingface']['files'].append({'name':f.rfilename,'size':f.size,'sha256':oid,'matches_diary':oid==expected[f.rfilename]})
except Exception as e:out['huggingface_error']=type(e).__name__+':'+str(e)[:250]
save()
roots=[P/'hf_publish_exo2ego_20260903',P/'V2SAM_Exo2Ego_20260825',P/'V2SAM_Bidirectional_Refresh_20260831',P/'V2SAM_NewMetrics_20260830']
for root in roots:
 if not root.exists():continue
 for current,dirs,files in os.walk(root):
  depth=len(Path(current).relative_to(root).parts)
  dirs[:]=[d for d in dirs if d not in ('.git','envs','site-packages','third_parts','mmengine','images','data','datasets','code','raw')]
  if depth>=4:dirs[:]=[]
  for name in files:
   if name not in expected:continue
   path=Path(current)/name;h=hashlib.sha256()
   with path.open('rb') as handle:
    for data in iter(lambda:handle.read(8*1024*1024),b''):h.update(data)
   out['local_candidates'].append({'path':str(path),'name':name,'bytes':path.stat().st_size,'sha256':h.hexdigest(),'matches_diary':h.hexdigest()==expected[name]});save()
out['state']='complete';save();print('CORRECTED_WEIGHTS_LOCATED',json.dumps(out),flush=True)
