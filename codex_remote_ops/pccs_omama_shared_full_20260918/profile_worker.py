"""Bounded 16-pair profiler in an isolated directory on the existing allocation."""
from pathlib import Path
import copy,json,os,pstats,shutil,subprocess,sys,yaml
R=Path(__file__).resolve().parent
if R.name=='perf':R=R.parent
D=R/'perf/profile_baseline';D.mkdir(parents=True,exist_ok=False)
m=json.loads((R/'manifest.json').read_text());(D/'code').symlink_to(R/'code',target_is_directory=True)
for n in m['code_sha256']:
    if not n.startswith('code/'):shutil.copy2(R/n,D/n)
shutil.copy2(R/'selection.json',D/'selection.json')
s=copy.deepcopy(m['phases']['full1']['shards'][0]);ann=json.loads(Path(s['annotation']).read_text());ann={k:ann[k] for k in list(ann)[:16]};ap=D/'annotation.json';ap.write_text(json.dumps(ann));cfg=yaml.safe_load(Path(s['config']).read_text());cfg['data']['annotations']['exo2ego']=str(ap);cp=D/'runtime.yaml';cp.write_text(yaml.safe_dump(cfg));s.update(annotation=str(ap),config=str(cp),pairs=len(ann),objects=sum(len(a['objects']) for a in ann.values()));m['phases']={'profile':{'pairs':s['pairs'],'objects':s['objects'],'shards':[s]}};(D/'manifest.json').write_text(json.dumps(m))
C=R.parent/'pccs_corrected_exoego_20260916';P=R.parent/'context_pccs_20260914'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(D)+':'+str(C/'python_deps')+':'+str(D/'code/mmengine')+':'+str(D/'code'),'LD_LIBRARY_PATH':str(P/'runtime_lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONNOUSERSITE':'1','PYTHONUNBUFFERED':'1'}
with (D/'worker.log').open('w') as f:
    result=subprocess.run([sys.executable,'-m','cProfile','-o',str(D/'cpu.prof'),str(D/'worker.py'),'--phase','profile','--rank','0'],env=env,cwd=D,stdout=f,stderr=subprocess.STDOUT,timeout=600)
stats=pstats.Stats(str(D/'cpu.prof'));rows=[]
for (file,line,name),(cc,nc,tt,ct,callers) in stats.stats.items():
    if any(x in file for x in ('pccs','roi_','dense_context','region_pooling')) or name in ('synchronize','hexdigest','read','write'):
        rows.append({'file':file,'line':line,'function':name,'calls':nc,'self_seconds':tt,'cumulative_seconds':ct})
rows.sort(key=lambda x:x['cumulative_seconds'],reverse=True)
(R/'perf/function_profile.json').write_text(json.dumps({'returncode':result.returncode,'pairs':len(ann),'objects':s['objects'],'note':'cProfile includes model initialization and runs concurrently with main job on GPU0; not a clean throughput benchmark','functions':rows[:100]},indent=2))
print('PROFILE_DONE',result.returncode,flush=True)
