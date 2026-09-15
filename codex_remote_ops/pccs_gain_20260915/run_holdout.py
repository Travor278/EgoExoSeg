from pathlib import Path
import json,os,re,subprocess,sys,time
R=Path(__file__).parent;G=R/'gate_training';H=G/'holdout';P=R.parent/'context_pccs_20260914'
def status(**x):
 p=H/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({**x,'updated_at':time.time(),'pid':os.getpid()},indent=2));tmp.replace(p)
env=dict(os.environ,PATH=str(P/'runtime_env/bin')+':'+os.environ['PATH'],PYTHONPATH=str(G/'python_deps')+':'+str(R/'code/mmengine')+':'+str(R/'code'),LD_LIBRARY_PATH=str(P/'runtime_lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),PCCS_CONTEXT='0',PCCS_GAIN_BANK=str(H/'candidate_bank'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
try:
 if (H/'runs').exists():raise RuntimeError('Fresh holdout candidate bank required')
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);(H/'gpu_preflight.txt').write_text(gpu)
 if gpu.strip():raise RuntimeError('GPU busy; no interference')
 with (H/'candidate_generation.log').open('w') as log:
  cmd=[sys.executable,str(R/'code/tools/pccs_eval.py'),'--config',str(H/'runtime.yaml'),'--direction','exo2ego','--gpus','0','--output-root',str(H/'runs')]
  child=subprocess.Popen(cmd,env=env,cwd=R/'code',stdout=log,stderr=subprocess.STDOUT)
  while child.poll() is None:
   content=(H/'candidate_generation.log').read_text(errors='replace');matches=re.findall(r'Iter\(test\) \[\s*(\d+)/(\d+)\]\s+eta: (\d+):(\d+):(\d+)',content);m=matches[-1] if matches else None
   status(state='running',stage='candidate_generation',pairs_done=int(m[0]) if m else 0,pairs_total=512,eta_seconds=sum(int(x)*f for x,f in zip(m[2:],(3600,60,1))) if m else None);time.sleep(20)
  if child.returncode:raise RuntimeError('Candidate generation failed '+str(child.returncode))
  receipt=re.search(r'PCCS_RANK_RECEIPT rank=0 total=(\d+) skipped=(\d+)',(H/'candidate_generation.log').read_text());assert receipt and receipt.groups()==('512','0')
 status(state='running',stage='omama_and_gate_scoring',eta_seconds=None)
 with (H/'scoring.log').open('w') as log:subprocess.run([sys.executable,str(R/'score_holdout.py')],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 (H/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True));status(state='complete',stage='complete',eta_seconds=0)
except Exception as e:status(state='failed',error=str(e));raise
