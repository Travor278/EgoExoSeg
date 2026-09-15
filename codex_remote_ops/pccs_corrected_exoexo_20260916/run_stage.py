from pathlib import Path
import argparse,hashlib,json,math,os,re,subprocess,sys,time
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914'
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--gpus',required=True);args=p.parse_args();gpus=[int(x) for x in args.gpus.split(',')];world=len(gpus);S=R/'runs'/args.stage
 manifest=json.loads((R/'manifest.json').read_text());spec=manifest['stages'][args.stage]
 if (S/'results.json').exists():raise RuntimeError('Stage already complete; do not duplicate '+args.stage)
 if S.exists():raise RuntimeError('Stage outputs already exist; inspect before resuming')
 S.mkdir(parents=True)
 def status(**x):
  value={**x,'stage':args.stage,'updated_at':time.time(),'pid':os.getpid()}
  for target in (R/'status.json',S/'status.json'):
   tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(target)
 env=dict(os.environ,PATH=str(P/'runtime_env/bin')+':'+os.environ['PATH'],PYTHONPATH=str(R/'python_deps')+':'+str(R/'code/mmengine')+':'+str(R/'code'),LD_LIBRARY_PATH=str(P/'runtime_lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),PCCS_CONTEXT='0',PCCS_GAIN_BANK=str(S/'candidate_bank'),PCCS_DIST_TIMEOUT='7200',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 try:
  for file,digest in manifest['frozen_files'].items():assert hashlib.sha256((R/file).read_bytes()).hexdigest()==digest,('Frozen file changed',file)
  gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);(S/'preflight_gpu_processes.txt').write_text(gpu)
  if gpu.strip():raise RuntimeError('GPU compute processes already active; do not interfere')
  resources=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True);(S/'preflight_resources.txt').write_text(resources)
  with (S/'candidate_generation.log').open('w') as log:
   child=subprocess.Popen([sys.executable,str(R/'code/tools/pccs_eval.py'),'--config',spec['config'],'--direction','exo2ego','--gpus',args.gpus,'--output-root',str(S/'candidates')],env=env,cwd=R/'code',stdout=log,stderr=subprocess.STDOUT)
   while child.poll() is None:
    content=(S/'candidate_generation.log').read_text(errors='replace');matches=re.findall(r'Iter\(test\) \[\s*(\d+)/(\d+)\]\s+eta: (\d+):(\d+):(\d+)',content);m=matches[-1] if matches else None
    status(state='running',phase='candidate_generation',rank0_batches_done=int(m[0]) if m else 0,rank0_batches_total=int(m[1]) if m else math.ceil(spec['pairs']/world),gpus=gpus,eta_seconds=sum(int(x)*f for x,f in zip(m[2:],(3600,60,1))) if m else None);time.sleep(20)
   if child.returncode:raise RuntimeError('Candidate generation failed '+str(child.returncode))
  content=(S/'candidate_generation.log').read_text();receipts={int(rank):(int(total),int(skipped)) for rank,total,skipped in re.findall(r'PCCS_RANK_RECEIPT rank=(\d+) total=(\d+) skipped=(\d+)',content)}
  assert receipts=={rank:(math.ceil(spec['pairs']/world),0) for rank in range(world)},receipts
  (S/'candidate_receipts.json').write_text(json.dumps(receipts,indent=2))
  status(state='running',phase='scoring',eta_seconds=None,gpus=gpus);children=[];logs=[]
  try:
   for rank,gpu in enumerate(gpus):
    log=(S/f'scoring_rank{rank}.log').open('w');logs.append(log)
    children.append(subprocess.Popen([sys.executable,str(R/'score.py'),'--stage',args.stage,'--rank',str(rank),'--world-size',str(world)],env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},cwd=R,stdout=log,stderr=subprocess.STDOUT))
   while any(c.poll() is None for c in children):
    if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Scoring rank failed')
    states=[]
    for rank in range(world):
     path=S/'scored'/f'rank{rank}_status.json';states.append(json.loads(path.read_text()) if path.exists() else {'state':'initializing'})
    known=all(x.get('eta_seconds') is not None for x in states)
    status(state='running',phase='scoring',ranks=states,eta_seconds=max(x['eta_seconds'] for x in states) if known else None,gpus=gpus);time.sleep(20)
   if any(c.returncode for c in children):raise RuntimeError('Scoring worker failed')
  finally:
   for c in children:
    if c.poll() is None:c.terminate()
   for c in children:
    if c.poll() is None:
     try:c.wait(timeout=10)
     except subprocess.TimeoutExpired:c.kill();c.wait()
   for log in logs:log.close()
  status(state='running',phase='summarize',eta_seconds=None,gpus=gpus)
  with (S/'summarize.log').open('w') as log:subprocess.run([sys.executable,str(R/'summarize.py'),'--stage',args.stage,'--world-size',str(world)],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
  (S/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True));status(state='complete',phase='complete',eta_seconds=0,gpus=gpus)
 except Exception as e:status(state='failed',error=str(e));raise
if __name__=='__main__':main()
