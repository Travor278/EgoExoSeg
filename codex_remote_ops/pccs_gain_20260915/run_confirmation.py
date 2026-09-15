from pathlib import Path
import json,os,subprocess,sys,time
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914'
def status(x):
    p=R/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({**x,'updated_at':time.time(),'pid':os.getpid()},indent=2));tmp.replace(p)
env=dict(os.environ,PATH=str(P/'runtime_env/bin')+':'+os.environ['PATH'],PYTHONPATH=str(R/'code/mmengine')+':'+str(R/'code'),LD_LIBRARY_PATH=str(P/'runtime_lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),PCCS_CONTEXT='0',PCCS_GAIN_BANK=str(R/'candidate_bank'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
try:
    if (R/'runs/confirmation').exists():raise RuntimeError('Fresh run required')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    (R/'gpu_preflight.txt').write_text(gpu)
    if gpu.strip():raise RuntimeError('GPU busy, do not interfere: '+gpu)
    plan=json.loads((R/'confirmation_plan.json').read_text());total=plan['objects']
    with (R/'candidate_generation.log').open('w') as log:
        cmd=[sys.executable,str(R/'code/tools/pccs_eval.py'),'--config',str(R/'confirmation_runtime.yaml'),'--direction','exo2ego','--gpus','0','--output-root',str(R/'runs/confirmation')]
        process=subprocess.Popen(cmd,env=env,cwd=R/'code',stdout=log,stderr=subprocess.STDOUT);start=time.monotonic()
        while process.poll() is None:
            n=len(list((R/'candidate_bank').glob('rank*/*.npz')));elapsed=time.monotonic()-start
            status({'state':'running','stage':'candidate_generation','done':n,'total':total,'eta_seconds':elapsed/n*(total-n) if n>=20 else None,'child_pid':process.pid})
            time.sleep(20)
        if process.returncode:raise RuntimeError('Candidate generation exit '+str(process.returncode))
    status({'state':'running','stage':'omama_scoring','eta_seconds':None})
    with (R/'scoring.log').open('w') as log:
        subprocess.run([sys.executable,str(R/'evaluate_confirmation.py')],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
    (R/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True))
    status({'state':'complete','stage':'complete','eta_seconds':0})
except Exception as e:status({'state':'failed','error':str(e)});raise
