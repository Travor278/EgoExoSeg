import os,subprocess,sys,time
from common import *
P=R.parent/'context_pccs_20260914'
env={**os.environ,'PATH':str(P/'runtime_env/bin')+':'+os.environ['PATH'],'PYTHONPATH':str(R)+':'+str(C/'python_deps')+':'+str(F/'code/mmengine')+':'+str(F/'code'),'LD_LIBRARY_PATH':str(P/'runtime_lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','PYTHONNOUSERSITE':'1'}
def parallel(script,phase):
    name=phase if script=='worker.py' else 'reference_'+phase;children=[];handles=[]
    try:
        for rank in range(4):
            h=(R/f'{name}_rank{rank}.log').open('a');handles.append(h);children.append(subprocess.Popen([sys.executable,str(R/script),'--phase',phase,'--rank',str(rank)],env={**env,'CUDA_VISIBLE_DEVICES':str(rank)},cwd=R,stdout=h,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError(name+' rank failed')
            states=[read(p) if p.exists() else {'state':'initializing'} for p in [R/'runs'/name/f'rank{i}'/'status.json' for i in range(4)]];eta=max(s['eta_seconds'] for s in states) if all(s.get('eta_seconds') is not None for s in states) else None;status(R/'status.json',state='running',phase=name,ranks=states,eta_seconds=eta);time.sleep(15)
        assert all(p.returncode==0 for p in children)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for h in handles:h.close()
def run(script,*args):
    status(R/'status.json',state='running',phase=script,eta_seconds=None)
    with (R/(script+'.log')).open('a') as f:subprocess.run([sys.executable,str(R/script),*args],env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
    stage=sys.argv[1] if len(sys.argv)>1 else 'generation'
    try:
        assert stage in ('generation','reference');gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not gpu.strip();(R/('gpu_preflight_'+stage+'.txt')).write_text(gpu)
        if stage=='generation':run('prepare.py');parallel('worker.py','smoke1');parallel('reference.py','smoke1');run('evaluate.py','--phase','smoke1');parallel('worker.py','full1')
        else:
            for i in range(4):verified_rows(R/'runs/full1'/f'rank{i}'/'records.jsonl')
            parallel('reference.py','full1');run('evaluate.py','--phase','full1')
        (R/('gpu_after_'+stage+'.txt')).write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True));status(R/'status.json',state='complete',phase=stage+'_complete',remaining_stage='reference' if stage=='generation' else None,eta_seconds=0)
    except Exception as e:status(R/'status.json',state='failed',phase=stage,error=repr(e));raise
