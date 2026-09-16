from pathlib import Path
import os,json,subprocess,sys,time
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914';F=R.parent/'pccs_corrected_exoego_20260916'
env=dict(os.environ,PATH=str(P/'runtime_env/bin')+':'+os.environ['PATH'],PYTHONPATH=str(R)+':'+str(F/'python_deps')+':'+str(R/'code/mmengine')+':'+str(R/'code'),LD_LIBRARY_PATH=str(P/'runtime_lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1')
def status(**x):
    p=R/'status.tmp';p.write_text(json.dumps({**x,'updated_at':time.time()}));p.replace(R/'status.json')
def run(script):
    with (R/(script+'.log')).open('w') as f:
        child=subprocess.Popen([sys.executable,str(R/script)],env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT)
        while child.poll() is None:status(state='running',phase=script,eta_seconds=None);time.sleep(10)
        if child.returncode:raise RuntimeError(script+' failed')
def phase_run(phase):
    children=[];logs=[]
    try:
        for rank in range(4):
            h=(R/f'{phase}_rank{rank}.log').open('w');logs.append(h);children.append(subprocess.Popen([sys.executable,str(R/'worker.py'),'--phase',phase,'--rank',str(rank)],env={**env,'CUDA_VISIBLE_DEVICES':str(rank)},cwd=R,stdout=h,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError(phase+' worker failed')
            states=[]
            for rank in range(4):
                p=R/'runs'/phase/f'rank{rank}'/'status.json';states.append(json.loads(p.read_text()) if p.exists() else {'state':'initializing'})
            eta=max(v['eta_seconds'] for v in states) if all(v.get('eta_seconds') is not None for v in states) else None;status(state='running',phase=phase,ranks=states,eta_seconds=eta);time.sleep(15)
        assert all(p.returncode==0 for p in children)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            if p.poll() is None:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for h in logs:h.close()
try:
    assert not (R/'status.json').exists();gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);(R/'gpu_preflight.txt').write_text(gpu);assert not gpu.strip()
    run('test_policies.py');run('test_dense.py');run('prepare.py')
    from data_utils import read
    for phase in ('smoke','train','calibration'):phase_run(phase);read(phase);print('DENSE_PHASE_COMPLETE',phase,flush=True)
    run('fit_and_select.py');choice=json.loads((R/'selection.json').read_text())
    if choice['selected']['family']!='baseline':
        import yaml
        from prepare import add_phase
        source=R.parent/'pccs_corrected_exoexo_20260916';m=json.loads((R/'manifest.json').read_text());a=json.loads((source/'full_annotation.json').read_text());cfg=yaml.safe_load((source/'full_runtime.yaml').read_text());add_phase(m,'exo2exo',a,cfg,R.parent/'pccs_candidate_union_20260916','exo2exo');(R/'manifest.json').write_text(json.dumps(m,indent=2));phase_run('exo2exo');read('exo2exo');run('evaluate.py')
    (R/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True));status(state='complete',phase='complete',selected=choice['selected']['name'],eta_seconds=0)
except Exception as e:status(state='failed',error=repr(e));raise
