from pathlib import Path
import json,os,subprocess,sys,time
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914';ARMS=['baseline','weighted_pool','reliable_points','both']
env=dict(os.environ,PATH=str(P/'runtime_env/bin')+':'+os.environ['PATH'],PYTHONPATH=str(R)+':'+str(R/'code/mmengine')+':'+str(R/'code'),LD_LIBRARY_PATH=str(P/'runtime_lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1')
def status(**x):
    t=R/'status.tmp';t.write_text(json.dumps({**x,'updated_at':time.time(),'pid':os.getpid()},indent=2));t.replace(R/'status.json')
def phase_run(phase,arms):
    procs=[];handles=[]
    try:
        for gpu,arm in enumerate(arms):
            h=(R/(phase+'_'+arm+'.log')).open('w');handles.append(h);procs.append(subprocess.Popen([sys.executable,str(R/'worker.py'),'--phase',phase,'--arm',arm],env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},cwd=R,stdout=h,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in procs):
            if any(p.poll() not in (None,0) for p in procs):raise RuntimeError('Worker failed in '+phase)
            states={}
            for arm in arms:
                p=R/'runs'/phase/arm/'status.json';states[arm]=json.loads(p.read_text()) if p.exists() else {'state':'initializing'}
            eta=max(x['eta_seconds'] for x in states.values()) if all(x.get('eta_seconds') is not None for x in states.values()) else None
            status(state='running',phase=phase,arms=states,eta_seconds=eta);time.sleep(15)
        assert all(p.returncode==0 for p in procs)
    finally:
        for p in procs:
            if p.poll() is None:p.terminate()
        for p in procs:
            if p.poll() is None:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for h in handles:h.close()
try:
    if (R/'status.json').exists():raise RuntimeError('Fresh experiment controller required')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);(R/'gpu_preflight.txt').write_text(gpu)
    assert not gpu.strip(),'GPU occupied'
    subprocess.run([sys.executable,str(R/'test_candidate_ops.py')],env=env,cwd=R,check=True)
    subprocess.run([sys.executable,str(R/'prepare.py')],env=env,cwd=R,check=True)
    from summarize import read_phase,decide
    for phase in ('smoke','screen','calibration'):
        phase_run(phase,ARMS);result=read_phase(phase);print('PHASE_COMPLETE',phase,flush=True)
    selection=decide();print('CALIBRATION_SELECTED',json.dumps(selection),flush=True)
    if selection['selected']!='baseline':
        import yaml
        source=R.parent/'pccs_corrected_exoexo_20260916';plan=json.loads((R/'manifest.json').read_text());a=json.loads((source/'full_annotation.json').read_text());cfg=yaml.safe_load((source/'full_runtime.yaml').read_text());cfg['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');cfg['runtime']['ports']['exo2ego']=29989
        cp=R/'exo2exo_runtime.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False));plan['phases']['exo2exo']={'annotation':str(source/'full_annotation.json'),'config':str(cp),'pairs':len(a),'objects':sum(len(v['objects']) for v in a.values())};(R/'manifest.json').write_text(json.dumps(plan,indent=2))
        phase_run('exo2exo',['baseline',selection['selected']]);read_phase('exo2exo',['baseline',selection['selected']])
    (R/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True));status(state='complete',phase='complete',selection=selection,eta_seconds=0)
except Exception as e:status(state='failed',error=repr(e));raise
