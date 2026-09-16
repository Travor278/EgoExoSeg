from pathlib import Path
import json,os,subprocess,sys,time
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914'
env=dict(os.environ,PATH=str(P/'runtime_env/bin')+':'+os.environ['PATH'],PYTHONPATH=str(R)+':'+str(R/'code/mmengine')+':'+str(R/'code'),LD_LIBRARY_PATH=str(P/'runtime_lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1')
def status(**x):
    p=R/'status.tmp';p.write_text(json.dumps({**x,'updated_at':time.time()}));p.replace(R/'status.json')
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
            eta=max(v['eta_seconds'] for v in states) if all(v.get('eta_seconds') is not None for v in states) else None
            status(state='running',phase=phase,ranks=states,eta_seconds=eta);time.sleep(15)
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
    assert not (R/'status.json').exists(),'Fresh controller required'
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);(R/'gpu_preflight.txt').write_text(gpu);assert not gpu.strip()
    for script in ('test_native.py','prepare.py'):
        with (R/(script+'.log')).open('w') as f:subprocess.run([sys.executable,str(R/script)],env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
    from summarize import read_phase,select
    for phase in ('smoke','screen','calibration'):phase_run(phase);read_phase(phase);print('NATIVE_PHASE_COMPLETE',phase,flush=True)
    selected=select();print('NATIVE_SELECTION',selected['selected'],flush=True)
    if selected['selected']!='baseline':
        import yaml
        source=R.parent/'pccs_corrected_exoexo_20260916';a=json.loads((source/'full_annotation.json').read_text());cfg=yaml.safe_load((source/'full_runtime.yaml').read_text());cfg['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');cfg['runtime']['ports']['exo2ego']=29889;keys=list(a);spec={'pairs':len(a),'objects':sum(len(v['objects']) for v in a.values()),'reference_root':str(R.parent/'pccs_candidate_union_20260916'),'reference_arm':'baseline','shards':[]}
        for rank in range(4):
            ann={k:a[k] for k in keys[rank::4]};ap=R/f'exo2exo_rank{rank}.json';ap.write_text(json.dumps(ann));c=json.loads(json.dumps(cfg));c['data']['annotations']['exo2ego']=str(ap);cp=R/f'exo2exo_rank{rank}.yaml';cp.write_text(yaml.safe_dump(c,sort_keys=False));spec['shards'].append({'annotation':str(ap),'config':str(cp),'pairs':len(ann),'objects':sum(len(v['objects']) for v in ann.values())})
        m=json.loads((R/'manifest.json').read_text());m['phases']['exo2exo']=spec;(R/'manifest.json').write_text(json.dumps(m,indent=2));phase_run('exo2exo');result=read_phase('exo2exo');result['preselected_primary']=selected['selected'];(R/'exo2exo_results.json').write_text(json.dumps(result,indent=2))
    (R/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True));status(state='complete',phase='complete',selected=selected,eta_seconds=0)
except Exception as e:status(state='failed',error=repr(e));raise
