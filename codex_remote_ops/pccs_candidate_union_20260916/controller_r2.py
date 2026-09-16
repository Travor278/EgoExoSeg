from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914';Q=R.parent/'pccs_candidate_quality_20260916';F=R.parent/'pccs_corrected_exoego_20260916'
RESIDUAL=['residual005','residual010','residual025']
env=dict(os.environ,PATH=str(P/'runtime_env/bin')+':'+os.environ['PATH'],PYTHONPATH=str(R)+':'+str(F/'python_deps')+':'+str(R/'code/mmengine')+':'+str(R/'code'),LD_LIBRARY_PATH=str(P/'runtime_lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1')
def status(**x):
    temp=R/'status.tmp';temp.write_text(json.dumps({**x,'updated_at':time.time(),'pid':os.getpid()},indent=2));temp.replace(R/'status.json')
def run(script,phase,args=(),gpu='0',progress=None):
    logname=phase+'.log'
    with (R/logname).open('w') as log:
        child=subprocess.Popen([sys.executable,str(R/script),*args],env={**env,'CUDA_VISIBLE_DEVICES':gpu},cwd=R,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            detail={}
            if progress and (R/progress).exists():detail=json.loads((R/progress).read_text())
            status(state='running',phase=phase,eta_seconds=detail.get('eta_seconds'),detail=detail,log=logname);time.sleep(10)
        if child.returncode:raise RuntimeError(phase+' failed: '+str(child.returncode))
def arms(phase,names):
    children=[];handles=[]
    try:
        for gpu,name in enumerate(names):
            h=(R/(phase+'_'+name+'.log')).open('w');handles.append(h);children.append(subprocess.Popen([sys.executable,str(R/'worker.py'),'--phase',phase,'--arm',name],env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},cwd=R,stdout=h,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Candidate worker failed: '+phase)
            states={}
            for name in names:
                p=R/'runs'/phase/name/'status.json';states[name]=json.loads(p.read_text()) if p.exists() else {'state':'initializing'}
            eta=max(v['eta_seconds'] for v in states.values()) if all(v.get('eta_seconds') is not None for v in states.values()) else None
            status(state='running',phase='candidates_'+phase,arms=states,eta_seconds=eta);time.sleep(15)
        assert all(p.returncode==0 for p in children)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            if p.poll() is None:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for h in handles:h.close()
def check_anchors(phase):
    def read(root,arm):
        p=root/'runs'/phase/arm/'per_object.jsonl';receipt=json.loads((p.parent/'receipt.json').read_text());assert receipt['coverage']=='exact' and receipt['records_sha256']==hashlib.sha256(p.read_bytes()).hexdigest();rows=[json.loads(s) for s in p.read_text().splitlines()];return {(r['video_id'],r['obj_id']):r for r in rows}
    base=read(Q,'baseline');checks={}
    for arm in RESIDUAL:
        other=read(R,arm);assert base.keys()==other.keys();mismatch=[k for k in base if base[k]['mask_sha256']['anchor']!=other[k]['mask_sha256']['anchor']];checks[arm]={'anchor_mismatches':len(mismatch),'objects':len(base)};assert not mismatch
    (R/(phase+'_isolation.json')).write_text(json.dumps(checks,indent=2))
try:
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);(R/'gpu_preflight_r2.txt').write_text(processes);assert not processes.strip(),'GPU occupied'
    for phase in ('smoke','screen','calibration'):check_anchors(phase)
    run('probe_precision.py','precision_probe')
    for phase in ('screen','calibration'):run('build_dataset.py','dataset_'+phase,['--phase',phase],gpu='0',progress='dataset_'+phase+'_status.json')
    run('fit_models.py','fit_quality_and_omama',gpu='0');choice=json.loads((R/'selection.json').read_text());print('FROZEN_SELECTION',choice['selected']['name'],flush=True)
    if choice['selected']['family']!='baseline':
        import yaml
        source=R.parent/'pccs_corrected_exoexo_20260916';manifest=json.loads((R/'manifest.json').read_text());a=json.loads((source/'full_annotation.json').read_text());cfg=yaml.safe_load((source/'full_runtime.yaml').read_text());cfg['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');cfg['runtime']['ports']['exo2ego']=29993
        cp=R/'exo2exo_runtime.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False));manifest['phases']['exo2exo']={'annotation':str(source/'full_annotation.json'),'config':str(cp),'pairs':len(a),'objects':sum(len(v['objects']) for v in a.values())};(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
        arms('exo2exo',['baseline','reliable_points','residual005','residual010']);arms('exo2exo',['residual025'])
        run('build_dataset.py','dataset_exo2exo',['--phase','exo2exo'],progress='dataset_exo2exo_status.json');run('evaluate_selected.py','evaluate_selected')
    (R/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True));status(state='complete',phase='complete',selected=choice['selected']['name'],eta_seconds=0)
except Exception as e:status(state='failed',error=repr(e));raise
