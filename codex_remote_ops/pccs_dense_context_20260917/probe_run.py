from pathlib import Path
import os,sys,subprocess,json,time,shutil
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914';F=R.parent/'pccs_corrected_exoego_20260916'
env=dict(os.environ,PATH=str(P/'runtime_env/bin')+':'+os.environ['PATH'],PYTHONPATH=str(R)+':'+str(F/'python_deps')+':'+str(R/'code/mmengine')+':'+str(R/'code'),LD_LIBRARY_PATH=str(P/'runtime_lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='0')
if (R/'status.json').exists() and not (R/'status_before_probe.json').exists():shutil.copy2(R/'status.json',R/'status_before_probe.json')
(R/'status.json').write_text(json.dumps({'state':'running','phase':'historical_identity_probe','eta_seconds':None,'updated_at':time.time()}))
with (R/'probe_failure.log').open('w') as log:p=subprocess.run([sys.executable,str(R/'probe_failure.py')],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
(R/'status.json').write_text(json.dumps({'state':'probe_complete','phase':'historical_identity_probe','passed':p.returncode==0,'eta_seconds':0,'updated_at':time.time()}))
sys.exit(p.returncode)
