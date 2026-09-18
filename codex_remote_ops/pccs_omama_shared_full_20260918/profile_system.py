"""Read-only utilization/process sampling for this job; no model/code changes."""
from pathlib import Path
import json,os,subprocess,time
R=Path(__file__).resolve().parent
if R.name=='perf':R=R.parent
def workers():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if str(R/'worker.py') in cmd and '--phase full1' in cmd:found.append(int(p.name))
        except (OSError,UnicodeError):pass
    return found
def snapshot(pid):
    try:
        p=Path('/proc')/str(pid);fields=(p/'stat').read_text().split(') ',1)[1].split();io={k:int(v) for k,v in (s.split(': ') for s in (p/'io').read_text().splitlines())}
        return {'cpu_seconds':(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),'io':io,'threads':int(fields[17])}
    except OSError:return None
pids=workers();start=time.monotonic();before={str(p):snapshot(p) for p in pids};samples=[]
for i in range(9):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
    samples.append({'time':time.time(),'gpu_csv':raw})
    if i<8:time.sleep(5)
elapsed=time.monotonic()-start;after={str(p):snapshot(p) for p in pids};usage={}
for p in before:
    a,b=before[p],after[p]
    if a and b:usage[p]={'cpu_percent_one_core':100*(b['cpu_seconds']-a['cpu_seconds'])/elapsed,'threads':b['threads'],'io_delta_bytes':{k:b['io'][k]-a['io'][k] for k in ('read_bytes','write_bytes','rchar','wchar')}}
print(json.dumps({'elapsed_seconds':elapsed,'worker_pids':pids,'process_usage':usage,'samples':samples},indent=2))
