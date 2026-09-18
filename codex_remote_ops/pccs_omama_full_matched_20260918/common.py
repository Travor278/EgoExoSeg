from pathlib import Path
import hashlib,json,time
R=Path(__file__).parent
F=R.parent/'pccs_roi_full_exoego_20260917'
C=R.parent/'pccs_corrected_exoego_20260916'
EXPERTS=('visual','anchor','fusion')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def status(p,**kw):write(p,{**kw,'updated_at':time.time()})
def verified_rows(p):
    p=Path(p);assert sha(p)==read(p.parent/'receipt.json')['records_sha256']
    return [json.loads(s) for s in p.read_text().splitlines()]
