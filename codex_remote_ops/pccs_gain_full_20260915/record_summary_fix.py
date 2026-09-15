from pathlib import Path
import hashlib,json,time
R=Path(__file__).parent;p=R/'manifest.json';manifest=json.loads(p.read_text())
old=manifest['frozen_files']['summarize.py'];new=hashlib.sha256((R/'summarize.py').read_bytes()).hexdigest()
manifest['frozen_files']['summarize.py']=new
manifest.setdefault('implementation_receipts',[]).append({'time':time.time(),'file':'summarize.py','old_sha256':old,'new_sha256':new,'reason':'Convert numpy Boolean count increments to Python ints for JSON serialization. No metric arithmetic, predictions, model or threshold changed. Four-shard965-object aggregation parity test passed.'})
p.write_text(json.dumps(manifest,indent=2))
