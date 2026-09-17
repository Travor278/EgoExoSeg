"""Resume only after the previous job has released its nodes; preserve completed pairs.

This helper does not alter the frozen worker, models, configuration or candidate seed.
Any truncated last record / incomplete final object group is archived before repair.
Interior corruption is rejected. It is not a license to delete failed examples.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time
R=Path(__file__).parent

def complete_prefix(raw,ann):
    lines=raw.splitlines(keepends=True);parsed=[];truncated=False
    for i,line in enumerate(lines):
        if not line.strip():assert i==len(lines)-1,'Interior empty record';continue
        try:row=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):
            assert i==len(lines)-1,'Interior JSON corruption';truncated=True;break
        key=(row['video_id'],str(row['obj_id']));assert key[0] in ann and key[1] in {str(k) for k in ann[key[0]]['objects']},'Unexpected record';parsed.append((line,row,key))
    assert len({x[2] for x in parsed})==len(parsed),'Duplicate object records'
    groups={};order=[]
    for _,_,(v,o) in parsed:
        if v not in groups:groups[v]=set();order.append(v)
        assert v==order[-1],'Non-contiguous pair records';groups[v].add(o)
    incomplete=[v for v,objs in groups.items() if objs!={str(k) for k in ann[v]['objects']}]
    assert not incomplete or incomplete==order[-1:],'Incomplete interior pair'
    kept=[x for x in parsed if not incomplete or x[2][0]!=incomplete[0]]
    content=b''.join(x[0] if x[0].endswith(b'\n') else x[0]+b'\n' for x in kept)
    return content,{'retained_pairs':len(groups)-len(incomplete),'retained_objects':len(kept),'truncated_last_line':truncated,'discarded_incomplete_final_pair':incomplete,'discarded_complete_object_records':len(parsed)-len(kept),'rewritten':content!=raw}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);p.add_argument('--release-receipt',required=True);args=p.parse_args()
    release=json.loads(Path(args.release_receipt).read_text());cells=release['cells'];assert str(cells[7])=='0' and cells[5] not in ('排队中','运行中','创建中','准备中'),'Prior job not released'
    assert cells[0].startswith('v2sam-roi-full-exoego-seed'+str(args.seed)), 'Release receipt belongs to another experiment/seed'
    current=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not current.strip(),'Current GPU already occupied'
    m=json.loads((R/'manifest.json').read_text());assert hashlib.sha256((R/'selection.json').read_bytes()).hexdigest()==m['frozen_selection_sha256']
    for n,h in m['code_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
    phase='full'+str(args.seed);archive=R/'recovery'/f'{phase}_{time.time_ns()}';archive.mkdir(parents=True,exist_ok=False);shutil.copy2(args.release_receipt,archive/'prior_job_release.json');audit=[]
    for n in ('status.json','controller.log',phase+'_results.json','summarize.py.log'):
        if (R/n).exists():shutil.copy2(R/n,archive/n)
    for stage in ('smoke'+str(args.seed),phase):
        for rank,spec in enumerate(m['phases'][stage]['shards']):
            out=R/'runs'/stage/f'rank{rank}';saved=archive/stage/f'rank{rank}';saved.mkdir(parents=True);path=out/'records.jsonl'
            for n in ('status.json','receipt.json'):
                if (out/n).exists():shutil.copy2(out/n,saved/n)
            log=R/f'{stage}_rank{rank}.log'
            if log.exists():shutil.copy2(log,saved/'worker.log')
            if not path.exists():audit.append({'phase':stage,'rank':rank,'retained_pairs':0});continue
            raw=path.read_bytes();shutil.copy2(path,saved/'records.jsonl');ann=json.loads(Path(spec['annotation']).read_text());new,info=complete_prefix(raw,ann)
            if (out/'receipt.json').exists():
                receipt=json.loads((out/'receipt.json').read_text());assert not info['rewritten'] and receipt['records_sha256']==hashlib.sha256(raw).hexdigest() and info['retained_pairs']==spec['pairs']
            elif info['rewritten']:
                temp=out/('records.resume.'+str(os.getpid())+'.tmp');temp.write_bytes(new);temp.replace(path)
            audit.append({'phase':stage,'rank':rank,**info,'archived_sha256':hashlib.sha256(raw).hexdigest(),'resume_sha256':hashlib.sha256(new).hexdigest()})
    (archive/'recovery_audit.json').write_text(json.dumps(audit,indent=2))
    import controller
    with (R/'recovery_latest.json').open('w') as f:json.dump({'archive':str(archive),'seed':args.seed,'prior_job_release':release},f,indent=2)
    controller.execute(args.seed)
if __name__=='__main__':main()
