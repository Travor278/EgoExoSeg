import gzip,shutil,yaml
from common import *
def main():
    if (R/'manifest.json').exists():return
    m=read(F/'manifest.json');result=read(F/'full1_results.json')
    assert (result['pairs'],result['objects'],result['takes'])==(46515,109253,295)
    for n,h in m['code_sha256'].items():assert sha(F/n)==h,n
    for a in m['assets'].values():assert sha(a['path'])==a['sha256']
    old=read(C/'manifest.json')
    for n in ('aligned_model.py','cache.py'):assert sha(C/n)==old['frozen_files'][n]
    assert m['full_annotation_sha256']==sha(C/'full_annotation.json')
    assert shutil.disk_usage(R).free>100*1024**3,'Need 100GiB free for immutable mask bank'
    with gzip.open(F/'full1_compact.jsonl.gz','rt') as f:compact=[json.loads(s) for s in f]
    assert len(compact)==109253
    plan={'source_manifest_sha256':sha(F/'manifest.json'),'source_compact_sha256':sha(F/'full1_compact.jsonl.gz'),'selection_sha256':sha(F/'selection.json'),'assets':m['assets'],'phases':{},'reference_code_sha256':{n:sha(C/n) for n in ('aligned_model.py','cache.py')},'strict_current_seed1_masks':True}
    plan['reference_weights']={n:sha(C/'weights'/n) for n in ('omama_exo2ego.pt','dinov2_vitb14_reg4_pretrain.pth')}
    for phase in ('pilot','full'):
        plan['phases'][phase]={'shards':[]}
        for rank,s in enumerate(m['phases']['full1']['shards']):
            ann=read(s['annotation']);keys=set(sorted(ann,key=lambda k:hashlib.sha256(('omama-matched-pilot:'+k).encode()).hexdigest())[:16]) if phase=='pilot' else set(ann)
            ann={k:v for k,v in ann.items() if k in keys};ap=R/f'{phase}_rank{rank}.json';write(ap,ann)
            cfg=yaml.safe_load(Path(s['config']).read_text());cfg['data']['annotations']['exo2ego']=str(ap);cp=R/f'{phase}_rank{rank}.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
            prior=[r for r in compact if r['video_id'] in keys];pp=R/f'{phase}_rank{rank}_prior.jsonl';pp.write_text(''.join(json.dumps(r)+'\n' for r in prior))
            plan['phases'][phase]['shards'].append({'annotation':str(ap),'config':str(cp),'prior':str(pp),'prior_sha256':sha(pp),'pairs':len(ann),'objects':len(prior),'annotation_sha256':sha(ap)})
        plan['phases'][phase].update(pairs=sum(x['pairs'] for x in plan['phases'][phase]['shards']),objects=sum(x['objects'] for x in plan['phases'][phase]['shards']))
    plan['files_sha256']={p.name:sha(p) for p in R.glob('*.py')}
    write(R/'manifest.json',plan);print('PREPARED_STRICT_CURRENT_SEED1',flush=True)
if __name__=='__main__':main()
