from pathlib import Path
import json,hashlib,shutil,time,yaml
R=Path(__file__).parent;P=R.parent/'pccs_roi_context_20260917';Q=R.parent/'pccs_candidate_quality_20260916';X=R.parent/'pccs_corrected_exoexo_20260916'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def add_phase(m,phase,ann,cfg):
    spec={'pairs':len(ann),'objects':sum(len(a['objects']) for a in ann.values()),'shards':[]};keys=list(ann);images=Path(cfg['data']['images'])
    for a in ann.values():
        for field in (a['prompt']['first_frame_image'],a['video_path']):
            for path in field if isinstance(field,list) else [field]:assert (images/path).is_file(),path
    for rank in range(4):
        a={k:ann[k] for k in keys[rank::4]};ap=R/f'{phase}_rank{rank}.json';ap.write_text(json.dumps(a));c=json.loads(json.dumps(cfg));c['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');c['data']['annotations']['exo2ego']=str(ap);c['runtime']['ports']['exo2ego']=29881;cp=R/f'{phase}_rank{rank}.yaml';cp.write_text(yaml.safe_dump(c,sort_keys=False));spec['shards'].append({'annotation':str(ap),'config':str(cp),'pairs':len(a),'objects':sum(len(v['objects']) for v in a.values())})
    m['phases'][phase]=spec
def main():
    from build_temporal import build
    from native_bridge import patch_metric
    oldann=json.loads((X/'full_annotation.json').read_text());expanded,ds=build(oldann);assert len(expanded)==6534;ap=R/'temporal_expanded_annotation.json';ap.write_text(json.dumps(expanded));assert sha(ap)=='dd82c38fd5515c61a885476649015e66fe0b01d2db2ec4636d0079cf77aa7be5';ds['annotation_sha256']=sha(ap);(R/'temporal_expansion_manifest.json').write_text(json.dumps(ds,indent=2))
    assert not (R/'code').exists();shutil.copytree(Q/'code',R/'code',symlinks=True);patch_metric(R/'code');old=json.loads((P/'manifest.json').read_text())
    for n in ('encoder_contract.py','roi_views.py','roi_evidence.py','roi_policy.py','dense_context.py','policies.py','native_bridge.py','randomness.py'):assert sha(R/n)==sha(P/n),n
    for item in old['assets'].values():assert sha(item['path'])==item['sha256']
    selection_sha='a913b51b9aa068706870adc070f8ec32e437776cb070c05150853ae896f043c3';assert sha(P/'selection.json')==selection_sha;shutil.copy2(P/'selection.json',R/'selection.json');sel=json.loads((R/'selection.json').read_text())
    for c in sel['matched_controls']+[sel['selected'],old['frozen_cycle']]:assert sha(c['checkpoint'])==c['sha256']
    m={'created_at':time.time(),'phases':{},'assets':old['assets'],'frozen_cycle':old['frozen_cycle'],'encoder_contract':json.loads((P/'runs/smoke/rank0/encoder_receipt.json').read_text()),'frozen_selection_sha256':selection_sha,'training_performed':False,'scope':ds,'external_reference':'Independent frozen O-MaMa same expanded candidate bank; no reused old outcomes'}
    cfg=yaml.safe_load((X/'full_runtime.yaml').read_text());smoke_keys=sorted(expanded,key=lambda k:hashlib.sha256(('expanded-smoke:'+k).encode()).hexdigest())[:8];add_phase(m,'smoke',{k:expanded[k] for k in smoke_keys},cfg);add_phase(m,'exo2exo',expanded,cfg)
    m['code_sha256']={n:sha(R/n) for n in ('worker.py','roi_views.py','roi_evidence.py','roi_policy.py','dense_context.py','policies.py','native_bridge.py','randomness.py','reference_worker.py','reference_collect.py','evaluate.py','code/projects/v2sam_pccs/evaluation/pccs_metric.py')};(R/'manifest.json').write_text(json.dumps(m,indent=2));print('EXPANDED6534_READY',flush=True)
if __name__=='__main__':main()
