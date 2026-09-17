from pathlib import Path
import hashlib,json,shutil,time,yaml
R=Path(__file__).parent;P=R.parent/'pccs_roi_context_20260917';Q=R.parent/'pccs_candidate_quality_20260916';C=R.parent/'pccs_corrected_exoego_20260916'
FROZEN=('encoder_contract.py','roi_views.py','roi_evidence.py','roi_policy.py','dense_context.py','policies.py','native_bridge.py')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()
def add_phase(m,phase,ann,cfg,seed):
    keys=list(ann);spec={'pairs':len(keys),'objects':sum(len(v['objects']) for v in ann.values()),'shards':[]}
    for rank in range(4):
        a={k:ann[k] for k in keys[rank::4]};ap=R/(phase+f'_rank{rank}.json');ap.write_text(json.dumps(a));c=json.loads(json.dumps(cfg));c['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');c['data']['annotations']['exo2ego']=str(ap);c['runtime']['ports']['exo2ego']=29841;cp=R/(phase+f'_rank{rank}.yaml');cp.write_text(yaml.safe_dump(c,sort_keys=False));spec['shards'].append({'seed':seed,'annotation':str(ap),'config':str(cp),'pairs':len(a),'objects':sum(len(v['objects']) for v in a.values()),'annotation_sha256':sha(ap)})
    m['phases'][phase]=spec
def main():
    assert not (R/'code').exists();shutil.copytree(Q/'code',R/'code',symlinks=True)
    from native_bridge import patch_metric
    patch_metric(R/'code');old=json.loads((P/'manifest.json').read_text());historical=json.loads((C/'manifest.json').read_text());previous=json.loads((C/'runs/full/results.json').read_text())
    assert previous['coverage']=='exact' and previous['pairs']==46515 and previous['objects']==109253
    for n in FROZEN:assert sha(R/n)==sha(P/n),n
    assert old['assets']==historical['expert_assets']
    for item in old['assets'].values():assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
    expected='a913b51b9aa068706870adc070f8ec32e437776cb070c05150853ae896f043c3';assert sha(P/'selection.json')==expected;shutil.copy2(P/'selection.json',R/'selection.json');sel=json.loads((R/'selection.json').read_text())
    for cfg in sel['matched_controls']+[sel['selected'],old['frozen_cycle']]:assert sha(cfg['checkpoint'])==cfg['sha256']
    full=json.loads((C/'full_annotation.json').read_text());cfg=yaml.safe_load((C/'full_runtime.yaml').read_text());assert len(full)==46515 and sum(len(v['objects']) for v in full.values())==109253
    for e,n in (('visual','vp_exo2ego_full.pth'),('fusion','fusion_exo2ego_full.pth')):assert cfg['models']['experts']['exo2ego'][e]==old['assets'][n]['path']
    images=Path(cfg['data']['images']);paths=set()
    for a in full.values():
        for v in (a['prompt']['first_frame_image'],a['video_path']):paths.update(v if isinstance(v,list) else [v])
    missing=[p for p in paths if not (images/p).is_file()];assert not missing,missing[:5]
    m={'created_at':time.time(),'phases':{},'assets':old['assets'],'encoder_contract':json.loads((P/'runs/smoke/rank0/encoder_receipt.json').read_text()),'frozen_cycle':old['frozen_cycle'],'frozen_selection_sha256':expected,'training_performed':False,'full_annotation_sha256':sha(C/'full_annotation.json'),'checked_images':len(paths),'source_code_commit':'v2sam-pccs@0e3bc33dec3e202ffbb86cec01038e60b18c162a','max_gpu_jobs_concurrent':1,'scope':'Full46515pair109253object Exo2Ego, candidate-quality-v1/v2. Parameters frozen; previously observed benchmark, not fresh blind test. Scientific controls separate from full selection.'}
    smoke=json.loads((Q/'smoke.json').read_text());scfg=yaml.safe_load((Q/'screen_runtime.yaml').read_text())
    for seed in (1,2):add_phase(m,'smoke'+str(seed),smoke,scfg,seed);add_phase(m,'full'+str(seed),full,cfg,seed)
    # Scientific ablations use frozen seed1 candidates for the already observed targets.
    for phase,oldphase in (('science_smoke','smoke'),('science_exo2exo','exo2exo'),('science_exo2ego','exo2ego')):
        oldspec=old['phases'][oldphase];m['phases'][phase]={'source_phase':oldphase,'pairs':oldspec['pairs'],'objects':oldspec['objects'],'shards':oldspec['shards']}
    names=FROZEN+('randomness.py','worker.py','science_views.py','science_worker.py','science_evaluate.py','summarize.py','controller.py','code/projects/v2sam_pccs/evaluation/pccs_metric.py');m['code_sha256']={n:sha(R/n) for n in names}
    (R/'manifest.json').write_text(json.dumps(m,indent=2))
    audit={'expert_assets_verified_by_bytes_and_sha256':old['assets'],'previous_omama_full_used_identical_assets':True,'previous_full_result_sha256':sha(C/'runs/full/results.json'),'previous_full_scope':{k:previous[k] for k in ('pairs','objects','takes','coverage')},'previous_full_pccs':previous['methods']['baseline']['frame'],'previous_full_omama_consensus':previous['methods']['geometry_consensus']['frame'],'historical_reference_is_not_same_candidate_seed_as_new_full':True,'hf_revision':historical['huggingface_revision']};(R/'weight_audit.json').write_text(json.dumps(audit,indent=2));print('FULL_PREPARED_AND_WEIGHTS_VERIFIED',flush=True)
if __name__=='__main__':main()
