from pathlib import Path
import hashlib,json,shutil,time,yaml
R=Path(__file__).parent;P=R.parent/'pccs_roi_context_20260917';Q=R.parent/'pccs_candidate_quality_20260916';X=R.parent/'pccs_corrected_exoexo_20260916'
METHOD_FILES=('encoder_contract.py','roi_views.py','roi_evidence.py','roi_policy.py','dense_context.py','policies.py','native_bridge.py')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def add_phase(m,phase,ann,cfg):
    images=Path(cfg['data']['images']);paths=set()
    for a in ann.values():
        for field in (a['prompt']['first_frame_image'],a['video_path']):paths.update(field if isinstance(field,list) else [field])
    assert all((images/p).is_file() for p in paths),'Missing '+phase+' images'
    spec={'pairs':len(ann),'objects':sum(len(a['objects']) for a in ann.values()),'shards':[],'images_checked':len(paths)};keys=list(ann)
    old=json.loads((P/'manifest.json').read_text())['phases'][phase]
    for rank in range(4):
        a={k:ann[k] for k in keys[rank::4]};oldann=json.loads(Path(old['shards'][rank]['annotation']).read_text());assert a==oldann and list(a)==list(oldann),'Dataset or rank ordering changed'
        ap=R/f'{phase}_rank{rank}.json';ap.write_text(json.dumps(a));c=json.loads(json.dumps(cfg));c['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');c['runtime']['ports']['exo2ego']=29821;c['data']['annotations']['exo2ego']=str(ap);cp=R/f'{phase}_rank{rank}.yaml';cp.write_text(yaml.safe_dump(c,sort_keys=False));spec['shards'].append({'annotation':str(ap),'config':str(cp),'pairs':len(a),'objects':sum(len(v['objects']) for v in a.values()),'annotation_sha256':sha(ap)})
    m['phases'][phase]=spec
def main():
    old=json.loads((P/'manifest.json').read_text());assert not (R/'code').exists();shutil.copytree(Q/'code',R/'code',symlinks=True)
    from native_bridge import patch_metric
    patch_metric(R/'code')
    for n in METHOD_FILES:assert sha(R/n)==sha(P/n),('Frozen method changed',n)
    for item in old['assets'].values():assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
    expected='a913b51b9aa068706870adc070f8ec32e437776cb070c05150853ae896f043c3';assert sha(P/'selection.json')==expected;shutil.copy2(P/'selection.json',R/'selection.json');selection=json.loads((R/'selection.json').read_text())
    for cfg in selection['matched_controls']+[selection['selected'],old['frozen_cycle']]:assert sha(cfg['checkpoint'])==cfg['sha256']
    m={'created_at':time.time(),'phases':{},'assets':old['assets'],'frozen_cycle':old['frozen_cycle'],'frozen_selection_sha256':expected,'seed_namespace':'candidate-quality-v2:','previous_seed_namespace':'candidate-quality-v1:','training_performed':False,'candidate_generation_algorithm_changed':False,'candidate_randomness_changed':True,'additional_encoding':'All original ROI ablation views; per-object encoded_views and elapsed_seconds recorded','encoder_contract':json.loads((P/'runs/smoke/rank0/encoder_receipt.json').read_text()),'pretrained_omama_used_in_native_method':False,'reference_process':'Separate frozen O-MaMa baseline on the same new candidate bank; historical witness checked before full evaluation'}
    add_phase(m,'smoke',json.loads((Q/'smoke.json').read_text()),yaml.safe_load((Q/'screen_runtime.yaml').read_text()))
    add_phase(m,'exo2exo',json.loads((X/'full_annotation.json').read_text()),yaml.safe_load((X/'full_runtime.yaml').read_text()))
    add_phase(m,'exo2ego',json.loads((R.parent/'pccs_gain_20260915/gate_training/holdout512.json').read_text()),yaml.safe_load((R.parent/'pccs_corrected_exoego_20260916/full_runtime.yaml').read_text()))
    assert m['phases']['exo2exo']['pairs']==1094 and m['phases']['exo2ego']['pairs']==512 and m['phases']['exo2ego']['objects']==965
    files=METHOD_FILES+('randomness.py','worker.py','reference_worker.py','reference_collect.py','data_utils.py','evaluate.py','controller.py','code/projects/v2sam_pccs/evaluation/pccs_metric.py');m['code_sha256']={n:sha(R/n) for n in files}
    refs=('aligned_model.py','weights/omama_exo2ego.pt','weights/dinov2_vitb14_reg4_pretrain.pth','reference/O-MaMa/model/model.py','reference/O-MaMa/descriptors/get_descriptors.py');m['reference_files_sha256']={str(Q/n):sha(Q/n) for n in refs}
    (R/'manifest.json').write_text(json.dumps(m,indent=2));print('FROZEN_REPEAT_PREPARED',flush=True)
if __name__=='__main__':main()
