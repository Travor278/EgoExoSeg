from pathlib import Path
import json,hashlib,shutil,yaml,time
R=Path(__file__).parent;Q=R.parent/'pccs_candidate_quality_20260916';G=R.parent/'pccs_gain_20260915/gate_training'
def add_phase(manifest,phase,ann,cfg,reference,refphase):
    spec={'pairs':len(ann),'objects':sum(len(v['objects']) for v in ann.values()),'shards':[],'reference_root':str(reference),'reference_phase':refphase};keys=list(ann)
    cfg=json.loads(json.dumps(cfg));cfg['models']['dinov3']['repo']=str(R/'code/third_parts/dinov3');cfg['runtime']['ports']['exo2ego']=29791
    for rank in range(4):
        a={k:ann[k] for k in keys[rank::4]};ap=R/f'{phase}_rank{rank}.json';ap.write_text(json.dumps(a));c=json.loads(json.dumps(cfg));c['data']['annotations']['exo2ego']=str(ap);cp=R/f'{phase}_rank{rank}.yaml';cp.write_text(yaml.safe_dump(c,sort_keys=False));spec['shards'].append({'annotation':str(ap),'config':str(cp),'pairs':len(a),'objects':sum(len(v['objects']) for v in a.values())})
    manifest['phases'][phase]=spec
def main():
    assert not (R/'code').exists();shutil.copytree(Q/'code',R/'code',symlinks=True)
    from native_bridge import patch_metric
    patch_metric(R/'code')
    old=json.loads((Q/'manifest.json').read_text());plan=json.loads((G/'plan.json').read_text());allann=json.loads((G/'combined.json').read_text());cfg=yaml.safe_load((Q/'screen_runtime.yaml').read_text())
    m={'created_at':time.time(),'phases':{},'assets':old['assets'],'pretrained_omama_used':False,'new_encoder_passes':0,'candidate_change':False,'training':'PCCS cycle-confidence calibration only; native features; four-fold take-held-out predictions','selection':'positive TRAIN take-OOF and held calibration delta, maximum calibration; freeze before target; report matched feature ablations'}
    add_phase(m,'smoke',json.loads((Q/'smoke.json').read_text()),cfg,Q,'smoke')
    train={k:allann[k] for k in plan['splits']['fit']['keys']};cal={k:allann[k] for k in plan['splits']['calibration']['keys']};assert len(train)==384 and len(cal)==128
    def takes(a):return {(v['prompt']['first_frame_image'][0] if isinstance(v['prompt']['first_frame_image'],list) else v['prompt']['first_frame_image']).split('/')[0] for v in a.values()}
    assert takes(train).isdisjoint(takes(cal));m['train_takes']=sorted(takes(train));m['calibration_takes']=sorted(takes(cal))
    add_phase(m,'train',train,cfg,Q,'screen');add_phase(m,'calibration',cal,cfg,Q,'calibration')
    confirm=json.loads((G/'holdout512.json').read_text());assert len(confirm)==512 and sum(len(v['objects']) for v in confirm.values())==965
    assert takes(confirm).isdisjoint(takes(train)|takes(cal))
    testcfg=yaml.safe_load((R.parent/'pccs_corrected_exoego_20260916/full_runtime.yaml').read_text())
    assert testcfg['models']['experts']['exo2ego']==cfg['models']['experts']['exo2ego']
    validate_images(confirm,testcfg)
    add_phase(m,'exo2ego',confirm,testcfg,Q,'smoke')
    m['test_scope']='Previously evaluated fixed 512-pair/965-object/64-take Exo2Ego holdout; never used for this fitting; not a new blind test or full test.'
    m['frozen_cycle']=json.loads((R.parent/'pccs_dense_context_20260917/selection.json').read_text())['selected']
    m['code_sha256']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ('fixed_context.py','dense_context.py','worker.py','randomness.py','native_bridge.py','policies.py','code/projects/v2sam_pccs/evaluation/pccs_metric.py')};(R/'manifest.json').write_text(json.dumps(m,indent=2));print('DENSE_PREPARED',flush=True)
def validate_images(ann,cfg):
    root=Path(cfg['data']['images']);paths=set()
    for row in ann.values():
        for field in (row['prompt']['first_frame_image'],row['video_path']):
            paths.update(field if isinstance(field,list) else [field])
    missing=[p for p in paths if not (root/p).is_file()]
    assert not missing,('Missing experiment images',str(root),missing[:5])
    return {'image_root':str(root),'checked_images':len(paths),'missing':0}

if __name__=='__main__':main()
