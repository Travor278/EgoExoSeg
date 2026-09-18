from common import *
import shutil,yaml
FROZEN=('encoder_contract.py','roi_views.py','roi_evidence.py','roi_policy.py','dense_context.py','policies.py','native_bridge.py','randomness.py')
def main():
    assert not (R/'manifest.json').exists(),'Fresh shared bank only'
    old=read(F/'manifest.json')
    for n,h in old['code_sha256'].items():assert sha(F/n)==h
    for n in FROZEN:shutil.copy2(F/n,R/n)
    (R/'code').symlink_to(F/'code',target_is_directory=True);shutil.copy2(F/'selection.json',R/'selection.json')
    assert sha(R/'selection.json')==old['frozen_selection_sha256']
    for a in old['assets'].values():assert sha(a['path'])==a['sha256']
    assert shutil.disk_usage(R).free>100*1024**3
    hist=read(C/'manifest.json')
    for n in ('aligned_model.py','cache.py'):assert sha(C/n)==hist['frozen_files'][n]
    m={k:old[k] for k in ('assets','encoder_contract','frozen_cycle','frozen_selection_sha256','full_annotation_sha256')}
    m.update(phases={},scope='Fresh shared candidate generation; NOT bit-identical reconstruction of previous full1. All methods share saved new masks; no target fitting.',reference_code_sha256={n:sha(C/n) for n in ('aligned_model.py','cache.py')},reference_weights={n:sha(C/'weights'/n) for n in ('omama_exo2ego.pt','dinov2_vitb14_reg4_pretrain.pth')})
    for phase in ('smoke1','full1'):
        spec=old['phases'][phase];m['phases'][phase]={'pairs':spec['pairs'],'objects':spec['objects'],'shards':[]}
        for rank,s in enumerate(spec['shards']):
            ann=read(s['annotation']);ap=R/f'{phase}_rank{rank}.json';write(ap,ann);cfg=yaml.safe_load(Path(s['config']).read_text());cfg['data']['annotations']['exo2ego']=str(ap);cp=R/f'{phase}_rank{rank}.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False));m['phases'][phase]['shards'].append({**s,'annotation':str(ap),'annotation_sha256':sha(ap),'config':str(cp)})
    m['code_sha256']={n:sha(R/n) for n in FROZEN+('worker.py','reference.py','evaluate.py','controller.py','code/projects/v2sam_pccs/evaluation/pccs_metric.py')}
    write(R/'manifest.json',m);print('NEW_SHARED_BANK_PREPARED',flush=True)
if __name__=='__main__':main()
