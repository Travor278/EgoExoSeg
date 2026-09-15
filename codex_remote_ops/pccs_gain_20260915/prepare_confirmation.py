from pathlib import Path
from collections import defaultdict
import hashlib,json,re,shutil,yaml
R=Path(__file__).parent;P=R.parent/'context_pccs_20260914'
O=R.parent/'omama_aligned_20260915'
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def take(rec):
    q=rec['prompt']['first_frame_image'];q=q[0] if isinstance(q,list) else q
    return re.search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',q).group()
audit=json.loads((R/'split_audit.json').read_text())
a=json.loads(Path(audit['test']['path']).read_text())
old=json.loads((P/'ablations_20260915/exo2ego_probe64.json').read_text())
excluded={take(v) for v in old.values()};groups=defaultdict(list)
for k,v in a.items():
    if take(v) not in excluded:groups[take(v)].append(k)
takes=sorted((t for t,ks in groups.items() if len(ks)>=8),key=lambda t:sha('gain-confirm-v1-take:'+t))[:64]
assert len(takes)==64
keys=[k for t in takes for k in sorted(groups[t],key=lambda k:sha('gain-confirm-v1-pair:'+k))[:8]]
selected={k:a[k] for k in keys};assert len(selected)==512
conf=yaml.safe_load((P/'runtime.yaml').read_text());image_root=Path(conf['data']['images'])
for rec in selected.values():
    for field in (rec['prompt']['first_frame_image'],rec['video_path']):
        path=field[0] if isinstance(field,list) else field
        assert (image_root/path).is_file(),path
ann=R/'confirmation512.json';ann.write_text(json.dumps(selected))
plan={'stage':'independent O-MaMa confirmation, no fitting','dataset':'exo2ego','pairs':512,
      'objects':sum(len(v['objects']) for v in selected.values()),'takes':takes,'excluded_exploratory_takes':sorted(excluded),
      'take_overlap':[],'annotation':str(ann),'annotation_sha256':hashlib.sha256(ann.read_bytes()).hexdigest(),
      'primary_methods':{'learned_object_only_top1':'omama_context_cross_zero; nonempty argmax, baseline first on ties',
                         'omama_margin005':'omama_learned; replace baseline only for cosine advantage > 0.05'},
      'secondary_methods':['omama_top1'],'candidate_generation':'once, upstream inference seed0; O-MaMa scoring seed42; all methods share byte-identical bank',
      'selection':'64 hash-selected takes, 8 hash-selected pairs/take; target outcomes not consulted',
      'official_validation_images_available':False,'no_test_fitting':True}
(R/'confirmation_plan.json').write_text(json.dumps(plan,indent=2))
code=R/'code'
if code.exists():raise RuntimeError('Independent code already exists')
shutil.copytree(P/'code',code,symlinks=True)
shutil.copy2(R/'candidate_bank.py',code/'projects/v2sam_pccs/candidate_bank.py')
p=code/'projects/v2sam_pccs/evaluation/pccs_metric.py';s=p.read_text()
s=s.replace('from ..context_consistency import select_context','from ..context_consistency import select_context\nfrom ..candidate_bank import save_bank')
needle='                context_fields = {}\n';assert s.count(needle)==1
s=s.replace(needle,"                bank_file = save_bank(raw_prompt_mask_j, {k:candidate_masks[k] for k in ('visual','anchor','fusion')}, best_pred_mask, video_id, object_ids[j] if object_ids is not None else str(j))\n"+needle)
s=s.replace('                        **context_fields,','                        **context_fields,\n                        bank_file=bank_file,')
p.write_text(s)
conf['data']['annotations']['exo2ego']=str(ann)
conf['models']['dinov3']['repo']=str(code/'third_parts/dinov3')
conf['runtime']['ports']['exo2ego']=29917
(R/'confirmation_runtime.yaml').write_text(yaml.safe_dump(conf,sort_keys=False))
for name in ('weights','reference'):(R/name).symlink_to(O/name,target_is_directory=True)
shutil.copy2(O/'aligned_model.py',R/'aligned_model.py')
plan['code_metric_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'confirmation_plan.json').write_text(json.dumps(plan,indent=2))
print(json.dumps({k:v for k,v in plan.items() if k not in ('takes','excluded_exploratory_takes')},indent=2))
