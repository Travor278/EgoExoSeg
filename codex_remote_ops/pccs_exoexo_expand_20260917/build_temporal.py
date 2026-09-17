"""Expand only pairings: keep approved source and original-resolution target GT intact."""
from pathlib import Path
from collections import defaultdict,Counter
import json,copy,hashlib
def build(old):
    groups=defaultdict(dict);prompts={};existing=set();conflicts=[]
    for k,v in old.items():
        m=v['candidate_meta'];sid=m['source_candidate_id'];g=(m['take'],m['source_object_id'],m['target_gt_cam']);path=m['target_gt_image'];obj=next(iter(v['objects'].values()))
        if path in groups[g] and groups[g][path]['objects']['0']['segmentation']!=obj['segmentation']:conflicts.append((g,path))
        groups[g][path]=v
        if sid in prompts:assert prompts[sid]['prompt']==v['prompt'],'Source prompt not frozen'
        prompts[sid]=v;existing.add((sid,path))
    assert not conflicts,('Conflicting target annotations',conflicts[:3]);out={}
    for sid,v in sorted(prompts.items()):
        m=v['candidate_meta'];g=(m['take'],m['source_object_id'],m['target_gt_cam']);options=sorted([p for p in groups[g] if (sid,p) not in existing],key=lambda p:(int(Path(p).stem),p));n=min(16,len(options));chosen=[options[i*(len(options)-1)//(n-1)] for i in range(n)] if n>1 else options
        for path in chosen:
            gt=groups[g][path];frame=Path(path).stem;key='expanded_exoexo_v1_'+hashlib.sha256((sid+'|'+path).encode()).hexdigest()[:32];a=copy.deepcopy(v);obj=copy.deepcopy(gt['objects']['0']);obj.update(video_id=key,video_path=[path]);a.update(video_id=key,video_path=[path],objects={'0':obj});a['candidate_meta']={**m,'target_gt_frame_id':frame,'target_gt_image':path,'prompt_target_absdiff':abs(int(frame)-int(m['prompt_frame_id'])),'temporal_mode':'expanded_additional_annotated_frame','construction':'time-quantile-max16-v1','source_candidate_id':sid,'target_gt_original_record':gt['video_id'],'new_independent_take':False}
            assert a['prompt']==v['prompt'] and a['objects']['0']['segmentation']==gt['objects']['0']['segmentation'];assert (sid,path) not in existing;assert m['prompt_exo_cam']!=m['target_gt_cam'];out[key]=a
    manifest={'protocol':'time-quantile-max16-v1','original_pairs':len(old),'additional_pairs':len(out),'combined_old_and_new_pairs':len(old)+len(out),'takes':len({v['candidate_meta']['take'] for v in out.values()}),'take_scoped_object_instances':len({(v['candidate_meta']['take'],v['candidate_meta']['source_object_id']) for v in out.values()}),'original_take_scoped_object_instances':len({(v['candidate_meta']['take'],v['candidate_meta']['source_object_id']) for v in old.values()}),'original_unique_object_name_strings':len({v['candidate_meta']['source_object_id'] for v in old.values()}),'source_prompts':len({v['candidate_meta']['source_candidate_id'] for v in out.values()}),'target_images':len({v['video_path'][0] for v in out.values()}),'new_independent_takes':0,'new_prompt_masks':0,'new_ground_truth_masks':0,'original_prompt_and_target_mask_bytes_preserved':True,'overlap_with_old_pair_keys':0,'selection':'At most16 new target frames per approved source prompt, evenly spaced through existing annotated times; no method scores or outcomes consulted','scope':'Frozen-method temporal recombination stress test, NOT increased scene diversity, NOT official full Ego-Exo4D benchmark','take_counts':dict(Counter(v['candidate_meta']['take'] for v in out.values()))}
    return out,manifest
if __name__=='__main__':
    r=Path(__file__).parent;inputs=json.loads((r/'private_inputs.json').read_text(encoding='utf-8-sig'));out,m=build(inputs['old']);p=r/'temporal_expanded_annotation.json';p.write_text(json.dumps(out));m['annotation_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(r/'temporal_expansion_manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
