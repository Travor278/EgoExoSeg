from build_temporal import build
def record(i,frame):
    sid='source'+str(i);key=sid+'_'+str(frame)
    return {'video_id':key,'video_path':[f't/c1/{frame}.jpg'],'prompt':{'first_frame_image':f't/c2/{i}.jpg','first_frame_anns':{'0':{'segmentation':{'size':[1,1],'counts':'01'}}}},'objects':{'0':{'segmentation':{'size':[1,1],'counts':str(frame)}}},'candidate_meta':{'take':'t','source_object_id':'cup_0','target_gt_cam':'c1','prompt_exo_cam':'c2','source_candidate_id':sid,'target_gt_image':f't/c1/{frame}.jpg','prompt_frame_id':str(i),'target_gt_frame_id':str(frame)}}
old={v['video_id']:v for v in (record(1,1),record(1,2),record(2,3),record(2,4))}
a,m=build(old);assert len(a)==4 and m['new_independent_takes']==0
for x in a.values():
    meta=x['candidate_meta'];assert x['objects']['0']['segmentation']==old[meta['target_gt_original_record']]['objects']['0']['segmentation'];assert x['prompt']==next(v['prompt'] for v in old.values() if v['candidate_meta']['source_candidate_id']==meta['source_candidate_id'])
assert build(old)[0]==a
print('EXPANDED_PAIR_DISJOINTNESS_AND_ORIGINAL_MASK_PRESERVATION_PASS')
