"""Prediction-only, explicit allowlist. No target metrics enter model features."""
import math
EXPERTS=('visual','anchor','fusion')
SCORES=('omama_learned','omama_context_cross_zero','dino2_object','dino2_object_context')
GLOBAL=('prompt_area_ratio','prompt_num_components','pair_num_objects','dino_forward_confidence','agreement_visual_anchor_iou','agreement_visual_fusion_iou','agreement_anchor_fusion_iou')
LOCAL=('area_ratio','num_components','sam_iou_score','sam_iou_margin','sam_object_logit','cycle_dist','dino_backward_confidence','forward_point_inside_ratio','prob_fg_mean','prob_margin_mean','prob_stability_iou','prob_uncertain_ratio')

def local_key(expert,key):
    if key in ('area_ratio','num_components'):return expert+'_'+key
    if key.startswith('prob_'):return 'prob_'+expert+'_'+key[5:]
    return key+'_'+expert

def prediction_inputs(row,scores,names):
    keys=list(GLOBAL)+[local_key(e,k) for e in EXPERTS for k in LOCAL]
    return {'values':{k:row.get(k) for k in keys},'baseline_expert':row['best_expert'],'scores':{k:scores[k] for k in SCORES},'names':list(names)}

def vector(inputs,index):
    names=inputs['names'];candidate=names[index];baseline=inputs['baseline_expert']
    assert index>0 and names[0]=='baseline' and candidate in EXPERTS
    x=[];columns=[]
    def add(k,v):
        columns.append(k)
        if v is None:x.append(float('nan'))
        else:
            v=float(v);x.append(v if math.isfinite(v) else float('nan'))
    for e in EXPERTS:add('candidate_is_'+e,candidate==e);add('baseline_is_'+e,baseline==e)
    for key in SCORES:
        s=inputs['scores'][key];c,b=s[index],s[0]
        add(key+'_candidate',c);add(key+'_baseline',b);add(key+'_delta',c-b);add(key+'_margin_to_others',c-max(s[j] for j in range(len(s)) if j!=index))
    for k in GLOBAL:add(k,inputs['values'][k])
    for k in LOCAL:
        c=inputs['values'][local_key(candidate,k)];b=inputs['values'][local_key(baseline,k)]
        add(k+'_candidate',c);add(k+'_baseline',b);add(k+'_delta',None if c is None or b is None else c-b)
    return x,columns
