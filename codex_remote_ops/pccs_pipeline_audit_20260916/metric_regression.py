"""Exercise the actual evaluator on a case separating new, positive-only and last-object means."""
from pathlib import Path
import json,os,sys
R=Path(__file__).parent;F=R.parent/'pccs_gain_full_20260915';sys.path[:0]=[str(F/'code/mmengine'),str(F/'code')]
from mmengine.logging import MMLogger
from projects.v2sam_pccs.evaluation.pccs_metric import PCCSMetric,PCCSFrameLevelMetric,compute_location_score,compute_boundary_f_measure
import numpy as np
MMLogger.get_instance('pccs_metric_audit')
def row(x):
 d={'IoU':x,'Dice':x,'shape_acc':x,'location_score':1-x,'best_expert':'fusion'}
 for e in ('visual','anchor','fusion'):d.update({f'iou_{e}':x,f'dice_{e}':x,f'shape_acc_{e}':x,f'location_score_{e}':1-x})
 return d
records=[[row(0.),row(1.)],[row(.2)]]
frame=PCCSFrameLevelMetric(collect_device='cpu',enable_vis=False,diagnostic_output=None).compute_metrics(records)
obj=PCCSMetric(collect_device='cpu',enable_vis=False,diagnostic_output=None).compute_metrics(records)
assert abs(frame['PCCS_IoU']-.35)<1e-12 and abs(obj['PCCS_IoU']-.4)<1e-12
assert frame['num_image_pairs']==2 and frame['num_objects']==3
mask=np.zeros((32,32),dtype=np.uint8);mask[8:16,8:16]=1
assert compute_boundary_f_measure(mask,mask)==1 and compute_location_score(mask,mask)==0
out={'state':'passed','actual_evaluator_frame_iou':frame['PCCS_IoU'],'actual_evaluator_object_iou':obj['PCCS_IoU'],'last_object_wrong_result':.6,'positive_only_wrong_result':.6,'pairs':2,'objects':3,'zero_iou_retained':True,'identical_mask_boundary':1,'identical_mask_location_error':0,'location_definition':'mean coordinates on largest external contour, normalized by image diagonal; not mask-area centroid; follows upstream'}
(R/'metric_regression.json').write_text(json.dumps(out,indent=2));print('METRIC_REGRESSION_PASS')
