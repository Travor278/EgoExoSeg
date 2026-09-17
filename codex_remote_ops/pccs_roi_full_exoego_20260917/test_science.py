import numpy as np
from PIL import Image
from roi_views import crop_view
from science_views import intervene
pixels=np.random.default_rng(71).integers(0,256,(200,300,3),dtype=np.uint8);im=Image.fromarray(pixels);mask=np.zeros((200,300),np.uint8);mask[70:110,140:180]=1;view=crop_view(im,mask,2.)
for arm in ('real','far','rolled'):
    v,meta=intervene(view,im,mask,arm);a=np.asarray(view['image']);b=np.asarray(v['image']);assert np.array_equal(a[view['mask']>0],b[view['mask']>0])
    assert np.array_equal(v['mask'],view['mask']) and v['box']==view['box']
    if arm=='real':assert np.array_equal(a,b)
    else:assert np.any(a!=b);v2,_=intervene(view,im,mask,arm);assert np.array_equal(np.asarray(v2['image']),b)
print('NATURAL_CONTEXT_INTERVENTION_FOREGROUND_AND_SHAM_PARITY_PASS')
