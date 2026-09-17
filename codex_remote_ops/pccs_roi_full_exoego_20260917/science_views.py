"""Frozen-model intervention: retain target and source foreground; alter source context only."""
import hashlib
import numpy as np
from PIL import Image,ImageFilter
from roi_views import crop_view,encode_views
from roi_evidence import pair_stats,KEYS
ARMS=('real','far','rolled')

def intervene(view,source_image,source_mask,arm):
    pixels=np.asarray(view['image']).copy();mask=view['mask']>0
    # Keep an 8-native-input-pixel band near the object; same boundary in both controls.
    guard=np.asarray(Image.fromarray((mask*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(17)))>0
    editable=~guard;meta={'arm':arm,'editable_fraction':float(editable.mean()),'guard_pixels':8}
    if arm=='real':return view,meta
    size=view['image'].width
    if arm=='rolled':donor=np.roll(pixels,(size//2,size//2),(0,1));meta['offset']=[size//2,size//2]
    else:
        assert arm=='far';im=source_image.convert('RGB');w,h=im.size;sm=np.asarray(source_mask)>0
        if sm.shape!=(h,w):sm=np.asarray(Image.fromarray(sm.astype(np.uint8)).resize((w,h),Image.Resampling.NEAREST))>0
        side=min(size,w,h);locations=[(x,y,x+side,y+side) for y in (0,(h-side)//2,h-side) for x in (0,(w-side)//2,w-side)]
        cx=(view['box'][0]+view['box'][2])/2;cy=(view['box'][1]+view['box'][3])/2
        def order(b):
            x0,y0,x1,y1=b;overlap=float(sm[y0:y1,x0:x1].mean());distance=((x0+x1)/2-cx)**2+((y0+y1)/2-cy)**2;return overlap,-distance,b
        box=min(locations,key=order);donor=np.asarray(im.crop(box).resize((size,size),Image.Resampling.BILINEAR));meta['donor_box']=box;meta['donor_foreground_fraction']=order(box)[0]
    edited=pixels.copy();edited[editable]=donor[editable];assert np.array_equal(edited[guard],pixels[guard]);meta['changed_pixel_fraction']=float(np.any(edited!=pixels,axis=2).mean());out={**view,'image':Image.fromarray(edited)};return out,meta

def collect(matcher,source_image,target_image,source_mask,masks,smoke=False):
    import time
    start=time.monotonic();views=[];indices={};metadata={}
    for mode,scale in (('local15',1.5),('local20',2.)):
        source=crop_view(source_image,source_mask,scale)
        if source is None:continue
        for arm in ARMS:
            v,meta=intervene(source,source_image,source_mask,arm);indices[(mode,arm,'source')]=len(views);views.append(v);metadata[mode+'/'+arm]=meta
        for e,mask in masks.items():
            v=crop_view(target_image,mask,scale)
            if v is not None:indices[(mode,'target',e)]=len(views);views.append(v)
    encoded=encode_views(matcher,views,4);result={arm:{e:{} for e in masks} for arm in ARMS}
    for mode in ('local15','local20'):
        for arm in ARMS:
            for e in masks:
                values={k:0. for k in KEYS}
                if (mode,'target',e) in indices:
                    qi=indices[(mode,'target',e)];ti=indices[(mode,arm,'source')];values.update(pair_stats(encoded[qi],encoded[ti],views[qi]['mask'],views[ti]['mask']),valid=1.,source_scale=views[ti]['effective_scale'],target_scale=views[qi]['effective_scale'])
                result[arm][e].update({mode+'_'+k:v for k,v in values.items()})
    assert all(np.isfinite(v) for arm in result.values() for e in arm.values() for v in e.values())
    return result,{'source_interventions':metadata,'encoded_views':len(views),'elapsed_seconds':time.monotonic()-start}
