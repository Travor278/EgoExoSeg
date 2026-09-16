"""Preserve original final output and add independently generated candidates."""
import hashlib
import numpy as np
EMPTY_SHA=hashlib.sha256(bytes(1024*1024)).hexdigest()
def catalog(records):
    candidates=[];indices={};labels={};max_disagreement=0.
    for source,record in records:
        for i,name in enumerate(record['candidate_names']):
            expert=record['pccs_expert'] if name=='baseline' else name
            ident=record['mask_sha256'][expert];scores=[float(record['scores'][m][i]) for m in ('canonical','native_interp')]
            if not np.isfinite(scores).all():raise ValueError('Nonfinite matching scores')
            if ident in indices:
                c=candidates[indices[ident]];max_disagreement=max(max_disagreement,float(np.max(np.abs(np.asarray(scores)-c['scores']))));c['provenance'].append(source+'/'+expert)
                assert np.allclose(labels[ident],record['metrics'][expert],atol=1e-7,rtol=0)
                continue
            indices[ident]=len(candidates);candidates.append({'id':ident,'scores':scores,'empty':ident==EMPTY_SHA,'provenance':[source+'/'+expert]});labels[ident]=record['metrics'][expert]
    # Prefer original-bank values for duplicate masks, never average across runs.
    assert max_disagreement<1e-4,('Cached matching scores disagree for identical masks',max_disagreement)
    base=records[0][1];fallback=base['mask_sha256'][base['consensus_expert']]
    assert fallback in indices
    return candidates,labels,fallback,max_disagreement
def route(candidates,fallback,margin=.05):
    byid={c['id']:c for c in candidates};base=byid[fallback]
    allowed=[c for c in candidates if not c['empty']]
    if not allowed or all(all(s==0 for s in c['scores']) for c in candidates):return fallback
    choices=[]
    for g in range(2):
        best=max(allowed,key=lambda c:c['scores'][g])
        choices.append(best['id'] if base['empty'] or best['scores'][g]>base['scores'][g]+margin else fallback)
    return choices[0] if choices[0]==choices[1] else fallback
def feature_vector(candidates,fallback,candidate_id):
    # No labels, metrics or ground-truth information are accepted here.
    c=next(c for c in candidates if c['id']==candidate_id);b=next(c for c in candidates if c['id']==fallback)
    s=np.asarray(c['scores']);sb=np.asarray(b['scores']);other=np.asarray([x['scores'] for x in candidates if x['id']!=candidate_id and not x['empty']])
    best_other=other.max(0) if len(other) else sb
    original=any(p.startswith('baseline/') for p in c['provenance'])
    vector=[*s,*sb,*(s-sb),*(s-best_other),float(abs(s[0]-s[1])),float(abs(sb[0]-sb[1])),float(original),float(b['empty']),float(len(candidates))]
    for source in ('reliable_points','residual005','residual010','residual025'):
        vector.append(float(any(p.startswith(source+'/') for p in c['provenance'])))
    for expert in ('visual','anchor','fusion'):vector.append(float(any(p.endswith('/'+expert) for p in c['provenance'])))
    vector.extend(c.get('geometry',[0.]*9))
    return vector
