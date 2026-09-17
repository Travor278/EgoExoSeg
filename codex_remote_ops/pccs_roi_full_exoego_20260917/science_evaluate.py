import json,copy
from data_utils import R,read,score
out={'scope':'Frozen seed1 model interventions, unchanged source foreground and target candidates; descriptive mechanism tests, not new method selection. Mean-background results are from prior matched separately-trained control, not interchangeable with frozen-model intervention.' ,'phases':{}}
for phase in ('science_exo2exo','science_exo2ego'):
    rows=read(phase);choices={k:[r['choices'][k] for r in rows] for k in rows[0]['choices']};results={k:score(rows,v) for k,v in choices.items()};contrasts={}
    for mode in ('local15','local20'):
        for arm in ('far','rolled'):
            refs=copy.deepcopy(rows)
            for r,e in zip(refs,choices[arm+'_'+mode]):r['baseline']=e
            contrasts['real minus '+arm+' '+mode]=score(refs,choices['real_'+mode])
    out['phases'][phase]={'methods':results,'real_minus_intervention':contrasts,'real_max_feature_error':max(r['real_max_feature_error'] for r in rows)}
(R/'science_results.json').write_text(json.dumps(out,indent=2));print('SCIENCE_EVALUATION_COMPLETE',flush=True)
