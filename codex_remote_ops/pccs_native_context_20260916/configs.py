from native_context import DEFAULTS
CONFIGS={
 'baseline':{},
 'zero':{**DEFAULTS,'cycle':True,'review':True,'strength':0.},
 'cycle':{**DEFAULTS,'cycle':True,'review':False},
 'review':{**DEFAULTS,'cycle':False,'review':True},
 'both':{**DEFAULTS,'cycle':True,'review':True},
 'wrong_context':{**DEFAULTS,'cycle':True,'review':True},
}
ELIGIBLE=('cycle','review','both')
