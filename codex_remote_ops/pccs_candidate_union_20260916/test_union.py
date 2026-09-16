from union_core import route,feature_vector
cs=[{'id':'old','scores':[.6,.6],'empty':False,'provenance':['baseline/fusion']},{'id':'new','scores':[.7,.7],'empty':False,'provenance':['reliable_points/anchor']}]
assert route(cs,'old')=='new'
cs[1]['scores']=[.7,.62];assert route(cs,'old')=='old'
cs[1]['scores']=[.61,.61];assert route(cs,'old')=='old'
cs[1]['scores']=[.7,.7];assert len(feature_vector(cs,'old','new'))==29
print('UNION_TESTS_PASS')
