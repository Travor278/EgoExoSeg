import json
from resume_full import complete_prefix
ann={'a':{'objects':{'0':{},'1':{}}},'b':{'objects':{'0':{}}}}
def row(v,o):return json.dumps({'video_id':v,'obj_id':o}).encode()+b'\n'
complete=row('a','0')+row('a','1')
assert complete_prefix(complete,ann)[0]==complete
fixed,info=complete_prefix(row('a','0')+b'{"video_id":',ann);assert fixed==b'' and info['truncated_last_line'] and info['discarded_complete_object_records']==1
fixed,info=complete_prefix(complete+row('b','0').rstrip(),ann);assert fixed==complete+row('b','0') and not info['discarded_incomplete_final_pair']
for bad in (b'{bad}\n'+complete,row('a','0')+row('b','0'),complete+row('a','1')):
    try:complete_prefix(bad,ann)
    except AssertionError:pass
    else:raise AssertionError('Interior corruption accepted')
print('RESUME_PRESERVES_COMPLETE_PAIRS_AND_REJECTS_INTERIOR_CORRUPTION')
