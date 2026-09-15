"""Check routing independently of model, target labels, and CUDA imports."""
from pathlib import Path
import ast,types,numpy as np
tree=ast.parse((Path(__file__).parent/'evaluate_confirmation.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select')
ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'select','exec'),ns)
select=ns['select'];names=['baseline','visual','anchor']
assert select([.5,.55,.1],names,[True]*3,.05)=='baseline'
assert select([.5,.5501,.1],names,[True]*3,.05)=='visual'
assert select([.5,.9,.6],names,[True,False,True])=='anchor'
assert select([.5,.5,.5],names,[True]*3)=='baseline'
assert select([0.,0.,0.],names,[False]*3)=='baseline'
assert select([.9,.1,.2],names,[False,True,True])=='anchor'
try:select([.1,float('nan'),.2],names,[True]*3)
except ValueError:pass
else:raise AssertionError('Nonfinite scores must fail')
print('7 routing checks passed; function has no target-label input')
