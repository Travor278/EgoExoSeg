"""Cross-Python source identity: ignore positions, docstrings and empty type_params."""
import ast,json,hashlib
def canonical(node):
 if node is Ellipsis:return {'literal':'Ellipsis'}
 if isinstance(node,bytes):return {'bytes':node.hex()}
 if isinstance(node,ast.AST):
  out={'node':type(node).__name__}
  for k,v in ast.iter_fields(node):
   if k=='type_params' and not v:continue
   if k=='body' and isinstance(v,list) and v and isinstance(v[0],ast.Expr) and isinstance(v[0].value,ast.Constant) and isinstance(v[0].value.value,str):v=v[1:]
   out[k]=canonical(v)
  return out
 if isinstance(node,list):return [canonical(x) for x in node]
 return node
def identity(src,name=None):
 n=ast.parse(src)
 if name:n=next(n for n in ast.walk(n) if isinstance(n,ast.FunctionDef) and n.name==name)
 return hashlib.sha256(json.dumps(canonical(n),sort_keys=True,ensure_ascii=True).encode()).hexdigest()
