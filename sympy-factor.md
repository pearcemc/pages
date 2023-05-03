```python
import sympy
import sympy as sy
import sympy.core.exprtools 
import sympy.core.numbers
from sympy.core.traversal import preorder_traversal
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import (common_prefix, common_suffix,
        variations, iterable, is_sequence)
```


```python
N = sy.Symbol("N", integer=True, positive=True)
P = sy.Symbol("P", integer=True, positive=True)

X = sy.MatrixSymbol("X", N, P)
Y = sy.MatrixSymbol("Y", N, P)
Z = sy.MatrixSymbol("Z", P, N)

A = sy.MatrixSymbol("A", N, N)
B = sy.MatrixSymbol("B", N, N)
C = sy.MatrixSymbol("C", N, N)
D = sy.MatrixSymbol("D", N, N)

example_a = X * Z + Y * Z              #  Expect: (X + Y)Z
example_b =  X * X.T + Y * X.T         #  Expect: (X + Y)X.T
example_c =  X * X.T + 3 * Y * X.T     #  Expect: (X + 3Y)X.T
example_d =  3 * X * Z + Y * 3 * Z     #  Expect: 3(X + Y)Z

example_e = A*C + B*C + B*D + A*D      #  Expect: (A + B)(C + D)
example_f = A*D + D**2                 #  Expect: (A + D)D
example_g = A*C + B*C + C
```


```python
def mask_matrix_expr(expr):
  """Masks all matrix symbols and simple unary expressions with non-commutative Symbols."""

  if not expr.is_Matrix:
    return expr, {expr: expr}

  if expr.is_symbol:
    new_sym = sy.Symbol(f'mask_{expr.name}', commutative=False)
    return new_sym, {expr: new_sym}
  
  if len(expr.args)==1 and sum(len(a.free_symbols) for a in expr.args)==1:
    new_sym = sy.Symbol(f'mask_{expr}', commutative=False)
    return new_sym, {expr: new_sym}  

  mapping = dict()
  new_args = []  
  for arg in expr.args:
    new_arg, sub_map = mask_matrix_expr(arg)
    for old, new in sub_map.items():
      if old in mapping.keys():
        new_arg = new_arg.subs(mapping)
      else:
        mapping[old] = new
    new_args.append(new_arg)

  try:
    if expr.is_Add:
      expr = sy.Add(*new_args)
    if expr.is_Mul:
      expr = sy.Mul(*new_args)
    if isinstance(expr, sy.MatPow):
      expr = sy.Pow(*new_args)    
  except Exception as err:
    for na in new_args:
      print(f"{type(expr)}, {type(na)}, {na}")
    raise err

  return expr, mapping


def unmask_matrix_expr(expr, unmapping):
  """Restores matrices in a masked expression.
  Unmapping should have mask keys and matrix values.
  """  

  if expr.is_symbol:
    return unmapping[expr]

  new_args = []
  for arg in expr.args:
    if arg.is_symbol:
      try:
        new_arg = unmapping[arg]
      except KeyError as err:
        print(unmapping)
        raise err
    else:
      new_arg = unmask_matrix_expr(arg, unmapping)
    new_args.append(new_arg)

  for i, arg in enumerate(new_args):
    #if not arg.is_Matrix:
    if isinstance(arg, sympy.core.numbers.One):
      rank = None
      for value in new_args:
        if value.is_Matrix and value.is_square:
          rank = value.rows
          break
      if rank is None:
        raise ValueError('Need to introduce identity, but cannot determine shape.')
      I = sy.MatrixSymbol(f'I', rank, rank)
      new_args[i] = I

  try:
    if any(arg.is_Matrix for arg in new_args):
      if expr.is_Add:
        expr = sy.MatAdd(*new_args)
      if expr.is_Mul:
        expr = sy.MatMul(*new_args)
      if isinstance(expr, sy.MatPow):
        expr = sy.MatPow(*new_args)    
  except Exception as err:
    print(f"{type(expr)}")
    for na in new_args:
      print(f"\t{type(na)}, {na}")
    raise err

  return expr


def factor_matrix_expr(expr):
  """Factor for matrix expressions.
  Masks the expression, factors it, then unmasks the results.
  """
  if not expr.is_Matrix:
    return expr.factor()
  masked, mapping = mask_matrix_expr(expr)
  unmapping = {v: k for k,v in mapping.items()}
  soln = masked.factor()
  return unmask_matrix_expr(soln, unmapping)
```


```python
example_a
```




$\displaystyle X Z + Y Z$




```python
factor_matrix_expr(example_a)
```




$\displaystyle \left(X + Y\right) Z$




```python
example_b
```




$\displaystyle X X^{T} + Y X^{T}$




```python
factor_matrix_expr(example_b)
```




$\displaystyle \left(X + Y\right) X^{T}$




```python
example_c
```




$\displaystyle X X^{T} + 3 Y X^{T}$




```python
factor_matrix_expr(example_c)
```




$\displaystyle \left(X + 3 Y\right) X^{T}$




```python
example_d
```




$\displaystyle 3 X Z + 3 Y Z$




```python
factor_matrix_expr(example_d)
```




$\displaystyle 3 \left(X + Y\right) Z$




```python
example_e
```




$\displaystyle A C + A D + B C + B D$




```python
factor_matrix_expr(example_e)
```




$\displaystyle \left(A + B\right) \left(C + D\right)$




```python
example_f
```




$\displaystyle D^{2} + A D$




```python
factor_matrix_expr(example_f)
```




$\displaystyle \left(A + D\right) D$




```python
example_g
```




$\displaystyle A C + B C + C$




```python
factor_matrix_expr(example_g)
```




$\displaystyle \left(A + B + I\right) C$
