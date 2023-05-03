```python
import sympy
```


```python
N = sympy.Symbol("N", integer=True, positive=True)
P = sympy.Symbol("P", integer=True, positive=True)

X = sympy.MatrixSymbol("X", N, P)
Y = sympy.MatrixSymbol("Y", N, P)
Z = sympy.MatrixSymbol("Z", P, N)

A = sympy.MatrixSymbol("A", N, N)
B = sympy.MatrixSymbol("B", N, N)
C = sympy.MatrixSymbol("C", N, N)
D = sympy.MatrixSymbol("D", N, N)

r = sympy.Symbol('r')

example_a = X * Z + Y * Z              #  Expect: (X + Y)Z
example_b =  X * X.T + Y * X.T         #  Expect: (X + Y)X.T
example_c =  X * X.T + 3 * Y * X.T     #  Expect: (X + 3Y)X.T
example_d =  3 * X * Z + Y * 3 * Z     #  Expect: 3(X + Y)Z

example_e = A*C + B*C + B*D + A*D      #  Expect: (A + B)(C + D)
example_f = A*D + D**2                 #  Expect: (A + D)D

example_g = A*C + B*C + C              #  Expect: (A + B + I)C
example_h = A*C + B*C + 2*C            #  Expect: (A + B + 2I)C
example_i = A*C + r*C                  #  Expect: (A + rI)C
```


```python
def mask_matrix_expr(expr):
  """Masks all matrix symbols and simple unary expressions with non-commutative Symbols."""

  if not expr.is_Matrix:
    return expr, {expr: expr}

  if expr.is_symbol:
    new_sym = sympy.Symbol(f'mask_{expr.name}', commutative=False)
    return new_sym, {expr: new_sym}
  
  if len(expr.args)==1 and sum(len(a.free_symbols) for a in expr.args)==1:
    new_sym = sympy.Symbol(f'mask_{expr}', commutative=False)
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
      expr = sympy.Add(*new_args)
    if expr.is_Mul:
      expr = sympy.Mul(*new_args)
    if isinstance(expr, sympy.MatPow):
      expr = sympy.Pow(*new_args)    
  except Exception as err:
    for na in new_args:
      print(f"{type(expr)}, {type(na)}, {na}")
    raise err

  return expr, mapping


def symbolic_identity_(expressions):
  rank = None
  for value in expressions:
    if value.is_Matrix and value.is_square:
      rank = value.rows
      break
  if rank is not None:
    return sympy.MatrixSymbol(f'I', rank, rank)


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

  value_is_matrix = any(arg.is_Matrix for arg in new_args)    
  # The arguments may contain newly introduced scalar values
  # e.g. YX + 2X -> (mask_Y + 2)mask_X, which in matrix format needs to be (Y + 2I)X
  if value_is_matrix and expr.is_Add:
    for i, arg in enumerate(new_args):
      if not arg.is_Matrix:
        I = symbolic_identity_(new_args)
        if I is None:
          raise ValueError('Identity matrix required, but cannot be determined.')
        new_args[i] = arg * I

  try:
    if value_is_matrix:
      if expr.is_Add:
        expr = sympy.MatAdd(*new_args)
      if expr.is_Mul:
        expr = sympy.MatMul(*new_args)
      if isinstance(expr, sympy.MatPow):
        expr = sympy.MatPow(*new_args)    
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




```python
example_h
```




$\displaystyle A C + B C + 2 C$




```python
factor_matrix_expr(example_h)
```




$\displaystyle \left(A + B + 2 I\right) C$




```python
example_i
```




$\displaystyle r C + A C$




```python
factor_matrix_expr(example_i)
```




$\displaystyle \left(r I + A\right) C$


