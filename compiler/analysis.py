import math
from dataclasses import replace
from emmy.compiler.ir.expr import Literal
from llvmlite import ir
I32 = ir.IntType(32)
STATIC_SHARED_LIMIT = 48 * 1024
INDEX_OPS = {'+', '-', '*', '/', '//', '%', '&', '|', '^', '<<', '>>', '==', '!=', '<', '<=', '>', '>=', '&&', '||'}

def kind(value):
    return type(value).__name__

def walk(body):
    for statement in body:
        yield statement
        for attr in ('body', 'else_body', 'combine_states'):
            yield from walk(getattr(statement, attr, ()))

def free_variables(expression):
    if isinstance(expression, int):
        return set()
    tag = kind(expression)
    if tag == 'Var':
        return {expression.name}
    if tag in ('Literal', 'Builtin'):
        return {expression.name} if tag == 'Builtin' else set()
    if tag == 'CastExpr':
        return free_variables(expression.expr)
    if tag == 'FlatIndex':
        return index_variables(expression.index)
    if tag == 'FuncCallExpr':
        return index_variables(expression.args)
    if tag == 'BinaryExpr':
        if expression.op not in INDEX_OPS:
            return None
        children = (expression.left, expression.right)
    elif tag == 'TernaryExpr':
        children = (expression.cond, expression.if_true, expression.if_false)
    else:
        return None
    return index_variables(children)

def index_variables(indices):
    result = set()
    for index in indices:
        names = free_variables(index)
        if names is None:
            return None
        result.update(names)
    return result

def substitute(expression, names):
    if isinstance(expression, int) or kind(expression) == 'Literal':
        return expression
    if kind(expression) == 'Var':
        return Literal(0, dtype='int') if expression.name in names else expression
    fields = {'BinaryExpr': ('left', 'right'), 'TernaryExpr': ('cond', 'if_true', 'if_false')}.get(kind(expression))
    if fields:
        return replace(expression, **{name: substitute(getattr(expression, name), names) for name in fields})
    raise ValueError('unsupported coordinate expression')

def canonical(expression):

    def frozen(value):
        return tuple(sorted(value.items(), key=repr))
    if isinstance(expression, int):
        return {(): expression} if expression else {}
    if kind(expression) == 'Literal':
        if not isinstance(expression.value, int):
            raise ValueError('noninteger coordinate')
        return canonical(expression.value)
    if kind(expression) == 'Var':
        return {('var', expression.name): 1}
    if kind(expression) == 'TernaryExpr':
        return {('select', *(frozen(canonical(getattr(expression, f))) for f in ('cond', 'if_true', 'if_false'))): 1}
    if kind(expression) != 'BinaryExpr':
        raise ValueError('unsupported coordinate')
    left, right = (canonical(expression.left), canonical(expression.right))
    if expression.op in ('+', '-'):
        return add_terms(left, *((term, coefficient * (1 if expression.op == '+' else -1)) for term, coefficient in right.items()))
    if expression.op == '*':
        if not left or not right:
            return {}
        for scalar, polynomial in ((left, right), (right, left)):
            if set(scalar) <= {()}:
                return {term: coefficient * scalar[()] for term, coefficient in polynomial.items()}
        raise ValueError('non-affine coordinate product')
    if expression.op in ('/', '//', '%', '<', '<=', '>', '>=', '==', '!=', '&', '|', '^', '<<', '>>'):
        if expression.op in ('/', '//', '%') and (not right):
            raise ValueError('zero divisor')
        if expression.op in ('/', '//') and right == {(): 1}:
            return left
        if not left and expression.op in ('/', '//', '%'):
            return {}
        return {(expression.op, frozen(left), frozen(right)): 1}
    raise ValueError('unsupported coordinate operator')

def add_terms(base, *terms):
    result = dict(base)
    for term, coefficient in terms:
        result[term] = result.get(term, 0) + coefficient
    return {term: coefficient for term, coefficient in result.items() if coefficient}

def difference(left, right):
    return add_terms(left, *((term, -coefficient) for term, coefficient in right.items()))

def constant_difference(left, right):
    value = difference(left, right)
    return value.get((), 0) if set(value) <= {()} else None

def known_multiple(expression):
    if isinstance(expression, int):
        return abs(expression)
    if kind(expression) == 'Literal':
        return abs(expression.value) if isinstance(expression.value, int) else 1
    if kind(expression) != 'BinaryExpr':
        return 1
    left, right = (known_multiple(expression.left), known_multiple(expression.right))
    if expression.op in ('+', '-'):
        return math.gcd(left, right)
    if expression.op == '*':
        return left * right
    if expression.op == '%':
        return math.gcd(left, right)
    if expression.op in ('/', '//'):
        divisor = canonical(expression.right)
        if set(divisor) == {()} and divisor[()] > 0 and (left % divisor[()] == 0):
            return left // divisor[()]
    return 1

def integer(value):
    return ir.Constant(I32, int(value))

def uses(statement):
    tag = kind(statement)
    if tag == 'Write':
        return set(statement.values)
    if tag == 'Assign':
        return set(statement.args)
    if tag == 'Accum':
        used = {statement.value}
        if statement.base is not None and statement.base != statement.name:
            used.add(statement.base)
        return used
    if tag in ('WarpShuffle', 'TreeHalve'):
        return set(statement.state) | set(statement.state_b)
    if tag == 'Select':
        return {branch.value for branch in statement.branches}
    if tag == 'Carry':
        return {statement.value}
    if tag == 'Pre':
        return {statement.carrier}
    if tag in ('Init', 'Load', 'Let', 'Cond', 'Loop', 'StridedLoop', 'Sync', 'Smem', 'CpAsyncCopy', 'CpAsyncCommit', 'CpAsyncWait', 'Tile'):
        return set()
    return None

def exprs(statement):
    tag = kind(statement)
    if tag in ('Load', 'Write'):
        return tuple(statement.index)
    if tag == 'Let':
        return (statement.value,)
    if tag == 'Cond':
        return (statement.cond,)
    if tag == 'StridedLoop':
        return (statement.start, statement.step) + ((statement.end,) if statement.end is not None else ())
    if tag == 'CpAsyncCopy':
        return tuple(statement.smem_index) + tuple(statement.lane_index or ()) + tuple(statement.src_index)
    if tag == 'Select':
        return tuple(branch.select for branch in statement.branches)
    if tag in ('Carry', 'Pre'):
        return tuple(statement.index)
    if tag in ('Assign', 'Accum', 'Init', 'Loop', 'Sync', 'Smem', 'CpAsyncCommit', 'CpAsyncWait', 'WarpShuffle', 'TreeHalve', 'Tile'):
        return ()
    return None

