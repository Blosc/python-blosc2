"""Lower comparison chains for JavaScript emission (native DSL handles its own)."""

import ast
import copy


def _has_chain(node):
    return any(isinstance(child, ast.Compare) and len(child.ops) > 1 for child in ast.walk(node))


class _ComparisonLowerer:
    def __init__(self, func):
        self.names = {node.id for node in ast.walk(func) if isinstance(node, ast.Name)}
        self.names.update(node.arg for node in ast.walk(func) if isinstance(node, ast.arg))
        self.counter = 0

    def _capture(self, value, statements):
        while True:
            name = f"b2_chain_{self.counter}"
            self.counter += 1
            if name not in self.names:
                break
        self.names.add(name)
        statements.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value))
        return ast.Name(id=name, ctx=ast.Load())

    @staticmethod
    def _assign(target, value):
        return ast.Assign(targets=[ast.Name(id=target.id, ctx=ast.Store())], value=value)

    def _operand(self, node, statements):
        prelude, value = self._expr(node)
        statements.extend(prelude)
        return self._capture(value, statements)

    def _expr(self, node):
        if not _has_chain(node):
            return [], node
        statements = []
        if isinstance(node, ast.Compare):
            left = self._operand(node.left, statements)
            right = self._operand(node.comparators[0], statements)
            result = self._capture(
                ast.Compare(left=left, ops=[node.ops[0]], comparators=[right]), statements
            )
            for op, operand in zip(node.ops[1:], node.comparators[1:], strict=True):
                guarded = []
                next_value = self._operand(operand, guarded)
                guarded.append(
                    self._assign(result, ast.Compare(left=right, ops=[op], comparators=[next_value]))
                )
                statements.append(ast.If(test=result, body=guarded, orelse=[]))
                right = next_value
            return statements, result
        if isinstance(node, ast.BoolOp):
            first = self._operand(node.values[0], statements)
            result = self._capture(
                ast.Call(func=ast.Name(id="bool", ctx=ast.Load()), args=[first], keywords=[]), statements
            )
            for operand in node.values[1:]:
                guarded = []
                value = self._operand(operand, guarded)
                guarded.append(
                    self._assign(
                        result, ast.Call(func=ast.Name(id="bool", ctx=ast.Load()), args=[value], keywords=[])
                    )
                )
                test = result if isinstance(node.op, ast.And) else ast.UnaryOp(op=ast.Not(), operand=result)
                statements.append(ast.If(test=test, body=guarded, orelse=[]))
            return statements, result
        if isinstance(node, ast.BinOp):
            node.left = self._operand(node.left, statements)
            node.right = self._operand(node.right, statements)
        elif isinstance(node, ast.UnaryOp):
            node.operand = self._operand(node.operand, statements)
        elif isinstance(node, ast.Call):
            node.args = [self._operand(arg, statements) for arg in node.args]
        else:
            raise ValueError(f"Cannot lower comparison chain in {type(node).__name__}")
        return statements, node

    def block(self, body):
        result = []
        for node in body:
            result.extend(self._stmt(node))
        return result

    def _stmt(self, node):
        if isinstance(node, ast.If):
            prelude, node.test = self._expr(node.test)
            node.body = self.block(node.body)
            node.orelse = self.block(node.orelse)
            return [*prelude, node]
        if isinstance(node, ast.While):
            prelude, test = self._expr(node.test)
            node.body = self.block(node.body)
            node.orelse = self.block(node.orelse)
            if prelude:
                # Re-evaluate at the top of every iteration, including continue.
                node.test = ast.Constant(value=1)
                stop = ast.If(test=ast.UnaryOp(op=ast.Not(), operand=test), body=[ast.Break()], orelse=[])
                node.body = [*prelude, stop, *node.body]
            return [node]
        if isinstance(node, ast.For):
            prelude = []
            if _has_chain(node.iter):
                node.iter.args = [self._operand(arg, prelude) for arg in node.iter.args]
            node.body = self.block(node.body)
            node.orelse = self.block(node.orelse)
            return [*prelude, node]
        if isinstance(node, ast.Assign | ast.Return | ast.Expr | ast.AugAssign) and node.value is not None:
            prelude, value = self._expr(node.value)
            if isinstance(node, ast.AugAssign) and prelude:
                # Capture the previous target value before evaluating the RHS.
                before = []
                left = self._capture(ast.Name(id=node.target.id, ctx=ast.Load()), before)
                assign = ast.Assign(
                    targets=[node.target], value=ast.BinOp(left=left, op=node.op, right=value)
                )
                return [*before, *prelude, assign]
            node.value = value
            return [*prelude, node]
        return [node]


def lower_chained_comparisons(func):
    """Return a lowered function AST, or the original AST when no chains exist.

    Generated names avoid every identifier in the function. Operand evaluation
    is left-to-right, once per chain, and subsequent links run only when all
    preceding comparisons succeed. Existing DSL Boolean operations return bool.
    """
    if not _has_chain(func):
        return func
    func = copy.deepcopy(func)
    func.body = _ComparisonLowerer(func).block(func.body)
    return ast.fix_missing_locations(func)
