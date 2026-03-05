# Phase 0 / T1: miniexpr DSL syntax inventory (source-anchored)

This is the **ground-truth inventory** from current `../miniexpr` sources/tests.
It is intended to drive `P0-T2` (`../miniexpr/doc/dsl-syntax.md`) and later Python-side validation.

## 1. Top-level program shape

- Exactly one top-level function definition is expected.
- Leading blank/comment lines are allowed.
- Anything after the function body is rejected.

References:
- `../miniexpr/src/dsl_parser.c:1531`
- `../miniexpr/src/dsl_parser.c:1548`
- `../miniexpr/src/dsl_parser.c:1553`

## 2. Pragmas

Supported header pragmas:
- `# me:fp=<strict|contract|fast>`
- `# me:compiler=<tcc|cc>`

Behavior:
- duplicates are rejected (`duplicate me:fp pragma`, `duplicate me:compiler pragma`)
- unknown `me:*` pragma is rejected
- malformed assignments/trailing content are rejected

References:
- `../miniexpr/src/dsl_parser.c:247`
- `../miniexpr/src/dsl_parser.c:312`
- `../miniexpr/src/dsl_parser.c:373`
- `../miniexpr/src/dsl_parser.c:411`
- `../miniexpr/src/dsl_parser.c:423`
- `../miniexpr/src/dsl_parser.c:435`
- `../miniexpr/tests/test_dsl_syntax.c:1203`
- `../miniexpr/tests/test_dsl_syntax.c:1388`

## 3. Statement kinds (parser enum)

Recognized statement kinds:
- assignment
- expression statement
- return
- print
- if/elif/else
- while
- for
- break
- continue

References:
- `../miniexpr/src/dsl_parser.h:16`
- `../miniexpr/src/dsl_parser.c:1269`

## 4. Function header and parameters

Function header rules:
- must start with `def`
- requires function name
- requires `(...)` parameter list
- requires trailing `:`
- duplicate parameter names rejected

References:
- `../miniexpr/src/dsl_parser.c:1461`
- `../miniexpr/src/dsl_parser.c:1491`
- `../miniexpr/src/dsl_parser.c:1510`
- `../miniexpr/src/dsl_parser.c:1520`
- `../miniexpr/src/dsl_parser.c:1429`
- `../miniexpr/src/dsl_parser.c:1445`

## 5. Blocks and indentation

Python-like indentation is enforced:
- block must be indented after `:`
- dedent ends block
- blank/comment-only lines are allowed in blocks

References:
- `../miniexpr/src/dsl_parser.c:808`
- `../miniexpr/src/dsl_parser.c:813`
- `../miniexpr/src/dsl_parser.c:1360`
- `../miniexpr/src/dsl_parser.c:1401`

## 6. Control flow forms

### 6.1 if/elif/else
- supports `if`, chained `elif`, optional `else`
- `elif` after `else` is rejected
- duplicate `else` is rejected
- stray `elif`/`else` rejected

References:
- `../miniexpr/src/dsl_parser.c:852`
- `../miniexpr/src/dsl_parser.c:914`
- `../miniexpr/src/dsl_parser.c:942`
- `../miniexpr/src/dsl_parser.c:1297`
- `../miniexpr/src/dsl_parser.c:1301`

### 6.2 while
- `while <expr>:` supported
- body required/indented
- runtime loop-iteration cap exists (`ME_DSL_WHILE_MAX_ITERS`)

References:
- `../miniexpr/src/dsl_parser.c:974`
- `../miniexpr/src/miniexpr.c:2745`
- `../miniexpr/src/miniexpr.c:8621`

### 6.3 for
Only this form is accepted:
- `for <var> in range(...):`

`range` arity at compile-time:
- 1 arg: `range(stop)`
- 2 args: `range(start, stop)`
- 3 args: `range(start, stop, step)`
- other arities rejected

Runtime:
- `step == 0` is runtime eval error

References:
- `../miniexpr/src/dsl_parser.c:1005`
- `../miniexpr/src/dsl_parser.c:1027`
- `../miniexpr/src/dsl_parser.c:1044`
- `../miniexpr/src/miniexpr.c:3638`
- `../miniexpr/src/miniexpr.c:3652`
- `../miniexpr/src/miniexpr.c:8747`
- `../miniexpr/tests/test_dsl_syntax.c:180`

### 6.4 break/continue
- only valid inside loops
- deprecated `break if ...` / `continue if ...` explicitly rejected

References:
- `../miniexpr/src/dsl_parser.c:717`
- `../miniexpr/src/dsl_parser.c:726`
- `../miniexpr/src/dsl_parser.c:733`
- `../miniexpr/src/miniexpr.c:7153`
- `../miniexpr/tests/test_dsl_syntax.c:472`

## 7. Assignments

Supported syntactic forms:
- `x = expr`
- `x += expr`
- `x -= expr`
- `x *= expr`
- `x /= expr`
- `x //= expr`

Desugaring:
- `//=` becomes `floor(lhs / (rhs))`

References:
- `../miniexpr/src/dsl_parser.c:1100`
- `../miniexpr/src/dsl_parser.c:1138`
- `../miniexpr/src/dsl_parser.c:1152`
- `../miniexpr/src/dsl_parser.c:1164`
- `../miniexpr/src/dsl_parser.c:214`
- `../miniexpr/tests/test_dsl_syntax.c:1604`

## 8. print statement

Parser recognizes `print(...)` as dedicated statement.
Compiler rules:
- at least one argument
- optional first string-format argument
- placeholder count must match supplied value args
- print args must be uniform expressions

References:
- `../miniexpr/src/dsl_parser.c:1245`
- `../miniexpr/src/dsl_parser.c:1305`
- `../miniexpr/src/miniexpr.c:6878`
- `../miniexpr/src/miniexpr.c:6932`
- `../miniexpr/src/miniexpr.c:6979`
- `../miniexpr/src/miniexpr.c:7026`
- `../miniexpr/tests/test_dsl_syntax.c:1451`

## 9. Expressions: parser vs compiler responsibilities

Parser-side expression handling is intentionally shallow:
- captures text until end-of-statement with balanced parentheses and string checks
- does **not** parse Python expression grammar deeply at DSL-parser level

Compilation/evaluation is delegated to miniexpr expression compiler (`private_compile_ex`) plus DSL semantic checks.

References:
- `../miniexpr/src/dsl_parser.c:575`
- `../miniexpr/src/dsl_parser.c:621`
- `../miniexpr/src/dsl_parser.c:633`
- `../miniexpr/src/miniexpr.c:3358`
- `../miniexpr/src/miniexpr.c:3429`

## 10. Reserved identifiers and ND symbols

Reserved names rejected for user vars/functions:
- `print`, `int`, `float`, `bool`, `def`, `return`, `_ndim`, `_i<d>`, `_n<d>`

ND reserved symbol handling:
- `_i0.._iN`, `_n0.._nN`, `_ndim` scanned and injected as synthetic vars when used.

References:
- `../miniexpr/src/miniexpr.c:546`
- `../miniexpr/src/miniexpr.c:602`
- `../miniexpr/src/miniexpr.c:7422`
- `../miniexpr/src/miniexpr.c:7431`
- `../miniexpr/src/miniexpr.c:7462`
- `../miniexpr/tests/test_dsl_syntax.c:855`

## 11. Cast intrinsics (current explicit support)

Supported intrinsics:
- `int(expr)`
- `float(expr)`
- `bool(expr)`

Validation:
- must be called form
- exactly one argument
- bad arity rejected

References:
- `../miniexpr/src/miniexpr.c:568`
- `../miniexpr/src/miniexpr.c:654`
- `../miniexpr/src/miniexpr.c:660`
- `../miniexpr/src/miniexpr.c:764`
- `../miniexpr/src/miniexpr.c:3377`
- `../miniexpr/tests/test_dsl_syntax.c:1485`
- `../miniexpr/tests/test_nd.c:116`

## 12. Signature and variable binding constraints

Compile-time constraints:
- DSL function parameters must match provided variable entries by name (set equality; order can differ)
- param count mismatch rejected
- duplicate/conflicting variable/function names rejected

References:
- `../miniexpr/src/miniexpr.c:7247`
- `../miniexpr/src/miniexpr.c:7268`
- `../miniexpr/src/miniexpr.c:7386`
- `../miniexpr/src/miniexpr.c:7395`
- `../miniexpr/tests/test_dsl_syntax.c:503`

## 13. Return semantics and dtype consistency

Compile-time:
- at least one return expression must be compilable
- all return paths that do return must share dtype

Runtime:
- non-guaranteed-return programs can compile, but missing return at runtime yields eval error

References:
- `../miniexpr/src/miniexpr.c:6857`
- `../miniexpr/src/miniexpr.c:6869`
- `../miniexpr/src/miniexpr.c:7497`
- `../miniexpr/tests/test_dsl_syntax.c:494`
- `../miniexpr/tests/test_dsl_syntax.c:552`

## 14. DSL detection and compile error mapping

- DSL candidate detection is heuristic (`dsl_is_candidate`)
- If parsed/treated as DSL and compile fails, compile API returns parse error with offset

References:
- `../miniexpr/src/miniexpr.c:2790`
- `../miniexpr/src/miniexpr.c:7534`
- `../miniexpr/src/miniexpr.c:7573`
- `../miniexpr/src/miniexpr.h:234`

## 15. Known unsupported / risky constructs (current behavior)

These are important for a Python-side validator because DSL parser accepts expression text broadly:

- Python expression forms not representable in miniexpr grammar (example: ternary `a if c else b`) are not blocked at DSL-parser level and rely on downstream expression compile behavior.
- Unsupported/unknown function calls in expressions are also largely deferred to expression compilation.
- Current user-facing diagnostics in Python can be poor if failures happen late; this matches the need for preflight syntax checks in `dsl_kernel.py`.

Evidence:
- parser stores expression text opaquely: `../miniexpr/src/dsl_parser.c:575`
- compile delegation: `../miniexpr/src/miniexpr.c:3429`
- DSL parse/compile failure path in compile API: `../miniexpr/src/miniexpr.c:7573`

## 16. Notes for P0-T2 doc authoring

When moving this into `../miniexpr/doc/dsl-syntax.md`, keep two explicit tables:
- **Syntax rejection (parse/compile time)**
- **Runtime semantic errors** (e.g., zero `range` step, missing return path)

This distinction is already visible in tests and should remain explicit.
