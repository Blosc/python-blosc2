# Menudet recipes in b2o v2

Implementation draft for Python-Blosc2 5.0 / C-Blosc2 4.0. Kernel language,
recipe and container versions are separate.

## Container records

New Menudet lazy arrays write fixed `b2o` metadata:

```python
{"kind": "menudet", "version": 2}
```

The variable-length `b2o` MessagePack payload contains:

```python
{
    "kind": "menudet",
    "version": 2,
    "recipe": {...},
    "shape": [...],       # logical input domain, not necessarily result shape
    "partitions": [...],  # immutable logical reduction/context groups
    "operands": {...},    # existing versioned references or inline recipes
}
```

Marker and payload kind/version must agree. Nested recipes dispatch by their
own kind/version, including MessagePack extension 43. Ordinary `lazyexpr` stays
v1. Existing Menudet v1 records still contain `artifact` JSON and use the v1
reader; standard builds retain that reader.

Computed and generated CTable columns use an analogous version-2 `recipe`
field in their own metadata, retaining bindings and independent-row geometry.
Older column metadata without a version remains a v1 JSON read.

## Reusable recipe

The recipe is a plain mapping:

| Field | Meaning |
| --- | --- |
| `version` | Recipe layout `1`, independent of b2o version `2`. |
| `language_version` | Explicit Menudet `"0.1"` or annotated `"0.2"`. |
| `source`, `entry_point` | Normalized portable function, never a Python callable. |
| `inputs` | Runtime parameter **names**, without inferred dtypes. |
| `constants` | Exact immutable captured scalars, with dtype and scalar category. |
| `output` | Authored dtype descriptor, or `None` for inference. A return annotation can constrain it. |
| `cardinality`, `ndim`, `casting` | Authored result/context/conversion contract. |
| `input_constraints` | Optional explicit constraints not expressible in source, such as fixed-string widths; also used when converting a fixed v1 contract. |
| `semantics` | Optional `integer_powers: "multiply"` requirement. Floating/math declarations stay in source pragmas. |
| `metadata` | Optional passive descriptive mapping. |

For example, `word: np.uint16` becomes `word: uint16` in source. An unannotated
`reading` remains unannotated after a float32 specialization; another caller can
bind float64. Reopening obtains actual dtypes from operand metadata, then calls
`MenudetKernel.from_recipe(recipe, input_dtypes)`, which prepares the native
specialization directly. The numerical interpreter remains authoritative.

Captured values retain the existing exact scalar descriptor encodings:
range-checked decimal integer strings, Boolean values, IEEE bits in hexadecimal,
and fixed-byte/Unicode32 big-endian hexadecimal strings with exact byte widths.
`category` distinguishes weak Python scalars from strong typed scalars. These
are fields in MessagePack, not a nested JSON document. An explicit parameter
annotation makes a matching weak capture strong when preparing 0.2 source.

A v1 fixed artifact cannot reveal whether its author declared or inferred a
dtype. Converting it to a v2 recipe therefore preserves **all** input and output
types as constraints. No automatic generalization occurs.

## Execution and validation

Recipe validation rejects foreign objects, NUL/invalid Unicode, unsupported
fields, excessive nesting and oversized passive payloads before copying or
resolving operand references. Native preparation validates the typed source,
annotations, cardinality, captures and casting; it never executes Python source.
Environment/scoped execution settings affect only an ephemeral prepared copy.
The authored recipe is snapshotted independently of those settings.

The Cython handle uses the miniexpr typed core through a transition adapter
which preserves the existing native array/graph schedulers and immutable-handle
provenance checks. Preparation performs no JSON encoding/decoding. Explicit
`to_json()` remains a fixed v1 export: annotations are erased, concrete input
and output types are written, and annotated captures retain their strong type.
Optimized author-backed graph handles retain the same immutable recipe snapshot;
replacing their Python-side recipe cannot change their persistence provenance.

Python file/CFrame/extension round trips, respecialization, independent-row
tables, v1 reads and JSON-free preparation are tested. A standalone C-Blosc2
reader/writer for the full `b2o` reference closure is still separate integration
work; the core does not resolve files, stores or remote references.

## Remaining legacy dependencies

The new preparation route is not yet a deletion boundary for every old C entry
point. The current Python caller inventory is:

| Caller | Remaining dependency |
| --- | --- |
| Ordinary formula prefilter → `NDArray._set_pref_expr()` | Direct formula preparation and shared `me_grid` geometry; JIT preference never wraps the formula in another language. |
| `blosc2_ext.pyx:aux_miniexpr()` | Ordinary formulas use the core grid handle; explicit full-source/unresolved formula-reduction compatibility retains the old ND handle and scalar-tail contract. |
| `lazyexpr.py` dtype inference and source-backed UDF width checks | `me_output_dtype()` uses typed formula discovery or explicitly versioned portable preparation. |
| `dsl_kernel.py:validate_dsl_jit()` | Direct portable preparation with binary captures; `_dsl_jit_status()` is a portable compatibility spelling, not an ND compiler probe. |
| `dsl_execution.py` source-backed compatibility executor | Explicit fixed JSON exports/imports remain outside new `@menudet` author preparation. |
| `native_graph.py` graph adapters | Graph JSON remains its own contract; v2 recipe storage does not replace it. |

The remaining full-source and formula-reduction routes are deliberate
compatibility consumers, not dead code. They retain their original language
under miniexpr's 1.x compatibility policy. Do not wrap formulas as Menudet
merely to gain JIT or silently assign block-reduction semantics to the new
formula API. Formula reductions are still deferred.

## Cross-language fixtures

Build miniexpr's `recipe_v2_peer` target and set `BLOSC2_MENUDET_INTEROP` to
that executable when running `tests/test_menudet_interop.py`. The C peer links
only the JSON-free core. Tests exchange zero-input coordinate generators with
exact float64 captures, annotated/unannotated nested inputs, and extension 43
in both directions; C-authored payloads also pass file/CFrame reopening.
Unsupported versions, capture types and geometry reject in the peer.
This is an explicitly bounded interoperability fixture subset, not a general
C-Blosc2 reference resolver. Existing v1 Menudet reads remain tested separately.
