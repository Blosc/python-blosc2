# Historical decision: rejected portable @jit prototype (2026-10-31)

**Not a supported API.** The temporary `@blosc2.jit(portable=True)` prototype was
removed after review. Use `@blosc2.menudet` for validated lazy kernel authoring and
`@blosc2.jit` for NumPy-style eager/tracing convenience and pandas integration.
See [menudet-api.md](menudet-api.md) for the chosen contracts and current 200M run.

The prototype preserved existing eager `@jit` behavior behind an explicit flag,
but that flag selected substantially different authoring semantics, result types
and methods. Keeping separate decorators is clearer than treating these as modes
of the same operation. Menudet remains the language, artifact and public type name;
there is no native ABI or saved-artifact rename.

Retained independently of the rejected consolidation:

- Native parallel generation, contiguous partition scheduling and integer JIT
  optimizations from the tip 11 performance investigation.
- Scoped Menudet backend selection, runtime-only compiler defaults and native
  validation of malformed artifacts. These do not require a new decorator mode.
- Existing Menudet specialization, scalar annotations/captures, native error
  propagation, table registration and durable recipe persistence.

The prototype's [raw 200M measurements](tip11-portable-jit-200m-cc.json) remain
unchanged as historical provenance. They measured 0.0446s for the hash versus
0.3643s for NumPy staging on the qualified host. They must not be relabeled as a
new `@menudet` run: the current benchmark/plot has been measured again using the
chosen decorator. The prototype-specific tests, API wrappers, option restrictions
and public guidance have been removed rather than retained as an alternative API.
