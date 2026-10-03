# Caterva2 access improvements for Python-Blosc2 and b2view

Status: implementation proposal; no implementation implied by this document.

Implementation record (updated as milestones land):

- M0/M1: selected `remote_service="auto" | "caterva2" | "fsspec"` and
  implemented root-marker recognition before format dispatch. String service
  URLs default to lazy access; explicit URLPath defaults remain unchanged.
  Inspection confirms `open` currently defaults to `mode="r"` (the proposal's
  concern about changing that default requires no code change).
- Repository design: use a public RemoteRepository subclass of RemoteStore as
  a browsing facade with independently owned roots and per-root cache budgets.
  Repository persistence is explicitly unsupported initially. Single-root
  servers flatten to their actual root object; empty/multi-root servers return
  the repository facade. Root owners stay alive through normal returned child
  handles after the facade closes.
- Roots discovery uses the mapping response implemented by cat2lite and
  Caterva2, not an assumed list. No global discovery cache is introduced.
- M1 validation: 20 offline URL/dispatch tests passed in the blosc2 environment.
- M2: Caterva2 groups now discover immediate children on demand, support direct
  descendant lookup, memoize completed listings under the owner lock, and roll
  back node-limit failures. Opening a string root uses the opener's metadata
  seed to avoid duplicate info calls. Catalog annotations are exposed separately
  on RemoteNode and retained in discovery metadata. Focused URL/Caterva2 tests:
  37 passed, including existing reference/table/cache regressions.
- M3: added bounded roots discovery (3-second deadline, 1 MiB response bound,
  same-origin redirects only), direct-source bypass, strict error/fallback
  classification, and the public RemoteRepository browsing facade. Roots open
  independently and lazily; facade aliases share owners, returned children
  outlive the facade, and authentication is frozen at opening. Per-root cache
  budgets and unsupported repository persistence are explicit. Validation:
  38 access tests passed, including zero/one/multiple roots and error handling.

## 1. Goal

Make Caterva2-compatible servers, including cat2lite, first-class URL-addressable
sources in Python-Blosc2. Users should be able to browse a curated remote
repository through a shared caching gateway with the same viewer and opening
API used for individual files.

Target usage:

```python
import blosc2

repository = blosc2.open("http://localhost:8000", mode="r")
public = blosc2.open("http://localhost:8000/@public", mode="r")
group = blosc2.open("http://localhost:8000/@public/hdf5", mode="r")
array = blosc2.open("http://localhost:8000/@public/hdf5/d0/d1/a2", mode="r")
```

```sh
b2view http://localhost:8000
b2view http://localhost:8000/@public
b2view https://cat2.cloud/demo/@public/example
```

The examples above describe proposed behavior. Preserve existing direct-file
opening, including HTTP B2ND/B2Z, HDF5, Zarr, and Parquet sources.

Architecture:

```text
b2view / Python applications
          |
          | Caterva2 metadata, listings, bounded slices
          v
cat2lite + shared server-side disk cache
          |
          v
remote repositories exposed through .catl mounts
```

Do not create a separate cat2lite-view application. Source recognition belongs
in Python-Blosc2; the existing b2view UI should consume the resulting objects.

## 2. Current implementation and gaps

Relevant code to inspect before implementation:

- `src/blosc2/schunk.py`: `open`, `_open_c2_urlpath`, remote option validation,
  lazy dispatch, and existing format-specific openers.
- `src/blosc2/c2array.py`: `URLPath` integration, API URL construction, shared
  HTTP clients, authentication, info/list/fetch helpers.
- `src/blosc2/remote_store.py`: Caterva2 source descriptors, `_open_caterva2`,
  `resolve`, `kind`, `list_children`, owner/cache lifecycle, and table fetching.
- `src/blosc2/b2view/model.py`: `StoreBrowser._open_store`, tree traversal,
  remote leaf ownership, previews, and table capabilities.
- `src/blosc2/b2view/app.py`: background opening, tree expansion, failure display.
- `src/blosc2/b2view/cli.py`: source arguments, cache options, initial path.

Existing foundations:

- Explicit `blosc2.URLPath(path, urlbase=...)` already selects Caterva2 handling.
- The lazy Caterva2 opener discovers groups, arrays, and CTables; groups can
  return `RemoteStore`, arrays use existing C2Array/RemoteArray semantics, and
  tables can return `RemoteCTable`.
- b2view already displays RemoteStore hierarchies and remote leaves.
- cat2lite already implements roots, info, list, and bounded data access.

Gaps:

1. A plain HTTP string is not converted into a Caterva2 URLPath.
2. Bare server URLs are treated as data files, without roots discovery.
3. `_open_caterva2` currently assumes an initial recursive listing describes the
   hierarchy and initializes discovered groups' child lists from that snapshot.
   cat2lite catalog listings deliberately stop at mounted-source boundaries.
   Thus a discovered mounted group can appear empty instead of being expanded
   through another API list request.
4. b2view has format-specific URL dispatch that must not override the new
   Caterva2 interpretation when a published dataset name ends in `.h5`, etc.
5. A server with multiple roots has no agreed repository-level browsing object.

## 3. URL and opening contract

### 3.1 Explicit dataset/group shorthand

Recognize an HTTP(S) URL containing an `@`-prefixed path component as Caterva2
shorthand, consistently with cat2lite's `.catl` convention.

Example:

```text
https://host/demo/@public/group/array
  urlbase = https://host/demo
  path    = @public/group/array
```

Rules:

- Parse with a real URL parser, preserving bracketed IPv6 and port numbers.
- Inspect path components only; ignore `@` in hostname, userinfo, or query.
- Use the first root-marker component; later `@` components are dataset names.
- Preserve the encoded deployment prefix; decode logical dataset components
  exactly once and use existing Caterva2 path validation/endpoint encoding.
- Reject empty root markers, traversal, encoded separators, malformed encoding,
  control characters, and unsupported query/fragment/selector combinations.
- Do not reinterpret fsspec `::/member` selectors as Caterva2 dataset selectors.
- Authentication uses existing Caterva2 facilities, not URL userinfo.
- Explicit URLPath inputs retain precedence and behavior.
- Return the object's actual type; do not wrap every dataset in RemoteStore.
- Preserve existing `mode`, `offset`, `lazy`, and cache-option validation.
  Remote sources remain read-only. Specify `mode="r"` in documentation rather
  than changing the global default mode as part of this work.
- `lazy=False` must retain its existing meaning or explicitly reject group/
  repository opening; do not silently ignore it to force browser semantics.

### 3.2 Ordinary HTTP override

An ordinary file server can legitimately contain `@` path components. Provide
an explicit bypass for Caterva2 recognition and repository probing.

Proposed API decision for M0: a narrowly scoped selector such as
`remote_service="auto" | "caterva2" | "fsspec"` on `blosc2.open`.
This name is provisional. Audit existing options before adding it; do not
overload `source_format`, which currently describes data formats.

- `auto`: root-marker shorthand, known-file dispatch, then eligible base probe.
- `caterva2`: explicitly interpret a URL as a repository base or dataset URL.
- `fsspec`: use the ordinary fsspec-backed source handling, even with `@`
  components, bypassing Caterva2 recognition and probing. This identifies the
  access backend, not a local file or a single-file-only source; remote
  containers and stores remain supported.
- Explicit URLPath plus a conflicting override is an error.
- Reject this option for inputs where it has no meaning rather than ignoring it.
- If needed, expose the same choice in b2view through a small CLI flag.

### 3.3 Bare server and deployment-base URLs

For an eligible ambiguous HTTP(S) URL, probe `<base>/api/roots` with a bounded
timeout and validate the response against the actual Caterva2 roots contract.
Inspect both Caterva2 and cat2lite responses during M0; do not assume a list
of root names if the API returns a mapping with metadata.

Recommended dispatch order:

1. Explicit URLPath or explicit service override.
2. Root-marker URL shorthand.
3. Known file/container URL, including selectors and trailing-slash Zarr.
4. Discovery for ambiguous base candidates, including `/demo` prefixes.
5. Existing generic remote-file path if discovery conclusively says this is
   not a Caterva2-compatible server.

Discovery must not add roots requests to recognized direct-file reads or turn
an existing permission/format error into an unrelated service-detection error.

Probe handling:

- Valid roots response: bind the server base and authentication context.
- Missing endpoint, HTML, malformed JSON, or invalid schema: classify as
  non-Caterva2 in auto mode; provide a clear discovery error in explicit mode.
- 401/403: report authentication/authorization failure; no anonymous fallback.
- Connection/TLS failure or timeout: report an actionable connection error,
  preserving the original cause; avoid serial retry/fallback delays.
- Server errors: preserve the server error rather than saying “not Caterva2”.
- Preserve deployment prefixes and follow existing redirect policy. Never
  forward authentication credentials to an unrelated redirect origin.
- Close response resources, reuse the existing transport, and avoid duplicate
  roots/info requests when passing discovery results into an owner.
- Do not introduce a process-global negative discovery cache initially.
  Any later positive cache must include authentication identity and lifetime.

## 4. Repository root behavior

Recommended user-facing behavior:

- Exactly one root: open it directly, so cat2lite's `@public` server feels like
  a directly opened hierarchy.
- Multiple roots: expose a synthetic repository group with roots as children.
- Zero visible roots: return an empty repository group, with b2view showing
  “No accessible roots”; this is not necessarily a server error.
- An explicit `/@public` URL always opens that root, regardless of other roots.

Implementation decision for M0: prefer a repository-backed RemoteStore mode
over a viewer-only wrapper. Confirm it fits ownership and persistence semantics
before choosing a new public class. All applications should get the same
`keys`, lookup, `kind`, attrs, context-manager, and close behavior.

The synthetic group:

- Is not sent to `/api/info` as a fabricated empty dataset path.
- Discovers each root only when accessed; it does not open all roots at startup.
- Isolates inaccessible/broken roots without hiding healthy siblings.
- Has its own identity distinct from a dataset-root descriptor.
- Keeps independent root owners under a repository lifetime; closing the
  repository must follow existing live-child ownership conventions.
- Makes cache allowance scope explicit. Prefer a repository-wide bound where
  the owner architecture supports it; otherwise document per-root allowances
  before shipping rather than implying a total bound.
- Either defines safe persistence/reopening explicitly or rejects repository
  persistence clearly in the first implementation. In-memory browsing must
  not accidentally write an invalid existing dataset descriptor.

Single-root flattening means relative paths can change if server roots change
between opens. Document this and recommend explicit root URLs for scripts.

## 5. Lazy Caterva2 hierarchy discovery

Refactor Caterva2 discovery into separate root metadata, node metadata, and
group-listing operations. Track whether a group has actually been listed;
“known group with no discovered children” is not “known empty group”.

Required behavior:

1. Open a known root with bounded metadata work, without recursive remote
   source expansion.
2. Expand a group through `/api/list/<path>` only when needed.
3. Accept existing recursive relative-path lists and catalog lists that stop
   at mounts. Synthesize structural parent groups where necessary.
4. Mark only the queried group's listing as complete. A returned group may
   need its own request, even if no descendants were returned initially.
5. Fetch and memoize metadata needed to classify actual children. Do not
   assume every listed name is an array or eagerly inspect every deep leaf.
6. Resolve a directly requested descendant through info requests even when its
   parents have not been expanded. Update the same node registry afterward.
7. Keep listings deterministic and deduplicated; empty groups remain visible.
8. Bound node growth and discovery work; reject malformed or escaping paths.
9. Coalesce concurrent discovery for the same group and publish registry
   updates atomically. Failed discovery must remain retryable.
10. Preserve attrs and separate `catalog_attrs` annotations without overwriting
    source attrs. Decide how annotations are displayed in b2view metadata.

There is a protocol constraint: an existing server may return a large recursive
list in a single response. This client change cannot promise paginated or
constant-size listings without a server API extension. Avoid eagerly fetching
info for every returned descendant; measure remaining list costs separately.

## 6. Cache and data-access semantics

- A cat2lite server cache is shared across its HTTP clients. Python-Blosc2's
  client cache is a separate layer, with independently configured policies.
- Reuse existing none/memory/disk options; do not invent a special server-cache
  policy or reinterpret `shared_cache=True` as “use cat2lite”. That existing
  option has its own local-cache semantics and locking requirements.
- Normalize shorthand strings and equivalent URLPath sources to the same
  dataset cache identity, including base prefix, path, and authentication scope.
- Repository synthetic identities must not collide with dataset identities.
- Do not persist credentials or allow a cached authenticated view to leak into
  a different identity. Follow the current authenticated-cache restrictions.
- Preserve exclusive disk-owner locking, bounded retention, restart behavior,
  and reference-counted leaf lifetimes.
- Browsing uses metadata only; previews fetch bounded slices/pages rather than
  downloading whole containers. An upstream backend may still prefetch source
  bytes according to its own semantics; distinguish this from client downloads.
- Preserve immutable-source assumptions and explicit invalidation requirements.
  No mutable-source TTL or automatic refresh protocol is introduced here.
- Surface server slice-limit errors with guidance to request a smaller preview.
- Do not imply all server-side table operations are available: cat2lite's
  unsupported filters/indices must not be silently ignored or trigger an
  unbounded local materialization.

## 7. b2view integration

- Route service recognition through shared Python-Blosc2 opening helpers.
- Retain special direct-format handling only where needed for existing array/
  table/group ownership behavior. Caterva2 interpretation takes precedence over
  extensions in published dataset names.
- Run discovery, group expansion, and previews in background work, keeping the
  UI responsive during slow upstream metadata reads.
- Show loading, empty-group, and retryable error states distinctly.
- Support initial paths within a root or synthetic repository.
- Check remote arrays, table paging/projection, scalar and multidimensional
  previews, attrs, and plotting through existing data adapters.
- Gate unsupported table actions consistently; do not offer operations that
  require full remote downloads merely because a local CTable supports them.
- Cancellation/closing the UI must release owners and transports cleanly.
- Keep CLI cache options controlling the client cache, and label this clearly
  in help. The user configures the shared server cache on cat2lite separately.

## 8. Implementation milestones

### M0 — Freeze contract and establish fixtures

- Audit roots schemas, opener defaults/lazy behavior, authentication, and
  existing descriptor/persistence constraints.
- Finalize the explicit HTTP override and multi-root object design.
- Add deterministic local HTTP fixtures for both conventional recursive
  Caterva2 listings and catalog mount-boundary listings.
- Record request counters and configurable failures/delays in those fixtures.

Gate: a documented dispatch/return-type matrix and fixtures representing both
protocol profiles; no network dependency for ordinary tests.

### M1 — Root-marker URL shorthand

- Implement reusable normalization and integrate it before file-format dispatch.
- Preserve all existing URLPath and option semantics.
- Add override support and documentation for literal `@` file URLs.

Gate: equivalent strings and URLPath objects open the same groups/arrays/tables,
including deployment prefixes and IPv6, without roots probing.

### M2 — Lazy mount-aware RemoteStore traversal

- Split metadata discovery from listing completion.
- Implement direct descendant lookup, lazy expansion, memoization, and bounded
  concurrent discovery.
- Cover nested mounts and conventional recursively listed stores.

Gate: a group omitted below the initial mount boundary can be expanded and
read; healthy siblings remain usable after a failed expansion.

### M3 — Base discovery and repository roots

- Add bounded roots discovery and transport/error handling.
- Implement single-root, empty-root, and multiple-root behavior.
- Define repository cache and lifecycle rules; cover auth-separated identities.

Gate: bare and prefixed server URLs browse correctly, while direct-file sources
retain their dispatch and do not acquire extra roots requests.

### M4 — b2view integration

- Connect the shared opener, lazy tree expansion, and repository root object.
- Update title/source display, capabilities, errors, and CLI help.
- Add headless Textual tests against local servers.

Gate: bare-server and explicit-root command forms browse groups and display
bounded array/table previews without blocking the event loop.

### M5 — Cross-project acceptance, documentation, and measurements

- Run against a real debug cat2lite server serving a mixed `.catl` fixture.
- Verify shared server-cache reuse across two independent clients and after
  restart where the backend guarantees persistent reuse.
- Update API docs, b2view guide, examples, and release notes.
- Record startup/expansion request counts and representative cache evidence.

Gate: all focused tests and relevant offline regressions pass; the use cases in
section 1 work with documented limits and no new viewer executable.

## 9. Regression and acceptance matrix

### URL dispatch

- HTTP/HTTPS, IPv4/IPv6, ports, trailing slash, deployment prefixes.
- Root-marker group, array, table, and extension-bearing published names.
- Multiple `@` components, encoded spaces, Unicode, encoded `@` root markers.
- Query/userinfo/fragment ambiguity, traversal and double-decoding attempts.
- Ordinary direct-file `@` paths via override; signed direct-file URLs.
- Existing fsspec selectors, explicit URLPath, local paths, and non-HTTP inputs.
- Cache-option forwarding and rejection of contradictory inputs.

### Discovery and hierarchy

- Valid zero/one/multiple roots; unexpected JSON/HTML; 404/401/403/5xx; timeout.
- Prefix-preserving redirects and credential behavior.
- Recursive listings, mount-boundary listings, empty groups, duplicate entries.
- Direct access before parent expansion, repeated and concurrent expansion.
- Node limits and per-group failures with subsequent successful retry.
- No eager contact with unrelated remote mounts merely to open a repository.

### Data and caching

- Scalar/ND arrays, CTable row windows and projections, attrs/annotations.
- No implicit full fetch for previews; correct handling of server byte limits.
- none/memory/disk, identity equivalence, auth separation, close/reopen, restart.
- Two processes with independent client caches reuse one server-side cache.
  Use upstream byte/request counters and fixtures larger than backend prefetch
  thresholds; do not confuse client-cache hits with server-cache hits.
- Concurrent owner lifetimes and expected exclusive-lock failures where local
  disk caches are deliberately shared without the supported sharing mode.

### Viewer and compatibility

- Headless tree expansion, starting path, empty/error states, preview paging,
  cancellation, and clean shutdown under warnings-as-errors.
- Existing direct HDF5/Zarr/B2Z/Parquet viewing remains functional.
- Explicit URLPath callers retain return types and lazy=False behavior.
- Existing Caterva2 and remote-store tests remain green.

Run Python/build commands in the `blosc2` conda environment. Use targeted pytest
runs during implementation, then the relevant offline opener/remote-store/
remote-array/remote-table/b2view suites, Ruff on touched files, and the full
offline suite for final validation. Follow actual test filenames discovered in
the repository rather than assuming names in this plan are commands.

## 10. Completion criteria and boundaries

Complete when all proposed CLI forms work, equivalent Python opening works,
catalog-mounted hierarchies expand lazily, multi-root behavior is documented,
and cache reuse is demonstrated with measured upstream traffic.

This work does not add a new viewer, a new remote protocol, write access,
catalog parsing in Python-Blosc2, automatic cache invalidation, a server
deployment/authentication system, or a guarantee that arbitrary remote formats
support cheap random access. It reuses Caterva2's protocol and existing object
adapters to make the shared-cache-server workflow convenient and predictable.
