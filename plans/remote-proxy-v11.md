# RemoteProxy v11: Caterva2 user attributes

## Goal

Expose consistent user attributes for arrays served through Caterva2, including
ordinary Blosc2 arrays, B2Z leaves, HDF5 leaves, and saved RemoteProxy arrays.
Reuse `/api/info` so reading attributes requires no additional endpoint or request.

The preceding Python-Blosc2 change adds `RemoteProxy.attrs` as a read-only alias
for `RemoteProxy.vlmeta` and filters Kerchunk's `_ARRAY_DIMENSIONS` from
`HDF5NDSource.vlmeta`.

## Current behavior and gap

Caterva2 already exposes variable metadata through `schunk.vlmeta` in array
metadata responses and through `File.vlmeta` in its Python client. HDF5 adapters
already copy dataset attributes into their backing Blosc2 array's variable
metadata. The web metadata panel displays this mapping directly.

Saved RemoteProxy arrays are different: `services/server.py:get_info` replaces
their variable metadata with only the `b2o` descriptor to avoid exposing binary
cache bookkeeping. This also discards the saved user attributes, which reside
under `_b2o_user_vlmeta` in the carrier. Python-Blosc2 already provides
`blosc2.b2objects.read_b2object_user_vlmeta()` to retrieve them.

## Proposed changes

### 1. Add public attributes to the existing response

Add an `attrs` mapping to the relevant `/api/info` metadata models. Populate it
in Caterva2's shared metadata builder (`services/srv_utils.py:read_metadata`)
so standalone arrays and container leaves use the same behavior:

- Ordinary Blosc2 arrays: expose user variable metadata.
- HDF5 leaves and legacy HDF5 proxies: expose dataset attributes, excluding
  adapter bookkeeping such as `_ftype` and `_dsetname`.
- Saved RemoteProxy arrays: use `read_b2object_user_vlmeta()` on the raw carrier,
  without resolving or fetching the remote array.

Preserve the existing `schunk.vlmeta` response contract, including control
information used by existing clients. Keep public attributes separate from
proxy descriptors, cache bookkeeping, and fill protocol fields. Identify
internal fields explicitly; do not remove every underscore-prefixed user key.

Keep the new field optional during compatibility handling: distinguish an
absent field from an explicitly empty mapping. Ensure peer responses and their
model conversion preserve the field.

### 2. Expose attributes in the clients

Add `.attrs` to Caterva2's `File` class (inherited by its dataset objects) and
Python-Blosc2's `C2Array`. Prefer the new response field; fall back to the
existing variable metadata when talking to an older server. Reuse each client's
existing metadata cache and refresh behavior.

Update `RemoteProxy`'s Caterva2 metadata path to consume these public attributes.
Its `.attrs` and `.vlmeta` must continue to return the same read-only mapping.
Preserve existing `.vlmeta` compatibility in the lower-level clients, since
their callers may depend on protocol fields there.

### 3. Update the web metadata panel and documentation

Have `services/templates/includes/info_metadata.html` display `attrs`, with a
fallback for older metadata responses. Label the section "Attributes" and avoid
showing proxy descriptors or cache bookkeeping as user attributes.

Document the new response field, client access, read-only remote behavior, and
older-server fallback. Explain that saved proxy attributes reflect the metadata
stored in the carrier; this change does not introduce remote metadata refreshes
on the server.

### 4. Verify the complete route

Extend existing tests rather than introducing a new test harness:

- `/api/info` preserves scalar and nested user attributes for ordinary Blosc2
  arrays, B2Z leaves, HDF5 leaves, and saved RemoteProxy arrays.
- Saved proxies expose their stored user attributes without resolving their
  source or exposing cache internals.
- Existing descriptor and fill protocol metadata remain available to callers
  that use the existing response fields.
- Caterva2 `.attrs`, C2Array `.attrs`, and RemoteProxy `.attrs` return the
  expected attributes; RemoteProxy `.attrs` and `.vlmeta` share their cache.
- Missing `attrs` falls back for older servers; explicit empty `attrs` stays
  empty. Peer forwarding and the web metadata panel preserve the result.

Run the affected API, container, HDF5, remote-proxy, and client tests in the
repositories' prescribed environments, together with their lint checks.

## Scope

Implementation spans `/Users/faltet/ironArray/caterva2` and
`/Users/faltet/blosc/python-blosc2`. The implementation is complete in both
working trees.

Attribute writes, a separate metadata endpoint, new serialization formats, and
server-side refresh of saved remote attributes are outside this change.

## Implementation and validation

Implemented the public `attrs` field, both client properties, RemoteProxy's
Caterva2 integration, and the web metadata panel. The file-less HDF5 leaf
adapter also needed to copy attributes using the existing HDF5 conversion;
its previous implementation copied only array geometry.

Validation in the `blosc2` conda environment:

- Python-Blosc2 RemoteProxy, open-C2Array, and HDF5 source modules: 121 passed.
- Focused Caterva2 metadata, client, HDF5, B2Z, table, saved-proxy, panel, and
  peer checks: 21 passed.
- Broader Caterva2 API, HDF5, TreeStore, CTable, and RemoteProxy modules, plus
  attribute checks: 296 passed, 110 skipped, two failures.
- Both failures reproduced against an unchanged Caterva2 HEAD snapshot:
  `test_remote_proxy_download_can_omit_cache[CachePolicy.DISK]` encounters the
  disabled-resolution policy on warm download, and
  `test_dir_named_like_container` encounters `dataset requires lazy=True`.
- Ruff lint, formatting, and diff whitespace checks pass for the changed files.
