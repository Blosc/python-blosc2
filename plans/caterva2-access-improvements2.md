# Caterva2 ordinary-file access and previews in b2view

Status: M0–M5 implemented and validated on `cat2-improvements` following explicit
user authorization. The design below is retained as the delivery/acceptance record.
It follows `plans/caterva2-access-improvements.md` and its completed URL discovery,
lazy hierarchy browsing, and viewer integration work.

## Implementation record

- Notebook follow-up: passive nbformat-4 cell views for local/fsspec/Caterva2
  files and local `.ipynb.b2` carriers. Markdown/code/raw cells and saved text
  outputs share 64 KiB / 1,000 displayed lines and 100-cell limits; complete JSON
  input is capped at 2 MiB. T toggles a bounded raw prefix. No Jupyter/kernel
  dependency, execution, HTML/JavaScript, widget or embedded-media rendering.

- Compressed local document follow-up: supported text/image/PDF names ending in
  `.b2` are read-only mapped SChunk byte streams. Metadata stays payload-free;
  text prefixes decode only intersecting chunks, images retain input/pixel caps,
  and copies decode/write incrementally under the original filename. Lazy chunk
  headers enforce 8 MiB compressed / 16 MiB decoded limits before payload copies.
  Native arrays/serialized objects/references are not document carriers; plain
  `.b2`/`.b2frame` dataset behavior is unchanged. Direct-fsspec carriers remain
  separate work. No original file content is deserialized or executed.

- Follow-up interface consistency: fallback panels have separate unavailable,
  missing-dependency and failed-preview states, with a common status header,
  reason and capability-aware action hints. Successful text previews share the
  same hints (O only for supported external document types; T only for Markdown).
  Read failures preserve file metadata and redact transport URLs/option values.
  Local/fsspec ordinary-file support is delivered as a separate viewer-only
  follow-up: direct regular-file inputs, shared passive previews/actions,
  background I/O, known-size bounded reads and streamed atomic copies. Native
  container formats retain their openers; unknown names get a small frame-magic
  check. Generic directories, hierarchy export and new public RemoteFile
  backends remain outside this follow-up. Backend caches are not Caterva2 caches.
  Earlier local/fsspec exclusions in this delivery record are superseded for
  direct ordinary-file inputs only.

- User-requested directory follow-up: ordinary local/file-URL/fsspec directories
  now expose immediate children lazily, with file previews/actions and dataset
  containers mounted as browsable subtrees. Native dataset directories preserve
  their opener. Parent traversal and local symlink following are refused.
  Earlier generic-directory exclusions are superseded by this follow-up.

- Follow-up UX decision: the extra trust checkbox is removed for all supported
  documents, including images and PDFs. The explicit O action and destination
  submission initiate external opening; signature/type restrictions and stale
  request/shutdown guards remain. Earlier checkbox requirements below are
  superseded by this user-requested decision.

- M0: verified Caterva2 demo README transport: `api/chunk?nchunk=0` returns
  a 552-byte compressed chunk that decompresses to the original 811 bytes;
  `api/download` returns original Markdown bytes, not the `.b2` carrier.
- API selected: `RemoteFile(RemoteObject)` with bounded `read_bytes`, streamed
  original-byte `download`, shared owner/cache/auth lifecycle, and unsupported
  reference persistence. Fixed-chunk SChunks are accepted, including typed byte
  payloads; irregular layouts and oversized chunks are refused explicitly.
- M1: recognition precedes array fallback; file handles participate in root and
  direct-leaf opening. Existing exact-payload cache coordination/disk machinery
  is reused with hash-separated file-chunk keys. Listing and metadata do not
  fetch payload. Downloads stage on the destination filesystem and publish
  atomically with no overwrite by default. Focused transport/access regression
  validation: 62 tests passed in the blosc2 environment.
- M2–M4: b2view recognizes file leaves, renders passive bounded text/Markdown
  (64 KiB / 1,000 lines), toggles raw text, and decodes JPEG/PNG under independent
  file/chunk/pixel limits. Image widgets reuse textual-image without matplotlib;
  absent dependencies have download/open fallbacks. PDFs require no renderer:
  D downloads original bytes and O prompts for destination and explicit trust
  consent before a shell-free platform launch. All fetch/decode/download work is
  backgrounded; downloads use independent handles and cancellation. Viewer/model
  plus file transport regressions: 175 passed with TUI cases enabled. Live demo
  checks: README renders; the JPEG decodes as 2034 × 1144 (bounded preview 1600 ×
  900); PDF offers download/external opening without fetching content on selection.
- M5/final review: added API/guide/installation/release documentation and the
  `images` extra (Pillow/textual-image, no matplotlib; included by hires). Real
  cat2lite native `.b2frame` file transport/download passes alongside existing
  remote-format/shared-cache acceptance. Native cat2lite exposes this carrier
  name; arbitrary ordinary-file publication/aliasing is not added to that server.
- Review hardening: bounded network and cached chunk reads before decompression;
  identity HTTP encoding, content-length/body caps, 10-second I/O timeouts and a
  checked 20-second chunk deadline; independent transfer aliases prepared on a
  background worker; no launcher after stale selection/session or shutdown.
  Old unsupported file snapshots rediscover metadata. Added download publication
  race, cancellation, empty-stream, corruption, cache eviction/refresh, auth,
  missing image dependency, consent, and responsive slow-transfer regressions.
- Final validation: default suite 10,670 passed, 38 skipped; focused offline
  access/file/viewer/model suite 230 passed with headless TUI cases enabled;
  opt-in live Caterva2 demo file downloads passed; both actual cat2lite acceptance
  tests passed. Ruff/diff checks passed. HTML docs built using the existing
  type-comment/notebook/generated-copy workarounds, with 764 wider autosummary/
  theme/cross-reference warnings; the build is not warnings-clean.
- Remaining constraints: fixed-chunk byte streams only; 8 MiB compressed / 16 MiB
  decoded chunk caps apply to downloads too, so large single-chunk files need
  server-side rechunking. No-overwrite downloads require hard-link-capable
  destination filesystems. File reference persistence/hierarchy export is deferred.
  MIME is a filename hint; binary/unknown files have no automatic preview. Image
  decoder copies are not a whole-process memory limit. External launchers depend
  on platform/GUI associations and are not a sandbox; tests mock launchers, never
  open documents automatically. User-selected downloads are retained (no hidden
  temporary external-viewer copies to manage). Local/fsspec regular-file support
  and in-terminal PDF rendering remain explicit non-goals.
- Live headless image integration also verified the real demo JPEG mounts an
  `AutoImage` widget with the installed textual-image package (not a mocked
  renderer). Explicit downloads remain bounded even when previews are unavailable;
  no automatic raw-download endpoint fallback bypasses chunk safety limits.

## 1. Goal and delivery order

Browse ordinary files published by Caterva2-compatible services without treating
them as unsupported array/table objects. Preserve the current hierarchy, root
conventions, lazy discovery, authentication, and client/server cache separation.

Deliver in this order:

1. Ordinary-file metadata, bounded byte access, explicit downloads, and text/
   Markdown previews.
2. Image previews using optional image dependencies and terminal capabilities.
3. PDF identification, download, and explicit external opening. Embedded PDF
   rendering is optional follow-up work, not required for this plan's completion.

Every file remains downloadable even if its format cannot be previewed. Do not
make downloading an entire file an implicit prerequisite for listing a directory.

Target viewer behavior:

| Selection | Metadata/data behavior |
| --- | --- |
| `@public/examples/README.md` | File metadata and bounded Markdown/text preview |
| `@public/examples/Wutujing-River.jpg` | File metadata and size-limited image preview, or a clear fallback |
| `@public/examples/cat2cloud-brochure.pdf` | File metadata, PDF notice, download/external-open actions |
| Unknown binary file | File metadata, preview-unavailable notice, download action |
| Generic SChunk without ordinary-file semantics | Byte-stream metadata/download; no invented original-file type |

Paths displayed for `@` roots have no leading slash. Internal browser selection
paths remain unchanged. Local and direct fsspec regular-file support is outside
this first delivery; do not accidentally intercept existing direct-format URLs.

## 2. Current implementation and verified evidence

Relevant code:

- `src/blosc2/remote_store.py`: `_caterva2_kind`, discovery metadata, node lookup,
  leaf handles, owner lifecycle, cache coordination, and `RemoteNode`.
- `src/blosc2/schunk.py`: public `open` and Caterva2 leaf dispatch.
- `src/blosc2/c2array.py`: existing authentication and info/fetch/chunk transport.
- `src/blosc2/remote_object.py`: common remote-object contract.
- `src/blosc2/remote_repository.py`: independently owned repository roots.
- `src/blosc2/b2view/model.py`: object kinds, metadata, preview selection, leaf
  lifetime, and display paths.
- `src/blosc2/b2view/app.py`: worker requests, stale-result rejection, download
  UI, and optional `textual-image` support currently used by plot screens.
- `src/blosc2/b2view/render.py`: Rich metadata and data rendering.
- `tests/test_remote_caterva2.py`, `tests/test_caterva2_access.py`, and
  `tests/b2view/test_caterva2.py`: deterministic API/viewer fixtures.

The current `_caterva2_kind` accepts groups, NDArrays, and CTables. Metadata
without array shape/dtype or table schema consequently becomes `unsupported`.

The following live info responses were inspected during planning at
`https://cat2.cloud/demo/api/info/@public/examples/<name>`:

| Name | Uncompressed bytes | Compressed bytes | Chunks |
| --- | ---: | ---: | ---: |
| `README.md` | 811 | 552 | 1 |
| `Wutujing-River.jpg` | 724,498 | 723,929 | 1 |
| `cat2cloud-brochure.pdf` | 44,155 | 36,889 | 1 |

These responses expose SChunk fields (`nbytes`, `cbytes`, `chunksize`, `nchunks`,
`cparams`, `vlmeta`) and server-internal `.b2` backing paths, not NDArray metadata.
Do not display or construct download names from those internal filesystem paths.
These observations establish metadata shape, not full transport compatibility:
chunk and download semantics must still be verified before implementation.

## 3. M0 — Protocol characterization and API decisions

### Protocol matrix

Create loopback fixtures for an ordinary-file SChunk, a generic byte SChunk, an
NDArray, a table, an unknown object, and a group. Characterize:

- Explicit kind values where available, plus legacy SChunk-shaped info responses.
- `api/chunk` addressing and response framing for SChunks; final partial chunks,
  empty streams, and variable-length/irregular layouts.
- Whether bounded `api/fetch` selection uses byte positions or typed elements.
- `api/download` semantics: original file bytes versus compressed Blosc frame,
  suffix handling, response headers, streaming, and redirects.
- Metadata preservation for MIME type, logical filename, user attrs, and catalog
  annotations, without assuming those fields exist on legacy servers.
- Compatibility with an actual Caterva2 deployment and cat2lite. cat2lite has
  SChunk and download routes, but arbitrary native-file publication is not an
  assumed existing capability. Record unsupported server combinations explicitly.

Prefer chunk-based lazy reads if the protocol supports them reliably. Do not
fall back to whole-file fetches silently for a bounded text preview. If a server
cannot provide bounded work, expose metadata/download and explain the preview
limitation, or require explicit consent for a capped full-file fetch.

### Proposed public abstraction

Use a `RemoteFile` (or a clearly documented byte-stream equivalent) inheriting
`RemoteObject`, separate from array indexing and table operations. Resolve its
final public name and API in M0 before implementation; audit existing SChunk and
Proxy machinery for reusable transport/cache behavior.

Proposed contract:

- `source`, `attrs`, `traffic`, cache policy/allowance/usage, `close`, and context
  management follow existing remote-object conventions.
- `nbytes`, logical `name`, optional media type, and chunk/compressed-size metadata.
- Explicit bounded `read_bytes(start, stop)` returning original uncompressed bytes.
  Validate offsets and bounds; do not overload NDArray slicing semantics.
- Streaming `download(destination, overwrite=False)` writes the original payload,
  never an undocumented `.b2` carrier, with optional progress/cancellation hooks.
- Downloading bytes is distinct from `save` of a remote reference. Reference
  persistence/materialization is explicitly unsupported initially unless existing
  artifact machinery can support it safely within scope.
- No writable source operations. Closing a root/repository leaves returned file
  handles usable, as with existing array/table handles.

Direct service leaf opening via `blosc2.open("https://host/@public/README.md")`
should return the same handle as hierarchy lookup. Preserve explicit URLPath
defaults: define and test the nonlazy byte-stream case rather than silently
changing all existing URLPath behavior.

## 4. M1 — Discovery, byte access, cache, and download

### Recognition and metadata

- Recognize explicit file/SChunk kinds and strictly validated legacy SChunk
  metadata. Classify NDArray/CTable metadata first to avoid confusing their
  embedded chunk metadata with a file.
- Validate nonnegative byte counts, chunk counts, and consistent layouts. Reject
  booleans masquerading as counts, invalid compression metadata, and impossible
  lengths. Do not assume every SChunk is a regular fixed-chunk byte stream.
- Separate the transport kind from preview format. A `.jpg` suffix does not make
  invalid metadata valid, nor guarantee that payload bytes are an image.
- Retain source attrs, catalog attrs, and file transport metadata separately.
- Preserve unknown metadata as an unsupported node with a useful diagnostic;
  an invalid file must not hide healthy siblings.

### Transport and resource ownership

- Reuse bound auth, deployment-prefix handling, safe URL components, transport
  pooling, and existing owner locks; never use a server-local `urlpath` as a URL.
- Fetch only necessary compressed chunks; validate frame/chunk headers and output
  length before allocating/decompressing. Charge actual response bytes once.
- State preview transfer bounds separately from output bounds: a 64 KiB prefix
  may require a much larger first chunk. Reject automatic preview work above
  compressed/decompressed chunk limits instead of claiming it is bounded.
- Streaming download uses bounded buffers/chunks and checks cancellation between
  reads. Enforce response and decompression limits on the streamed path, not only
  after loading a response into memory.
- Never deserialize Python objects or execute embedded metadata to read a file.

### Cache semantics

- `MEMORY`: share the owner's existing retained-payload allowance across file,
  array, and table leaves. Do not create a hidden per-file unlimited cache.
- `NONE`: retain no payload between operations. Transient preview/download
  buffers still have explicit limits and are not called persistent cache.
- `DISK`: reuse source/auth identity, exclusive ownership, atomic publication,
  and eviction conventions. Test reopened warm chunks and partial downloads.
- Keep client caching distinct from gateway caching. Repository budgets remain
  per root. Immutable-source assumptions and explicit invalidation remain.
- Preview render buffers/decoded images have a separate, documented lifecycle
  and limit; they are not silently counted as compressed chunk-cache bytes.

### Safe downloads

- Destination is chosen by the user. Sanitize suggested names from logical paths,
  not Content-Disposition or server-local paths; reject traversal and separators.
- No overwrite without confirmation; publish only a completed download. Use a
  staging file on the destination filesystem and a no-clobber publication strategy
  when overwrite is false, including concurrent-creation tests.
- Failure/cancellation cleans up only this operation's staging file. Leave prior
  destination contents untouched. Offer actionable permission/disk-full errors.
- Download the original `.md`, `.jpg`, or `.pdf` bytes; expose compressed-carrier
  export only as a separately named future operation if needed.

Acceptance: list/inspect without payload reads; exact arbitrary byte ranges and
streamed original-file downloads; mixed-leaf owner/cache/lifecycle tests pass.

## 5. M2 — Text/Markdown previews and viewer actions

Add a file node kind/icon, lightweight file metadata, and a preview result type
separate from numerical array/table previews. Do not route files through grid
paging, table filters, plotting, or scalar-array coercion.

Suggested initial limits (finalize constants in M0 and document them):

- Automatic text prefix: 64 KiB of original bytes, at most 1,000 displayed lines.
- Automatic chunk work: at most 8 MiB compressed and 16 MiB decompressed per
  chunk; higher limits require explicit user action, not silent escalation.
- Download: streamed rather than held whole in RAM; explicit consent includes
  known original size and destination. Unknown size gets a warning and progress
  bytes without a fabricated percentage.

Decode UTF-8/UTF-8 BOM first, with incremental decoding for a prefix that cuts a
multibyte character. Detect binary/NUL-heavy content and avoid garbage previews.
Invalid text gets an honest replacement/encoding notice; do not guess encodings
aggressively. Escape terminal control/ANSI sequences, including OSC hyperlinks.

For Markdown, use Textual's existing Markdown capabilities without new mandatory
dependencies. Offer raw text fallback/toggle. Disable automatic embedded-image
fetches, local file access, and external-link launching: a preview must not issue
unrelated network requests. Mark truncated content, including unmatched fences;
rendering errors fall back to safe text rather than breaking the viewer.

Add explicit Download and Open externally actions. Audit existing keybindings
before selecting keys; document them in help and expose enabled/disabled states.
For PDF/unknown binary files, these actions are available even without a preview.

All remote fetch/decode/download work runs outside the UI thread. Bind results
to session, selection, and request IDs. Stale results must not change the visible
preview or launch an external application. Show progress/errors, preserve healthy
sibling browsing, and support retry. Shutdown releases all owned resources.

Acceptance: README renders at its published path, large text is visibly truncated,
binary files remain downloadable, and delayed/cancelled reads do not freeze or
overwrite the current selection.

## 6. M3 — Images

Reuse Pillow and the optional `textual-image` integration; do not require
matplotlib just to display an existing JPEG/PNG. Audit packaging extras so the
installation hint accurately describes the minimum image dependencies.

- Validate file signature/decoder result, not suffix alone. Start with JPEG/PNG;
  do not promise arbitrary image formats or SVG/remote-resource rendering.
- A normal image preview may require the complete original file. Suggested
  automatic original-payload cap: 16 MiB, with independent chunk/transfer bounds.
- Inspect dimensions before full decoding; suggested decoded-image budget:
  64 MiB with an explicit pixel cap. Treat Pillow decompression-bomb warnings as
  refusal and reject malformed/oversized inputs. Downsampling after decoding is
  not a substitute for a predecode limit.
- Correct EXIF orientation, aspect ratio, and resize-to-panel behavior. Show
  dimensions and detected format. Bound animated-image handling to a first frame.
- If dependencies or terminal protocols are unavailable, show metadata and a
  fallback notice with Download/Open externally. Do not leave a blank panel.
- Release image buffers/widgets on selection change and shutdown; no unbounded
  decoded-thumbnail cache. Preserve worker/stale-result checks during resize.

Acceptance: small JPEG/PNG fixtures display with image support enabled; missing
dependency/protocol and unsafe-image cases have useful, tested fallbacks.

## 7. M4 — PDFs and external opening

PDF support in this delivery means recognition, metadata, and convenient safe
download/opening, not a mandatory terminal PDF renderer. A PDF may be rendered
externally even when the terminal cannot show images.

- Identify PDF candidates from extension/media metadata and verify signature when
  bytes are read. Do not invoke a PDF parser just to list or inspect file size.
- Display "PDF preview unavailable; download or open externally" and actionable
  controls. Missing renderer is not an unsupported-file error.
- External opening is an explicit user action, never triggered by selection,
  refresh, MIME detection, or restoring a session.
- Prefer opening a completed local download, so bound service credentials never
  appear in command-line URLs. Use platform launchers with argument arrays and
  no shell (`open`, `xdg-open`, or the appropriate Windows API), check availability,
  and use a safe absolute filename. Test launcher behavior with mocks only.
- Confirm opening untrusted content. Never mark downloads executable or dispatch
  unknown scripts/binaries to a launcher; external-open initially has a narrow
  document/image allowlist. Download remains available for other file types.
- Distinguish user-owned downloads from application-owned temporary copies.
  External viewers can outlive b2view: document retention/cleanup and do not
  delete a temporary file immediately after launch. Use a private session/temp
  directory with cleanup of owned stale copies under an explicit policy.
- Report no-GUI/headless/missing-launcher failures without losing the downloaded
  file; show its location and allow the user to open it manually.

Optional later milestone: PDF first-page thumbnails/text extraction using an
opt-in backend, with page/pixel/time limits and untrusted-parser isolation where
appropriate. This is not needed for M4 acceptance and must not become a mandatory
dependency for ordinary-file access.

## 8. M5 — Integration, documentation, and final review

Extend offline fixtures with Markdown, invalid UTF-8, arbitrary binary data,
JPEG/PNG, and PDF bytes, all transported as deterministic SChunks. Include empty,
partial final chunk, typed/irregular SChunk, false suffix, and malformed responses.

Acceptance matrix:

| Area | Required checks |
| --- | --- |
| Discovery | Legacy/explicit kinds; groups/arrays/tables unchanged; no payload on listing |
| Opening | String URL, URLPath, bare single/multi-root services, nested mounts; @ display paths |
| Byte access | Empty/prefix/cross-chunk/end ranges; invalid bounds; output-length mismatch |
| Cache | NONE/MEMORY/DISK; mixed-leaf budgets; eviction; reopened warm data; auth isolation |
| Lifecycle | Children outlive parents; refresh/stale handles; cancellation and concurrency |
| Text | Markdown/raw; UTF-8 boundary/BOM; binary/control characters; truncation; no linked fetches |
| Images | JPEG/PNG; missing dependency/protocol; corruption; pixel/decode caps; resize/release |
| Download | Original byte equality; overwrite race; unsafe names; cancelled/truncated response; disk failure |
| External open | Explicit consent; no shell; launcher unavailable; no secrets; stale launch suppressed |
| PDF | Useful nonrendering fallback; download/open without PDF dependency |
| UI | Headless worker responsiveness; failed leaf isolation; actions/help and existing grids unaffected |

Add opt-in real-service acceptance, not public-network dependency in the default
suite. Where supported, verify two independent clients can reuse gateway payload
cache; distinguish metadata rereads from payload rereads. Extend the actual
cat2lite acceptance only for supported publication/protocol behavior, not an
invented arbitrary-file server capability.

Update public class/open documentation, remote-object guide, b2view guide,
optional-extra installation instructions, and release notes. Include limits,
download versus reference export, platform opening behavior, and fallbacks.

Run Python/tests/build commands only in the `blosc2` conda environment. Run Ruff,
focused access/viewer tests (including headless TUI cases), the default suite,
and documentation checks. Preserve unrelated work and downloaded user files.

Final review focuses on:

1. Protocol assumptions validated on supported server versions.
2. Peak-memory/transfer limits, especially one-chunk large files and images.
3. Auth and path safety, unsafe content, and no implicit external side effects.
4. Shared cache accounting and returned-handle lifetime.
5. Worker cancellation, stale-result suppression, and safe atomic download.
6. Missing-dependency/headless/platform fallbacks and remaining limitations.

## 9. Known tradeoffs and non-goals

- Compression granularity may make a tiny byte prefix expensive. Preview refusal
  is preferable to an unbounded hidden download.
- MIME/suffix hints are imperfect; content recognition is bounded and conservative.
- Budgets bound retained compressed payload, not all decoder/transient RAM;
  preview/decompression limits must be enforced independently.
- External applications have their own security and lifecycle behavior; explicit
  consent is not a sandbox. Credentials must never be passed to them.
- No recursive directory download, archive extraction, file editing/upload,
  automatic script execution, OCR, animated playback, or embedded browser.
- No compulsory PDF renderer; download/external opening is a complete first
  delivery for PDF and terminal-unrenderable supported documents.
- Ordinary-file reference archives and full fsspec/local file-browser expansion
  are separate future work unless M0 proves a small, safe existing integration.
