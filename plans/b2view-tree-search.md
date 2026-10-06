# b2view tree search

Implemented following user authorization. F with tree focus (or global Ctrl+F)
opens a filtered mirror of
discovered nodes; case-insensitive path filtering performs no I/O. Ancestors
remain visible, and closing restores the unchanged main tree selection/expansion.

Explicit recursive search traverses existing StoreBrowser listings breadth-first
on a worker, reusing known listings. Local, Caterva2 and fsspec roots use the same
implementation. It does not call preview/data access: necessary discovery
metadata (including empty Caterva2 table schema frames) is allowed.

Limits: 200 directories, 10,000 nodes, depth 12; the final atomic listing may
overshoot the node count. Only 500 matches are rendered at once. Progress is
throttled to keep result-tree rebuilding responsive. Cancellation and session
guards reject obsolete callbacks/results; in-flight provider calls cannot be
forcibly interrupted. Errors and partial-search status are reported explicitly.

Selecting a result merges discovered listings and reveals the node using normal
navigation. Closing retains delivered discoveries without expanding the main
tree; refresh invalidates extra discovery snapshots. No backend-specific global
catalog optimization or filename-content/full-text search is introduced.
