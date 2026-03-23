# Containers Tutorial Plan

Target notebook:
`doc/getting_started/tutorials/13.containers.ipynb`

Goal:
Create a small but comprehensive tutorial for the main `python-blosc2` data containers, designed for reading and running as a Jupyter notebook. The style should be concept-first, visual, and practical, similar in spirit to the FFmpeg libav tutorial referenced by the user: short explanations, clear mental models, small runnable examples, and lightweight but expressive figures.

## Scope

The tutorial will cover these main containers, in this order:

1. `SChunk`
2. `NDArray`
3. `VLArray`
4. `BatchStore`
5. `EmbedStore`
6. `DictStore`
7. `TreeStore`
8. `C2Array`

Notes:

- `SChunk` comes first because it is the basis for the higher-level local containers.
- `Batch` is not part of the main list because it is a view returned by `BatchStore`, not a top-level container.
- `Batch` can still be mentioned briefly inside the `BatchStore` section.

## Proposed Table of Contents

1. Introduction
   Explain what “container” means in `python-blosc2` and what the tutorial covers.

2. The Big Picture
   Present a family overview of the containers and how they relate.

3. `SChunk`: The Foundation
   Explain that it is a sequence of compressed chunks plus metadata.
   Show why it is the storage basis for higher-level containers.

4. `NDArray`: Compressed N-D Arrays
   Explain that it builds array semantics on top of an `SChunk`.
   Cover slicing, chunking, persistence, and typical array workflows.

5. `VLArray`: Variable-Length Items
   Explain that it stores one serialized variable-length item per entry.
   Position it as the ragged/object-like container.

6. `BatchStore`: Batched Variable-Length Data
   Explain that it stores batches in compressed chunks, with optional block-local reads inside each batch.
   Position it for batch-oriented ingestion and access.

7. `EmbedStore`: Bundle Several Containers Into One Store
   Explain how it embeds serialized containers/nodes into one backing store.
   Position it for portability and packaging.

8. `DictStore`: Key-Value Collection of Containers
   Explain the directory/zip-backed keyed collection model.
   Position it for multi-object datasets.

9. `TreeStore`: Hierarchical Datasets
   Explain it as a hierarchical extension of `DictStore`.
   Position it for tree-structured datasets.

10. `C2Array`: Remote Arrays
    Explain it as a remote handle over Caterva2/HTTP.
    Position it for remote array access without full local copies.

11. Choosing the Right Container
    Provide a compact comparison table across the containers.

12. Final Notes
    Summarize common usage patterns and point to deeper documentation.

## Per-Section Template

Each main container section should follow the same pattern:

1. Short description
2. How it is implemented
3. What features it provides
4. What it is useful for
5. Small runnable code example
6. Small figure

This repetition should make the notebook predictable and easy to scan.

## Content Style

The notebook should aim for:

- short paragraphs
- direct language
- minimal theory beyond what is needed to build intuition
- small examples that run quickly
- progressive complexity
- visuals that reinforce the mental model rather than decorate the page

The notebook should avoid:

- long API reference dumps
- too many parameters in each example
- overly abstract prose
- figures that try to encode too much information at once

## Image Strategy

Images should be simple, consistent, and expressive.

Recommended visual grammar:

- deep blue outline: container
- dark yellow blocks: compressed chunks / payload pieces
- light blue strip: metadata
- dashed arrows: references or remote access
- folder/zip shapes only for store-like containers

Preferred palette, based on the Blosc2 logo:

- dark yellow: `#df9e00`
- light blue: `#007a86`
- deep blue: `#002a64`

Suggested mapping:

- deep blue (`#002a64`) for container outlines, titles, and main structural elements
- dark yellow (`#df9e00`) for chunk payload blocks and highlighted storage pieces
- light blue (`#007a86`) for metadata bands, secondary structure, and remote/reference accents

The figures do not need to use only these colors, but these three should define the visual identity of the container diagrams.

Recommended first-pass figure list:

1. Overview map
   Show the relationships among `SChunk`, `NDArray`, `VLArray`, `BatchStore`, `EmbedStore`, `DictStore`, `TreeStore`, and `C2Array`.

2. `SChunk`
   Show a sequence of compressed chunks plus metadata.

3. `NDArray`
   Show array semantics on top of chunked compressed storage.

4. `VLArray` vs `BatchStore`
   Side-by-side comparison:
   `VLArray` as one variable-sized item per entry;
   `BatchStore` as one chunk per batch with internal subdivision.

5. `EmbedStore` / `DictStore` / `TreeStore`
   Show the progression from embedded bundle to key-value store to hierarchical tree.

This should be enough to make the notebook visual without overproducing assets.

## Asset Format

Preferred format: SVG

Reasons:

- crisp rendering in notebooks
- easy to version-control
- easy to tweak
- lightweight and portable

Suggested asset location:

- `doc/getting_started/tutorials/images/containers/overview.svg`
- `doc/getting_started/tutorials/images/containers/schunk.svg`
- `doc/getting_started/tutorials/images/containers/ndarray.svg`
- `doc/getting_started/tutorials/images/containers/vlarray-batchstore.svg`
- `doc/getting_started/tutorials/images/containers/stores.svg`

## Collaboration Workflow For Images

Proposed workflow:

1. Draft a one-line image spec for each figure.
2. Review the metaphor and emphasis with the user.
3. Produce a simple SVG draft.
4. Review for clarity first, polish second.
5. Revise if the figure is visually clean but conceptually ambiguous.

The goal is not artistic polish. The goal is instant comprehension.

## Suggested Notebook Flow

The notebook itself should likely be built from cells in this order:

1. Title and short intro
2. Overview diagram
3. One short “family map” section
4. One section per container
5. Comparison table
6. Closing notes

For each container section:

- markdown cell with description and use cases
- markdown cell or callout with implementation notes
- code cell with tiny example
- markdown cell with figure

## Small Example Guidelines

Examples should be:

- short enough to fit in one notebook cell
- independent where possible
- fast to run
- focused on one idea

Examples should demonstrate:

- `SChunk`: append/get/decompress or basic chunk operations
- `NDArray`: create, persist, slice
- `VLArray`: append and retrieve variable-length items
- `BatchStore`: append a batch, iterate batches or items
- `EmbedStore`: put/get a couple of nodes
- `DictStore`: assign named entries
- `TreeStore`: assign hierarchical paths and traverse
- `C2Array`: open remote metadata and retrieve a small slice

For `C2Array`, the example may need extra care because it depends on remote access; if needed, keep it lightweight and make clear that it requires network access.

## Open Decisions For Next Iteration

These should be settled next:

1. Exact section titles and tone of the notebook.
2. The first-pass image specs, one by one.
3. Whether all code examples should be executable offline except `C2Array`.
4. Whether to include one summary table near the top, near the end, or both.
5. Whether to add one “common patterns” section showing how containers compose.

## First-Pass SVG Image Specs

These specs are meant to be simple enough to implement quickly, but concrete enough that the figures will already be useful in a first draft.

### 1. `overview.svg`

Purpose:

- Give the reader a fast mental map of the container family.

Core message:

- `SChunk` is the storage foundation.
- `NDArray`, `VLArray`, and `BatchStore` build on top of it.
- `EmbedStore`, `DictStore`, and `TreeStore` organize multiple containers.
- `C2Array` is the remote-facing member of the family.

Suggested layout:

- One central `SChunk` box.
- Three boxes above or to the right: `NDArray`, `VLArray`, `BatchStore`.
- Three store boxes further out: `EmbedStore`, `DictStore`, `TreeStore`.
- One separate remote box: `C2Array`.
- Solid arrows from `SChunk` to `NDArray`, `VLArray`, `BatchStore`.
- Solid arrow from `DictStore` to `TreeStore`.
- Dashed arrow between stores and `C2Array` to indicate references/remote links.

Suggested labels inside boxes:

- `SChunk`: “compressed chunks + metadata”
- `NDArray`: “N-D array semantics”
- `VLArray`: “one variable-length item per entry”
- `BatchStore`: “one batch per chunk”
- `EmbedStore`: “embedded nodes”
- `DictStore`: “named collection”
- `TreeStore`: “hierarchical collection”
- `C2Array`: “remote array handle”

Visual note:

- This should be the least detailed figure, optimized for orientation.

### 2. `schunk.svg`

Purpose:

- Show what an `SChunk` physically/conceptually looks like.

Core message:

- `SChunk` is a sequence of compressed chunks plus metadata.

Suggested layout:

- One large horizontal container box labeled `SChunk`.
- Green strip at the top or left labeled `meta / vlmeta`.
- Several orange rectangular blocks inside labeled `chunk 0`, `chunk 1`, `chunk 2`, `...`.
- Optional small caption under the box: “persistent or in-memory”.

Suggested callouts:

- “append/update/delete chunks”
- “compressed payload”
- “basis for higher-level containers”

Visual note:

- This should be the simplest figure of the set.

### 3. `ndarray.svg`

Purpose:

- Show that `NDArray` is an array interface over chunked compressed storage.

Core message:

- `NDArray` provides shape/dtype/slicing semantics on top of an `SChunk`.

Suggested layout:

- Top layer: a blue `NDArray` box with small labels:
  `shape`, `dtype`, `chunks`, `blocks`
- Under it: a simplified grid or 1D strip labeled “logical array view”.
- Under that: an `SChunk` box with orange chunk blocks.
- Arrow from `NDArray` to `SChunk`.

Suggested callouts:

- “array semantics”
- “slicing”
- “persistent `.b2nd` or in-memory”

Visual note:

- Show the distinction between logical array view and physical chunk storage.

### 4. `vlarray.svg`

Purpose:

- Show how `VLArray` differs from `NDArray` and why it fits variable-length values.

Core message:

- One logical entry maps to one independently compressed serialized payload.

Suggested layout:

- Left side: a vertical list labeled `VLArray` with entries like:
  `{"a": 1}`
  `"hello"`
  `[1, 2, 3, 4]`
  `b"..."`
- Right side: an `SChunk` with orange blocks of visibly different lengths.
- Arrows from each logical entry to one chunk block.

Suggested callouts:

- “serialize”
- “compress”
- “independent entries”

Visual note:

- Different block widths are important here to visually reinforce variable-length storage.

### 5. `vlarray-batchstore.svg`

Purpose:

- Compare `VLArray` and `BatchStore` directly.

Core message:

- `VLArray`: one item per chunk.
- `BatchStore`: one batch per chunk, possibly subdivided internally.

Suggested layout:

- Two panels side by side.

Left panel:

- `VLArray`
- Three logical entries, each mapping to one orange block.

Right panel:

- `BatchStore`
- Three logical batches, each mapping to one larger orange block.
- Inside each batch block, draw smaller subdivisions to suggest internal blocks/items.

Suggested callouts:

- left: “fine-grained item storage”
- right: “batch-oriented storage”

Visual note:

- This should make the contrast obvious at a glance.

### 6. `embedstore.svg`

Purpose:

- Show that `EmbedStore` bundles different nodes into one backing store.

Core message:

- Multiple containers are embedded into one portable store.

Suggested layout:

- One big blue outer box labeled `EmbedStore`.
- Inside it:
  one small `NDArray` node,
  one `SChunk` node,
  one `VLArray` or `BatchStore` node,
  one dashed-link node for `C2Array` reference.
- A green map/index strip on one side labeled `key -> offset/length`.

Suggested callouts:

- “single bundled store”
- “embedded serialized nodes”
- “remote references possible”

Visual note:

- Keep it compact; this image is about bundling, not hierarchy.

### 7. `dictstore.svg`

Purpose:

- Show `DictStore` as a named collection with embedded and external leaves.

Core message:

- `DictStore` organizes multiple named objects in a directory/zip-like structure.

Suggested layout:

- Folder or zip-shaped outer boundary labeled `DictStore`.
- Inside:
  one embedded file-like box labeled `embed.b2e`,
  a few external leaves with names like `a.b2nd`, `b.b2b`, `c.b2f`.
- A short list of sample keys on the left:
  `/a`, `/b`, `/c`

Suggested callouts:

- “named collection”
- “embedded + external storage”
- “`.b2d` / `.b2z`”

Visual note:

- This figure should emphasize storage organization rather than data layout.

### 8. `stores.svg`

Purpose:

- Show the progression from `EmbedStore` to `DictStore` to `TreeStore`.

Core message:

- The stores differ mainly in how they organize multiple objects.

Suggested layout:

- Three panels left-to-right:
  `EmbedStore` -> `DictStore` -> `TreeStore`
- `EmbedStore`: simple bundle
- `DictStore`: flat named collection
- `TreeStore`: hierarchical `/group/subgroup/node`

Suggested callouts:

- `EmbedStore`: “bundle”
- `DictStore`: “flat keys”
- `TreeStore`: “hierarchical keys”

Visual note:

- This should help readers understand why both `DictStore` and `TreeStore` exist.

### 9. `c2array.svg`

Purpose:

- Show `C2Array` as a remote array handle.

Core message:

- `C2Array` does not own local storage in the same way; it points to remote array data and fetches metadata/slices on demand.

Suggested layout:

- Left: local client box labeled `C2Array`.
- Middle: dashed network arrow labeled `HTTP`.
- Right: remote service/cloud box containing a remote array rectangle.
- Optional small metadata card near the client:
  `shape`, `dtype`, `chunks`

Suggested callouts:

- “remote metadata”
- “remote slice fetch”
- “Caterva2-backed”

Visual note:

- Keep this visually distinct from the local-storage figures.

## Recommended Next Steps

1. Finalize the exact notebook outline and section titles.
2. Review and refine these image specs.
3. Create the notebook skeleton with markdown headings and placeholder figure cells.
4. Fill in the runnable examples.
5. Add the SVG assets.
6. Refine the narrative and transitions between sections.
