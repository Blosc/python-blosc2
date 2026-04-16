# Plan: Containers Tutorial

Target notebook:
`doc/getting_started/tutorials/13.containers.ipynb`

## Goal

Land a solid v1 tutorial that gives users a practical mental map of the main
`python-blosc2` containers, with short runnable examples and a small number of
supporting diagrams.

This tutorial should answer:

- what each container is
- how the containers relate to each other
- which container to choose for a given workflow

## Scope For V1

Included sections:

1. `SChunk`
2. `NDArray`
3. `VLArray`
4. `BatchArray`
5. `EmbedStore`
6. `DictStore`
7. `TreeStore`
8. `C2Array`

V1 constraints:

- every local-container section gets a short runnable example
- `C2Array` remains offline-safe by default and does not fetch remote data
- only a small number of diagrams are required
- the notebook must be linked from `doc/getting_started/tutorials.rst`

## Current Status

- `13.containers.ipynb` now has runnable v1 content for the local containers
- `overview.svg` is now present under `doc/getting_started/tutorials/images/containers/`
- `doc/getting_started/tutorials.rst` now indexes tutorials `12`, `13`, and `14`
- the notebook keeps `C2Array` offline-safe by default

## Implementation Plan

### Phase 1: Make The Tutorial Landable

- add `plans/containers-tutorial.md` as the live plan and progress record
- index `12.batcharray`, `13.containers`, and `14.indexing-arrays` in
  `doc/getting_started/tutorials.rst`
- keep the tutorial focused on current APIs only

### Phase 2: Replace Placeholder Cells

- add a shared setup cell for imports, temp paths, and cleanup helpers
- replace every `TODO` code cell with a compact runnable example
- use local temp paths so repeated notebook runs stay deterministic

Planned examples:

- `SChunk`: create, append chunks, inspect chunk counts and metadata
- `NDArray`: create, persist, slice, reopen
- `VLArray`: append variable-length values, inspect entries, reopen
- `BatchArray`: append batches, inspect per-batch and item-level access
- `EmbedStore`: bundle a couple of objects in one container
- `DictStore`: store named leaves and reopen them
- `TreeStore`: create a small hierarchy and walk a subtree
- `C2Array`: show the URLPath/open pattern with remote access disabled by default

### Phase 3: Trim The Visual Scope

- keep `overview.svg` as the main family diagram
- add one lightweight store-oriented diagram if needed
- remove per-section placeholder figure text that would otherwise make the
  notebook feel unfinished

### Phase 4: Verify

- run the notebook or equivalent code path checks locally
- confirm that all image paths resolve
- confirm that the tutorial appears in the rendered docs index

## Benefits

This tutorial is worth continuing because it fills a real gap:

- current docs have single-feature tutorials, but not a family overview
- users can see how `SChunk`, array containers, and store containers fit together
- the comparison section helps users choose the right container earlier
- it should reduce confusion around when to use `VLArray`, `BatchArray`,
  `EmbedStore`, `DictStore`, or `TreeStore`

## Progress Log

### 2026-04-12

- decided to keep a narrow v1 scope
- decided to favor runnable examples over a large diagram set
- decided to keep `C2Array` offline-safe by default
- completed:
  - added `plans/containers-tutorial.md`
  - indexed `12.batcharray`, `13.containers`, and `14.indexing-arrays`
  - replaced placeholder cells in `13.containers.ipynb` with runnable examples
  - added `images/containers/overview.svg`
  - added a reduced-scope asset README
- verification:
  - all notebook code cells executed successfully in a direct Python pass
  - `jupyter nbconvert --execute` could not be used in the sandbox because the
    Jupyter kernel could not bind local ports
