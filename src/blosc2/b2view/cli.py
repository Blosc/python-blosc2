"""Command line entry point for b2view."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browse a Blosc2 TreeStore bundle in the terminal.")
    parser.add_argument("urlpath", help="Path to a .b2d directory or .b2z file")
    parser.add_argument("path", nargs="?", default="/", help="Optional starting path inside the bundle")
    parser.add_argument("--preview-rows", type=int, default=20, help="Maximum preview rows")
    parser.add_argument("--preview-cols", type=int, default=10, help="Maximum preview columns")
    parser.add_argument(
        "--panel",
        choices=["tree", "meta", "vlmeta", "data"],
        default="tree",
        help="Panel to focus on startup",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="Capture the mouse for clicking and scrolling (disables the terminal's native text selection)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import blosc2

    if blosc2.IS_WASM:
        print(
            "b2view is an interactive terminal UI and is not supported in the "
            "Pyodide/WebAssembly build of blosc2:\nthere is no terminal driver "
            "(termios) available in this environment.\n"
            "Run b2view from a native (CPython) install instead.",
            file=sys.stderr,
        )
        return 1

    try:
        from blosc2.b2view.app import B2ViewApp
    except ImportError as exc:
        print(
            "b2view could not import its TUI dependencies. Install them with:\n\n    pip install textual\n",
            file=sys.stderr,
        )
        print(f"Original import error: {exc}", file=sys.stderr)
        return 2

    app = B2ViewApp(
        args.urlpath,
        start_path=args.path,
        start_panel=args.panel,
        preview_rows=args.preview_rows,
        preview_cols=args.preview_cols,
    )
    app.run(mouse=args.mouse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
