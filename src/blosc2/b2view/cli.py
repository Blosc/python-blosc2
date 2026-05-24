"""Command line entry point for b2view."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browse a Blosc2 TreeStore bundle in the terminal.")
    parser.add_argument("urlpath", help="Path to a .b2d directory or .b2z file")
    parser.add_argument("--preview-rows", type=int, default=20, help="Maximum preview rows")
    parser.add_argument("--preview-cols", type=int, default=10, help="Maximum preview columns")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from blosc2.b2view.app import B2ViewApp
    except ImportError as exc:
        print(
            "b2view requires the optional TUI dependencies. Install them with:\n"
            "\n"
            '    pip install "blosc2[tui]"\n',
            file=sys.stderr,
        )
        print(f"Original import error: {exc}", file=sys.stderr)
        return 2

    app = B2ViewApp(args.urlpath, preview_rows=args.preview_rows, preview_cols=args.preview_cols)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
