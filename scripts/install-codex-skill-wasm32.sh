#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install-codex-skill-wasm32.sh [--force] [--copy]

Install the repo-local wasm32 skill into Codex skill discovery path.

Options:
  --force   Replace existing destination skill if present
  --copy    Copy files instead of creating a symlink
EOF
}

force=0
copy_mode=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) force=1 ;;
    --copy) copy_mode=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_skill="${repo_root}/.codex/skills/wasm32-pyodide-dev"
codex_home="${CODEX_HOME:-${HOME}/.codex}"
dst_skill="${codex_home}/skills/wasm32-pyodide-dev"

if [[ ! -d "${src_skill}" ]]; then
  echo "Source skill not found: ${src_skill}" >&2
  exit 1
fi

mkdir -p "$(dirname "${dst_skill}")"

if [[ -e "${dst_skill}" || -L "${dst_skill}" ]]; then
  if [[ -L "${dst_skill}" ]]; then
    current="$(readlink -f "${dst_skill}" || true)"
    expected="$(readlink -f "${src_skill}" || true)"
    if [[ "${current}" == "${expected}" ]]; then
      echo "Skill already installed at ${dst_skill}"
      exit 0
    fi
  fi

  if [[ "${force}" -ne 1 ]]; then
    echo "Destination exists: ${dst_skill}" >&2
    echo "Re-run with --force to replace it." >&2
    exit 1
  fi

  rm -rf "${dst_skill}"
fi

if [[ "${copy_mode}" -eq 1 ]]; then
  cp -a "${src_skill}" "${dst_skill}"
  echo "Copied skill to ${dst_skill}"
else
  ln -s "${src_skill}" "${dst_skill}"
  echo "Linked skill to ${dst_skill}"
fi
