#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_USER="uvv78gt"
REMOTE_HOST="jean-zay.idris.fr"
REMOTE_PATH="/lustre/fswork/projects/rech/nxk/uvv78gt/deCIFer"

EXCLUDES=(
  "__pycache__"
  ".mypy_cache"
  ".pytest_cache"
  "*.pyc"
  ".DS_Store"
  ".idea"
  ".vscode"
  "venv"
  ".venv"
)

rsync_opts=(-avh --info=progress2 --filter=':- .gitignore')

for arg in "$@"; do
  case "$arg" in
    --delete)
      rsync_opts+=(--delete)
      ;;
    --dry-run|-n)
      rsync_opts+=(--dry-run)
      ;;
    *)
      echo "Option inconnue: ${arg}" >&2
      echo "Utilisation: $0 [--delete] [--dry-run]" >&2
      exit 1
      ;;
  esac
done

for pattern in "${EXCLUDES[@]}"; do
  rsync_opts+=(--exclude "$pattern")
done

echo "Syncing ${REPO_ROOT} -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
if [[ " ${rsync_opts[*]} " == *" --dry-run "* ]]; then
  echo "(dry run: aucun fichier ne sera transféré)"
fi
rsync "${rsync_opts[@]}" "${REPO_ROOT}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
