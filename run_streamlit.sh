#!/usr/bin/env bash
# ==============================================================================
# Launch Streamlit Oral Reading Fluency & Articulation Assessment Web App
# ==============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PORT=8501
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port|-p)
      PORT="$2"
      shift 2
      ;;
    [0-9]*)
      PORT="$1"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "Starting Streamlit ASR Reading Fluency Assessment App on port ${PORT}..."
./.venv/bin/python -m streamlit run streamlit/app.py --server.port "${PORT}" "${EXTRA_ARGS[@]}"
