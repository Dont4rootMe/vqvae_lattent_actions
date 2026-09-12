#!/usr/bin/env bash
# Upload Comet offline archives (written when compute nodes have no internet) from a node that does.
source "$(dirname "$0")/common.sh"
DIR="${1:?usage: comet_upload.sh <directory with *.zip>}"
for archive in "$DIR"/*.zip; do
  echo "[comet] uploading $(basename "$archive")"
  python -m comet_ml.scripts.comet_upload "$archive"
done
