#!/usr/bin/env bash
set -euo pipefail


BUCKET="gs://flyem-optic-lobe/v1.1"
DEST="flyem_optic_lobe_v1_1"

mkdir -p "$DEST"

echo "Listing top-level objects in $BUCKET ..."
gsutil ls "$BUCKET/"

echo "\nTo download a specific object or prefix, run e.g.:"
echo "  gsutil -m cp -r $BUCKET/<prefix_or_file> $DEST/"
