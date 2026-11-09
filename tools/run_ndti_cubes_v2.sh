#!/bin/bash
# Local worker pool for NDTI harvest cubes. Usage: N=5 bash tools/run_ndti_cubes.sh <tasks.txt>
# tasks.txt holds one "block season" pair per line. Finished cubes are skipped (ndti_local.py checks).
HERE="$(cd "$(dirname "$0")" && pwd)"
TASKS="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
cd "$HERE/../pipelines"
export AWS_NO_SIGN_REQUEST=YES GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR GDAL_HTTP_MAX_RETRY=8 GDAL_HTTP_RETRY_DELAY=3
mkdir -p ../logs/ndti_v2
N=${N:-5}
xargs -P "$N" -L 1 sh -c 'for try in 1 2 3; do python ndti_cube_v2.py --block $0 --season $1 > ../logs/ndti_v2/blk_$0_$1.log 2>&1 && { echo "ok $0 $1"; exit 0; }; sleep 20; done; echo "FAILED $0 $1"' < "$TASKS"
echo "ALL NDTI CUBES FINISHED"
