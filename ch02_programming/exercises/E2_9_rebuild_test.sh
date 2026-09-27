#!/usr/bin/env bash
# E2.9  Delete the environment, rebuild it from requirements.txt, rerun -- three times.
# Works on a COPY of E2_11_solution in a temporary folder; needs internet for pip.
#   bash exercises/E2_9_rebuild_test.sh
set -e
SRC="$(cd "$(dirname "$0")/E2_11_solution" && pwd)"
WORK=$(mktemp -d)
cp "$SRC"/sim.py "$SRC"/requirements.txt "$WORK"/
cd "$WORK"
for run in 1 2 3; do
  rm -rf .venv
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
  echo "--- run $run"
  .venv/bin/python sim.py | grep -v saved | tee "run$run.txt"
done
if cmp -s run1.txt run2.txt && cmp -s run2.txt run3.txt; then
  echo "all three rebuilds give identical numbers"
else
  echo "numbers DIFFER between rebuilds -> check the five layers in tab:repro-layers"
fi
