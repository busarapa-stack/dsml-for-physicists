#!/usr/bin/env bash
# Ch.2  code:git-flow -- one full Git cycle in a throw-away folder (safe to run).
#   bash shell/git_flow_demo.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEMO=$(mktemp -d)
cd "$DEMO"
git init -q -b main
git config user.name  "Student"
git config user.email "student@example.com"

echo "# physics-ch02 demo" > README.md
echo 'print("hello")'      > sim.py
cp "$HERE/gitignore_python.txt" .gitignore

git add README.md sim.py .gitignore
git commit -q -m "Initial scaffold"

# edit sim.py, then record again
cat >> sim.py <<'PY'
import numpy as np
omega = 2 * np.pi
PY
git add sim.py
git commit -q -m "Implement undamped HO with energy check"

mkdir -p .venv && touch .venv/should_not_be_tracked
echo "--- git status (!! = ignored by .gitignore, so never committed):"
git status --short --ignored
echo "--- git log --oneline:"
git log --oneline
echo "--- git diff HEAD~1 -- sim.py:"
git diff HEAD~1 -- sim.py
echo "(demo repository in $DEMO)"
