#!/bin/bash
# Daily KOSPI forward-sim routine.
# Designed to be invoked by Claude Code Routines, a cron job, or by hand.
# Idempotent on a fresh clone: re-initialises any missing sessions before running.

set -euo pipefail

cd "$(dirname "$0")/.."

# Make sure the jusik CLI is on PATH; install in editable mode if not.
if ! command -v jusik >/dev/null 2>&1; then
    pip install -e . >/dev/null
fi

bash scripts/init_recommended.sh

jusik forward run-all

# Stage only the tracked artifacts (session state + daily reports).
git add results/forward reports || true

if git diff --staged --quiet; then
    echo "no changes to commit"
    exit 0
fi

git -c user.email="${GIT_EMAIL:-claude-routine@anthropic.com}" \
    -c user.name="${GIT_NAME:-claude-routine}" \
    commit -m "chore: daily forward run $(date -u +%F)"

# Push to whichever branch HEAD is on. Routines clone main by default; if
# unrestricted branch pushes are enabled the commit lands on main.
branch="$(git rev-parse --abbrev-ref HEAD)"
for attempt in 1 2 3 4; do
    if git push -u origin "$branch"; then
        exit 0
    fi
    sleep $((2 ** attempt))
done
echo "push failed after retries" >&2
exit 1
