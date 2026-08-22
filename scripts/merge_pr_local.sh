#!/bin/sh
# Land a GitHub PR locally, never server-side.
#
# Server-side merges (web UI button, merge queue, and `gh pr merge`) create
# commits committed by GitHub <noreply@github.com> and are forbidden by this
# repo's identity policy. This script is the only supported merge path.
#
# Usage:
#   scripts/merge_pr_local.sh <pr-number>            # internal PR, merge --no-ff
#   scripts/merge_pr_local.sh <pr-number> --squash   # external PR, squash-land
#
# Internal PRs (commits already carry the anonymous identity) are merged
# --no-ff so GitHub auto-marks the PR merged when the push lands. External
# PRs are squashed into a single commit under the anonymous identity (their
# original author identities are intentionally not preserved; policy in
# AGENTS.md), then the PR is closed with a pointer comment.
#
# Run from the publisher clone only: hooks must be active and the repo
# identity must be the anonymous one. Both are verified below. Hygiene is
# checked before the merge, after the merge, and again by the pre-push hook.
set -eu

PR="${1:?usage: merge_pr_local.sh <pr-number> [--squash]}"
MODE="${2:-merge}"

ROOT=$(git rev-parse --show-toplevel)
CHECK="$ROOT/scripts/check_public_hygiene.py"

[ "$(git config core.hooksPath)" = ".githooks" ] || {
    echo "abort: core.hooksPath is not .githooks; run from the publisher clone" >&2
    exit 1
}
[ "$(git config user.name) <$(git config user.email)>" = "some one <someone@example.com>" ] || {
    echo "abort: repo identity is not the anonymous identity" >&2
    exit 1
}
[ -z "$(git status --porcelain)" ] || {
    echo "abort: working tree not clean" >&2
    exit 1
}

BASE=$(gh pr view "$PR" --json baseRefName -q .baseRefName)
TITLE=$(gh pr view "$PR" --json title -q .title)

git checkout "$BASE"
git pull --ff-only origin "$BASE"
git fetch origin "pull/$PR/head:pr-$PR"

case "$MODE" in
merge)
    # Internal PR: every incoming commit must already be clean.
    python3 "$CHECK" "$BASE..pr-$PR"
    git merge --no-ff "pr-$PR" -m "Merge PR #$PR: $TITLE"
    ;;
--squash)
    git merge --squash "pr-$PR"
    git commit -m "PR #$PR (squashed): $TITLE"
    ;;
*)
    echo "abort: unknown mode $MODE" >&2
    exit 1
    ;;
esac

# Verify everything about to go out, then push (pre-push hook re-checks).
python3 "$CHECK" "origin/$BASE..$BASE"
git push origin "$BASE"

if [ "$MODE" = "--squash" ]; then
    gh pr close "$PR" --comment "Landed locally as $(git rev-parse --short "$BASE") (squashed per identity policy)."
fi
git branch -D "pr-$PR"
echo "PR #$PR landed on $BASE."
