#!/bin/sh
# Land a GitHub PR locally, never server-side.
#
# Server-side merges (web UI button, merge queue, and `gh pr merge`) create
# commits committed by GitHub <noreply@github.com> and are forbidden by this
# repo's identity policy. This script is the only supported merge path.
#
# Usage:
#   scripts/merge_pr_local.sh <pr-number>            # internal PR, exact fast-forward
#   scripts/merge_pr_local.sh <pr-number> --squash   # external PR, squash-land
#
# Internal PRs (commits already carry the anonymous identity) are landed by
# fast-forwarding the base to the exact reviewed PR head. External
# PRs are squashed into a single commit under the anonymous identity (their
# original author identities are intentionally not preserved; policy in
# AGENTS.md). Closing/commenting on that external PR is a separate publisher-
# account action; this script will not fall back to the operator's `gh` account.
#
# Run from the publisher clone only: hooks must be active and the repo
# identity must be the anonymous one. Both are verified below. Hygiene is
# checked before the merge, after the merge, and again by the pre-push hook.
set -eu

PR="${1:?usage: merge_pr_local.sh <pr-number> [--squash]}"
MODE="${2:-merge}"

ROOT=$(git rev-parse --show-toplevel)
CHECK="$ROOT/scripts/check_public_hygiene.py"
PUBLISHER_CHECK="$ROOT/scripts/check_publisher_context.py"

python3 "$PUBLISHER_CHECK" --remote origin

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
EXPECTED_HEAD=$(gh pr view "$PR" --json headRefOid -q .headRefOid)
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

wait_for_required_check_registration() {
    if ! required=$(gh api \
        "repos/$REPO/branches/$BASE/protection/required_status_checks" \
        --jq '.contexts[]'); then
        echo "abort: could not read required checks for $BASE" >&2
        return 1
    fi
    if [ -z "$required" ]; then
        echo "abort: no required checks are configured for $BASE" >&2
        return 1
    fi
    expected_required='hygiene
Rust tests passed'
    while IFS= read -r context; do
        if ! printf '%s\n' "$required" | grep -Fqx -- "$context"; then
            echo "abort: configured required check is missing: $context" >&2
            return 1
        fi
    done <<EOF
$expected_required
EOF

    timeout=${CHECK_WAIT_TIMEOUT_SECONDS:-1800}
    deadline=$(($(date +%s) + timeout))
    while :; do
        observed=$(gh pr checks "$PR" --required --json name \
            --jq '.[].name' 2>/dev/null || true)
        missing=""
        while IFS= read -r context; do
            [ -n "$context" ] || continue
            if ! printf '%s\n' "$observed" | grep -Fqx -- "$context"; then
                missing="${missing}${missing:+, }$context"
            fi
        done <<EOF
$required
EOF
        [ -z "$missing" ] && return 0
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "abort: required checks never registered: $missing" >&2
            return 1
        fi
        echo "waiting for required checks to register: $missing"
        sleep 5
    done
}

# Do not turn a not-yet-green push into a reason to use another landing path.
wait_for_required_check_registration
gh pr checks "$PR" --required --watch

git checkout "$BASE"
git pull --ff-only origin "$BASE"
git fetch origin "+pull/$PR/head:pr-$PR"
[ "$(git rev-parse "pr-$PR")" = "$EXPECTED_HEAD" ] || {
    echo "abort: PR head moved after required checks completed" >&2
    exit 1
}

case "$MODE" in
merge)
    # Internal PR: every incoming commit must already be clean.
    python3 "$CHECK" "$BASE..pr-$PR"
    git merge-base --is-ancestor "$BASE" "pr-$PR" || {
        echo "abort: base is not an ancestor of the reviewed PR head" >&2
        exit 1
    }
    [ -z "$(git rev-list --merges "$BASE..pr-$PR")" ] || {
        echo "abort: reviewed PR range contains a merge commit" >&2
        exit 1
    }
    git merge --ff-only "pr-$PR"
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

if ! lock_enabled=$(gh api "repos/$REPO/branches/$BASE/protection" \
    --jq '.lock_branch.enabled // false'); then
    echo "abort: could not read branch-lock state for $BASE" >&2
    exit 1
fi
[ "$lock_enabled" != true ] || {
    echo "abort: $BASE is locked; configure the controlled publisher bypass" >&2
    exit 1
}
git push origin "$BASE"

if [ "$MODE" = "--squash" ]; then
    echo "External PR landed locally; close/comment only from the publisher account."
fi
git branch -D "pr-$PR"
echo "PR #$PR landed on $BASE."
