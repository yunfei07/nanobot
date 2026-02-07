#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
BASE_BRANCH="${BASE_BRANCH:-main}"
SYNC_DATE="${SYNC_DATE:-$(date -u +%Y%m%d)}"
SYNC_BRANCH="${SYNC_BRANCH:-sync/upstream-${SYNC_DATE}}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -z "${GITHUB_OUTPUT:-}" ]]; then
  echo "GITHUB_OUTPUT is required for this script" >&2
  exit 1
fi

write_output() {
  local key="$1"
  local value="$2"
  echo "${key}=${value}" >> "${GITHUB_OUTPUT}"
}

ensure_remote_branch() {
  local remote_name="$1"
  local branch_name="$2"
  if ! git rev-parse --verify --quiet "refs/remotes/${remote_name}/${branch_name}" >/dev/null; then
    echo "Missing remote branch ${remote_name}/${branch_name}" >&2
    exit 1
  fi
}

git fetch "${UPSTREAM_REMOTE}" "${UPSTREAM_BRANCH}" --prune
git fetch origin "${BASE_BRANCH}" --prune

ensure_remote_branch "${UPSTREAM_REMOTE}" "${UPSTREAM_BRANCH}"
ensure_remote_branch origin "${BASE_BRANCH}"

git checkout -B "${SYNC_BRANCH}" "origin/${BASE_BRANCH}"

status="unknown"
labels="upstream-sync"
conflict_files=""
pr_title=""
pr_body_file=".github/sync-reports/pr-body-${SYNC_DATE}.md"
mkdir -p .github/sync-reports

if git merge --no-ff --no-commit "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"; then
  if git diff --cached --quiet; then
    git merge --abort || true
    status="up-to-date"
    write_output "status" "${status}"
    write_output "has_changes" "false"
    write_output "sync_branch" "${SYNC_BRANCH}"
    echo "Branch is already up-to-date."
    exit 0
  fi

  status="clean"
  pr_title="chore(sync): merge ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH} into ${BASE_BRANCH} (${SYNC_DATE})"
  git commit -m "${pr_title}"

  cat > "${pr_body_file}" <<EOF
## Upstream Sync (Clean)

- Source: \`${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}\`
- Target: \`${BASE_BRANCH}\`
- Sync date (UTC): \`${SYNC_DATE}\`
- Status: ✅ clean merge

### Notes
- This PR was generated automatically by \`upstream-sync.yml\`.
- Please run normal review checks before merge.
EOF
else
  status="conflict"
  labels="upstream-sync,needs-conflict-resolve"
  conflict_files="$(git diff --name-only --diff-filter=U | sed '/^$/d' | paste -sd ', ' -)"
  git merge --abort || true

  report_file=".github/sync-reports/upstream-sync-${SYNC_DATE}.md"
  cat > "${report_file}" <<EOF
# Upstream Sync Conflict Report (${SYNC_DATE})

Source: \`${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}\`  
Target: \`${BASE_BRANCH}\`

## Conflicts

${conflict_files}

## Resolution Checklist

1. Checkout this branch locally.
2. Merge \`${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}\` into \`${BASE_BRANCH}\`.
3. Resolve conflicts and rerun validation.
4. Push updates to this PR branch.
EOF

  git add "${report_file}"
  pr_title="chore(sync): upstream sync requires conflict resolution (${SYNC_DATE})"
  git commit -m "${pr_title}"

  cat > "${pr_body_file}" <<EOF
## Upstream Sync (Conflict)

- Source: \`${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}\`
- Target: \`${BASE_BRANCH}\`
- Sync date (UTC): \`${SYNC_DATE}\`
- Status: ⚠️ manual conflict resolution required

### Conflict files
\`${conflict_files}\`

### Next actions
1. Pull this branch locally.
2. Resolve the listed conflicts.
3. Push resolution commits back to this branch.
EOF
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY_RUN=true, skip push for ${SYNC_BRANCH}"
else
  git push --force-with-lease --set-upstream origin "${SYNC_BRANCH}"
fi

write_output "status" "${status}"
write_output "has_changes" "true"
write_output "sync_branch" "${SYNC_BRANCH}"
write_output "pr_title" "${pr_title}"
write_output "pr_labels" "${labels}"
write_output "pr_body_file" "${pr_body_file}"

echo "Sync status: ${status}"
