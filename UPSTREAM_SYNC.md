# Upstream Sync Guide

This repository tracks two remotes:

- `origin`: your fork (`yunfei07/nanobot`)
- `upstream`: official source (`HKUDS/nanobot`)

## Branch strategy

- `main`: product branch for custom features
- `upstream-main`: mirror branch that tracks `upstream/main`
- `sync/upstream-YYYYMMDD`: auto-generated sync branch

## Automation

- Workflow: `.github/workflows/upstream-sync.yml`
- Schedule: every Monday at `02:00 UTC`
- Script: `scripts/upstream_sync.sh`

Flow:

1. Fetch `upstream/main` and `origin/main`
2. Create/update `sync/upstream-YYYYMMDD`
3. Try to merge `upstream/main` into `origin/main`
4. If clean, open/update a PR labeled `upstream-sync`
5. If conflict, create a conflict report and open/update a PR labeled:
   - `upstream-sync`
   - `needs-conflict-resolve`

## Sync PR validation

- Workflow: `.github/workflows/sync-ci.yml`
- Runs on PRs to `main` from branches starting with `sync/upstream-`
- Checks:
  - `pytest -q tests/test_tool_validation.py`
  - `python -m compileall -q nanobot`

## Manual trigger

You can run sync manually from GitHub Actions:

1. Open the **Actions** tab
2. Select **upstream-sync**
3. Click **Run workflow**

