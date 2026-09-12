#!/usr/bin/env bash
# Runbook execution (SAN-31): the four down-campaigns, sequentially, each
# followed by --summarize (the w_min localization). Logs land next to the
# campaign roots; the whole sequence is safe to detach and inspect later.
set -u
cd /workspace/rotboson
mkdir -p out/campaigns

for l in 1 2 3 4; do
  spec="configs/runbook_l${l}_down.toml"
  log="out/campaigns/runbook-l${l}.driver.log"
  echo "=== [$(date '+%H:%M:%S')] campaign l=${l}: ${spec} ==="
  uv run tools/sweep_driver.py "${spec}" 2>&1 | tee "${log}"
  echo "=== [$(date '+%H:%M:%S')] l=${l} summarize ==="
  uv run tools/sweep_driver.py "${spec}" --summarize 2>&1 | tee -a "${log}"
done
echo "=== [$(date '+%H:%M:%S')] runbook campaigns complete ==="
