#!/bin/bash
cd /workspace/rotboson
for l in 3 4; do
  echo "=== $(date -u +%FT%TZ) fold-E l=$l ==="
  uv run tools/sweep_driver.py out/sweep_l${l}_foldE.toml 2>&1 | tee out/campaigns/fold-l${l}-E.driver.log
done
echo "=== FOLD-E DONE $(date -u +%FT%TZ) ==="
