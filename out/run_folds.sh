#!/bin/bash
cd /workspace/rotboson
for l in 1 2 3 4; do
  echo "=== $(date -u +%FT%TZ) fold campaign l=$l ==="
  uv run tools/sweep_driver.py out/sweep_l${l}_fold320.toml 2>&1 | tee out/campaigns/fold-l${l}-320.driver.log
done
echo "=== ALL FOLD CAMPAIGNS DONE $(date -u +%FT%TZ) ==="
