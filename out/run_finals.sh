#!/bin/bash
cd /workspace/rotboson
for l in 1 2 3 4; do
  echo "=== $(date -u +%FT%TZ) starting l=$l ==="
  uv run tools/sweep_driver.py --fresh out/sweep_l${l}_final.toml 2>&1 | tee out/campaigns/sweep-l${l}-final.driver.log
done
echo "=== ALL FINAL CAMPAIGNS DONE $(date -u +%FT%TZ) ==="
