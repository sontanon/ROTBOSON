#!/bin/bash
cd /workspace/rotboson
for l in 3 4; do
  echo "=== $(date -u +%FT%TZ) fold-c l=$l ==="
  uv run tools/sweep_driver.py out/sweep_l${l}_fold320c.toml 2>&1 | tee out/campaigns/fold-l${l}-320c.driver.log
done
echo "=== FOLD-C DONE $(date -u +%FT%TZ) ==="
