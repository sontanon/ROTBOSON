#!/bin/bash
cd /workspace/rotboson
for l in 2 3 4; do
  echo "=== $(date -u +%FT%TZ) fold-b l=$l ==="
  uv run tools/sweep_driver.py out/sweep_l${l}_fold320b.toml 2>&1 | tee out/campaigns/fold-l${l}-320b.driver.log
done
echo "=== FOLD-B DONE $(date -u +%FT%TZ) ==="
