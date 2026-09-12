#!/bin/bash
cd /workspace/rotboson
for l in 3 4; do
  echo "=== $(date -u +%FT%TZ) fold-d l=$l ==="
  uv run tools/sweep_driver.py out/sweep_l${l}_fold320d.toml 2>&1 | tee out/campaigns/fold-l${l}-320d.driver.log
done
echo "=== FOLD-D DONE $(date -u +%FT%TZ) ==="
