#!/bin/bash
cd /workspace/rotboson
for job in 5-up 5-down 6-up 6-down; do
  l=${job%-*}; d=${job#*-}
  echo "=== $(date -u +%FT%TZ) l=$l $d ==="
  uv run tools/sweep_driver.py "out/sweep_l${l}_${d}.toml" 2>&1 | tee "out/campaigns/sweep-l${l}-${d}.driver.log"
done
echo "=== L56 DONE $(date -u +%FT%TZ) ==="
