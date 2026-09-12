#!/bin/bash
# Overnight serial sweep driver runs for l=2..6 (2026-09-10).
cd /workspace/rotboson
for l in 2 3 4 5 6; do
  echo "=== $(date -u +%FT%TZ) starting l=$l ==="
  uv run tools/sweep_driver.py out/sweep_l${l}_v2.toml 2>&1 | tee out/campaigns/sweep-l${l}-v2.driver.log
  echo "=== $(date -u +%FT%TZ) finished l=$l (exit $?) ==="
done
echo "=== ALL CAMPAIGNS DONE $(date -u +%FT%TZ) ==="
