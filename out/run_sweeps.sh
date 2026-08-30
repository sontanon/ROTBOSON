#!/bin/bash
# Sequential l=1 validation sweeps: up through the turning point, then down
# to the Newtonian end. Each campaign resumes automatically if restarted.
cd /workspace/rotboson
echo "=== UP sweep start $(date -u +%H:%M:%S) ==="
uv run tools/sweep_driver.py out/sweep_l1_up.toml
echo "=== UP sweep exit $? at $(date -u +%H:%M:%S) ==="
echo "=== DOWN sweep start $(date -u +%H:%M:%S) ==="
uv run tools/sweep_driver.py out/sweep_l1_down.toml
echo "=== DOWN sweep exit $? at $(date -u +%H:%M:%S) ==="
