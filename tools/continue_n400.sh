#!/bin/bash
# SAN-33 continuation watcher: after l=3 and l=4 reach terminal, resume any
# of l=2,3,4 that stopped at `done:psi0_target` (fold not reached — the M2
# psi0_target safety nets are too low for the N=200 branch) with a raised
# target, capturing the fold. Runs in the background while the main runner
# handles l=3/l=4; low-overhead (sleeps until l=3 AND l=4 are terminal).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/out/campaigns"
LOG="$ROOT/out/n400_continue.log"
CONFIG_DIR="$ROOT/configs"

echo "=== n400 continuation watcher armed $(date '+%F %T') ===" >> "$LOG"

terminal() {
    local state="$1"
    python3 - "$state" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if d.get('status') in ('done', 'stopped', 'failed') else 1)
EOF
}

wait_both() {  # wait until l=3 AND l=4 are terminal
    while true; do
        t3=1; t4=1
        terminal "$OUT/runbook-n400-l3/state.json" || t3=0
        terminal "$OUT/runbook-n400-l4/state.json" || t4=0
        [ "$t3" = 1 ] && [ "$t4" = 1 ] && return 0
        sleep 180
    done
}

# raised targets per l (original -> continuation)
raise_target() {  # l -> new psi0_target
    case "$1" in
        2) echo 2.5 ;;
        3) echo 6.0 ;;
        4) echo 4.0 ;;
    esac
}

# wait for l=3 and l=4 to finish (the runner handles them)
echo "waiting for l=3 and l=4 to reach terminal..." >> "$LOG"
wait_both

for L in 2 3 4; do
    STATE="$OUT/runbook-n400-l${L}/state.json"
    CFG="$CONFIG_DIR/runbook_l${L}_n400.toml"
    status=$(python3 -c "import json;d=json.load(open('$STATE'));print(d.get('stop_reason'))" 2>/dev/null)
    echo "l=${L} stop_reason=${status}" >> "$LOG"
    if [ "$status" = "done:psi0_target" ]; then
        NEWT=$(raise_target "$L")
        echo "l=${L} hit target without fold; resuming with psi0_target=$NEWT" >> "$LOG"
        # bump psi0_target in the config
        python3 - "$CFG" "$NEWT" <<'EOF'
import sys, re
p, newt = sys.argv[1], sys.argv[2]
s = open(p).read()
s = re.sub(r'psi0_target\s*=\s*[\d.]+', f'psi0_target = {newt}', s, count=1)
open(p, 'w').write(s)
EOF
        # reconcile the state's spec_hash to the edited config
        uv run python - "$CFG" "$STATE" <<'EOF'
import sys
sys.path.insert(0, 'tools')
from sweep_driver import load_spec
from pathlib import Path
import json
spec = load_spec(Path(sys.argv[1]))
st = json.load(open(sys.argv[2]))
st['spec_hash'] = spec.spec_hash
st['status'] = 'running'
st['stop_reason'] = None
st['turning_point'] = None
json.dump(st, open(sys.argv[2], 'w'), indent=2)
print('reconciled spec_hash for', sys.argv[2])
EOF
        echo "== l=${L} continuation start $(date '+%F %T') ==" >> "$LOG"
        uv run tools/sweep_driver.py "$CFG" >> "$LOG" 2>&1
        uv run tools/sweep_driver.py "$CFG" --summarize >> "$LOG" 2>&1
        tp=$(python3 -c "import json;d=json.load(open('$STATE'));t=d.get('turning_point') or {};print(t.get('omega') or t.get('omega_sample_min') or '')" 2>/dev/null)
        stop=$(python3 -c "import json;d=json.load(open('$STATE'));print(d.get('stop_reason'))" 2>/dev/null)
        echo "== l=${L} continuation end: stop=$stop w_min=$tp $(date '+%F %T') ==" >> "$LOG"
    fi
done
echo "=== CONTINUATION DONE $(date '+%F %T') ===" >> "$LOG"
