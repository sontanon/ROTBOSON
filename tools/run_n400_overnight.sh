#!/bin/bash
# Overnight N400 campaign runner (M3 / SAN-33).
# Runs l=2,3,4 N400 campaigns SEQUENTIALLY, one after the other, after l=1
# reaches its terminal state. Each campaign logs to the shared summary and
# writes a .STATUS marker; on each completion a best-effort Telegram ping is
# sent via `pi --print` (the bridge has no background-send CLI).
#
# Run in tmux (e.g. session "n400"):
#   tmux new-session -s n400 -d 'bash tools/run_n400_overnight.sh'
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/out/n400_overnight.log"
CONFIG_DIR="$ROOT/configs"
OUT_DIR="$ROOT/out/campaigns"

echo "=== n400 overnight runner armed $(date '+%F %T') ===" >> "$LOG"

# terminal status check: 0 if state.json reached a terminal status
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

status_line() {
    local state="$1"
    python3 - "$state" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
tp = d.get('turning_point') or {}
w = tp.get('omega') or (tp.get('omega_sample_min') if tp else None)
print(f"status={d.get('status')} stop={d.get('stop_reason')} steps={len(d.get('steps', []))}"
      + (f" w_min={w:.6E}" if w is not None else ""))
EOF
}

wait_terminal() {  # wait until a campaign state.json reaches terminal status
    local camdir="$1"
    while ! terminal "$camdir/state.json"; do
        sleep 120
    done
}

notify() {  # best-effort Telegram ping via a pi sub-turn
    local msg="$1"
    if command -v pi >/dev/null 2>&1; then
        pi --print "Use the telegram_message tool to send this exact text to the paired Telegram chat: $msg" >/dev/null 2>&1 &
    fi
}

run_campaign() {
    local L="$1"
    local CFG="$CONFIG_DIR/runbook_l${L}_n400.toml"
    local CAMDIR="$OUT_DIR/runbook-n400-l${L}"
    echo "== l=${L} start $(date '+%F %T') ==" >> "$LOG"
    uv run tools/sweep_driver.py "$CFG" >> "$LOG" 2>&1
    local ec=$?
    uv run tools/sweep_driver.py "$CFG" --summarize >> "$LOG" 2>&1
    local line
    line="$(status_line "$CAMDIR/state.json")"
    echo "== l=${L} exit=$ec $line $(date '+%F %T') ==" >> "$LOG"
    echo "l=${L} $line" > "$CAMDIR/.STATUS"
    notify "[ROTBOSON N400] l=${L} campaign finished: $line"
}

# ---- wait for l=1 (currently running in the background) ----------------
echo "waiting for l=1 to reach terminal..." >> "$LOG"
wait_terminal "$OUT_DIR/runbook-n400-l1"
l1="$(status_line "$OUT_DIR/runbook-n400-l1/state.json")"
echo "== l=1 terminal: $l1 $(date '+%F %T') ==" >> "$LOG"
notify "[ROTBOSON N400] l=1 finished: $l1"

# ---- run l=2,3,4 sequentially ------------------------------------------
for L in 2 3 4; do
    run_campaign "$L"
done

echo "=== ALL N400 CAMPAIGNS DONE $(date '+%F %T') ===" >> "$LOG"
notify "[ROTBOSON N400] All l=2,3,4 campaigns complete. See $ROOT/out/n400_overnight.log"
