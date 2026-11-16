#!/bin/bash
# Re-arm the idle watchdog: delete logs/watchdog.disable if present, and start scripts/idle_watchdog.sh
# (detached, setsid nohup) unless one is already running (PID file + kill -0). Prints one status line.
# Called at the start of scripts/gpu_session.sh and scripts/grpo_pipeline.sh.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOGDIR=$REPO/logs
mkdir -p "$LOGDIR"
if [ -f "$LOGDIR/watchdog.disable" ]; then rm -f "$LOGDIR/watchdog.disable"; echo "removed $LOGDIR/watchdog.disable"; fi
PID=$(cat "$LOGDIR/watchdog.pid" 2>/dev/null || true)
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && grep -q idle_watchdog "/proc/$PID/cmdline" 2>/dev/null; then
  echo "watchdog already running (PID $PID)"; exit 0
fi
rm -f "$LOGDIR/watchdog.pid"   # stale (not a running watchdog); idle_watchdog.sh would otherwise trust it
setsid nohup "$REPO/scripts/idle_watchdog.sh" > /dev/null 2>&1 < /dev/null &
for _ in 1 2 3 4 5 6 7 8 9 10; do
  sleep 0.5
  PID=$(cat "$LOGDIR/watchdog.pid" 2>/dev/null || true)
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then echo "watchdog started (PID $PID)"; exit 0; fi
done
echo "ERROR: watchdog failed to start (see $LOGDIR/watchdog.log)"; exit 1
