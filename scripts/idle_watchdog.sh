#!/bin/bash
# Idle auto-stop watchdog for this RunPod pod.
# Every CHECK_EVERY seconds (default 300) it reads GPU utilization with nvidia-smi.
# If utilization has been below THRESH% (default 5) at every check for IDLE_LIMIT seconds
# (default 3600 = 60 min, measured from the first low check of the streak), it runs
#     runpodctl stop pod $RUNPOD_POD_ID
# It only ever uses "stop" (never "remove"/"terminate"); a stopped pod keeps /workspace.
# Every check is logged to /workspace/logs/watchdog.log.
#
# Start:   nohup /workspace/scripts/idle_watchdog.sh >/dev/null 2>&1 &
# Disable: touch /workspace/logs/watchdog.disable    (it exits at the next check)
#     or:  kill "$(cat /workspace/logs/watchdog.pid)"
# Re-enable: rm -f /workspace/logs/watchdog.disable, then start again.
set -u
POD_ID=${RUNPOD_POD_ID:?RUNPOD_POD_ID not set}
CHECK_EVERY=${CHECK_EVERY:-300}
IDLE_LIMIT=${IDLE_LIMIT:-3600}
THRESH=${THRESH:-5}
LOGDIR=${WATCHDOG_LOGDIR:-/workspace/logs}   # override only for testing
LOG=$LOGDIR/watchdog.log
PIDFILE=$LOGDIR/watchdog.pid
DISABLE=$LOGDIR/watchdog.disable
mkdir -p "$LOGDIR"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  log "another watchdog is already running (PID $(cat "$PIDFILE")); exiting"; exit 0
fi
echo $$ > "$PIDFILE"
trap 'log "watchdog PID $$ exiting"; rm -f "$PIDFILE"' EXIT
log "watchdog started: PID $$, pod $POD_ID, check every ${CHECK_EVERY}s, stop after ${IDLE_LIMIT}s below ${THRESH}%"

idle_since=""
while true; do
  if [ -f "$DISABLE" ]; then log "disable file $DISABLE present; exiting"; exit 0; fi
  now=$(date +%s)
  # max over GPUs (one GPU here); any error -> treat as busy, never stop on a failed reading
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1 | tr -d ' ')
  if ! [[ "$util" =~ ^[0-9]+$ ]]; then
    log "check: nvidia-smi reading failed ('$util'); treating as busy, idle streak reset"
    idle_since=""
  elif [ "$util" -ge "$THRESH" ]; then
    log "check: util=${util}% (busy); idle streak reset"
    idle_since=""
  else
    [ -z "$idle_since" ] && idle_since=$now
    idle=$((now - idle_since))
    log "check: util=${util}% (idle); idle for $((idle / 60)) min (limit $((IDLE_LIMIT / 60)) min)"
    if [ "$idle" -ge "$IDLE_LIMIT" ]; then
      log "GPU idle for $((idle / 60)) min -> runpodctl stop pod $POD_ID"
      for i in 1 2 3 4 5; do
        if out=$(runpodctl stop pod "$POD_ID" 2>&1); then log "stop command sent: $out"; exit 0; fi
        log "runpodctl stop failed (attempt $i): $out; retrying in 60s"; sleep 60
      done
      log "ERROR: could not stop pod after 5 attempts; idle streak reset, continuing to watch"
      idle_since=""
    fi
  fi
  sleep "$CHECK_EVERY"
done
