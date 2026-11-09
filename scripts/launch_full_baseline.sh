#!/bin/bash
# Launch the full 1M phase 2 baseline + its watcher, both detached (setsid nohup) so they do not depend
# on the Claude Code session. The run's shell writes its own PID to a file (setsid may fork, so $! is
# not reliable); the watcher is only started once that PID is known and alive.
set -eu
cd /workspace
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=/workspace/results/phase2/${TS}_full1M
RUN_LOG=/workspace/logs/phase2_full1M_${TS}.log
TASK_OUT=/workspace/logs/phase2_full1M_${TS}.exit
PID_FILE=/workspace/logs/phase2_full1M_${TS}.pid
WATCH_LOG=/workspace/logs/watcher_full1M_${TS}.log
ARGS="--n_contexts 1000000 --max_chunks_per_doc 4 --checkpoint_every 50000 --out_dir $RUN_DIR"
setsid nohup bash -c "echo \$\$ > $PID_FILE; cd /workspace; python scripts/phase2_baseline.py $ARGS > $RUN_LOG 2>&1; echo \"exit \$?\" > $TASK_OUT" \
  > /dev/null 2>&1 < /dev/null &
for i in $(seq 50); do [ -s "$PID_FILE" ] && break; sleep 0.2; done
RUN_PID=$(cat "$PID_FILE")
kill -0 "$RUN_PID" || { echo "ERROR: run process $RUN_PID not alive"; exit 1; }
setsid nohup bash -c "echo \$\$ > $PID_FILE.watcher; exec env RUN_PID=$RUN_PID RUN_DIR=$RUN_DIR RUN_LOG=$RUN_LOG TASK_OUT=$TASK_OUT bash scripts/watch_and_stop.sh" \
  > $WATCH_LOG 2>&1 < /dev/null &
for i in $(seq 50); do [ -s "$PID_FILE.watcher" ] && break; sleep 0.2; done
echo "RUN_DIR=$RUN_DIR"; echo "RUN_LOG=$RUN_LOG"; echo "TASK_OUT=$TASK_OUT"; echo "WATCH_LOG=$WATCH_LOG"
echo "RUN_PID=$RUN_PID"; echo "WATCH_PID=$(cat $PID_FILE.watcher)"; echo "ARGS=$ARGS"
