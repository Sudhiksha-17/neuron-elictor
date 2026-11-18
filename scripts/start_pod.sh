#!/bin/bash
# Bootstrap any fresh pod (new or restarted). Run as root from anywhere:
#   bash /workspace/neuron-elictor/scripts/start_pod.sh
# (on a brand-new volume without the repo: curl -fsSL https://raw.githubusercontent.com/Sudhiksha-17/neuron-elictor/main/scripts/start_pod.sh | bash)
# Steps:
#   1. apt-get install tmux / git if missing
#   2. install Claude Code if missing (~/.local/bin/claude); add ~/.local/bin to PATH in ~/.bashrc
#   3. clone the repo into /workspace/neuron-elictor if missing, otherwise git pull (fast-forward only)
#   4. persist Claude Code login + conversation history: ~/.claude and ~/.claude.json live under
#      /workspace/.claude-home/ (on the persistent volume) and are symlinked into $HOME
#   5. start (or attach to) tmux session "pilot" in the repo with claude running
# Idempotent: safe to rerun. Never deletes anything: a pre-existing non-symlink ~/.claude or ~/.claude.json that would
# shadow the persisted copy is moved aside to <name>.bak.<timestamp>.
set -euo pipefail
REPO_URL=https://github.com/Sudhiksha-17/neuron-elictor
REPO=/workspace/neuron-elictor
PERSIST=/workspace/.claude-home
SESSION=pilot
TS=$(date +%Y%m%d_%H%M%S)
say() { echo "[start_pod] $*"; }

# 1. tmux, git
missing=()
for p in tmux git; do command -v "$p" > /dev/null 2>&1 || missing+=("$p"); done
command -v curl > /dev/null 2>&1 || missing+=(curl)
if [ ${#missing[@]} -gt 0 ]; then
  say "apt-get install ${missing[*]}"
  SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
  $SUDO apt-get update -qq && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${missing[@]}"
else say "tmux, git, curl present"; fi

# 2. Claude Code + PATH
export PATH="$HOME/.local/bin:$PATH"
grep -qs 'HOME/.local/bin' "$HOME/.bashrc" || { echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"; say "added ~/.local/bin to PATH in ~/.bashrc"; }
if command -v claude > /dev/null 2>&1; then say "claude present: $(command -v claude)"
else say "installing Claude Code"; curl -fsSL https://claude.ai/install.sh | bash; fi
command -v claude > /dev/null 2>&1 || { say "ERROR: claude not found after install"; exit 1; }

# 3. repo
if [ -d "$REPO/.git" ]; then
  say "git pull in $REPO"
  git -C "$REPO" pull --ff-only || say "WARNING: git pull failed (local changes or offline); continuing with the local copy"
else
  say "cloning $REPO_URL into $REPO"
  mkdir -p "$(dirname "$REPO")"; git clone "$REPO_URL" "$REPO"
fi

# 4. persist ~/.claude and ~/.claude.json under $PERSIST
mkdir -p "$PERSIST"
persist() {   # persist <name>: make $HOME/<name> a symlink to $PERSIST/<name>
  local name=$1 home="$HOME/$1" store="$PERSIST/$1"
  if [ -L "$home" ] && [ "$(readlink "$home")" = "$store" ]; then say "$home -> $store (already linked)"; return; fi
  if [ -e "$home" ] || [ -L "$home" ]; then
    if [ ! -e "$store" ] && [ ! -L "$home" ]; then
      mv "$home" "$store"; say "moved existing $home into $store"           # first run: adopt the current login/history
    else
      mv "$home" "$home.bak.$TS"; say "moved $home aside to $home.bak.$TS (persisted copy wins)"
    fi
  fi
  if [ ! -e "$store" ]; then
    if [ "$name" = .claude ]; then mkdir -p "$store"; else echo '{}' > "$store"; fi
    say "created empty $store"
  fi
  ln -s "$store" "$home"; say "linked $home -> $store"
}
persist .claude
persist .claude.json

# 5. tmux session with claude
if tmux has-session -t "$SESSION" 2> /dev/null; then say "tmux session '$SESSION' exists"
else
  say "starting tmux session '$SESSION' in $REPO with claude"
  tmux new-session -d -s "$SESSION" -c "$REPO" "bash -lc 'export PATH=\"\$HOME/.local/bin:\$PATH\"; claude; exec bash'"
fi
if [ -n "${TMUX:-}" ]; then exec tmux switch-client -t "$SESSION"
elif [ -t 1 ]; then exec tmux attach -t "$SESSION"
else say "not a terminal; attach with: tmux attach -t $SESSION"; fi
