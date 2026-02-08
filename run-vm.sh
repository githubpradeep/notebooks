#!/usr/bin/env bash
#sudo apt-get update
#sudo apt-get install -y xvfb x11-xserver-utils x11vnc xfce4 dbus-x11 xauth x11-utils git

set -e

log() {
  printf '%s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "Missing required command: $1"
    return 1
  fi
  return 0
}

BASE_DIR="${VNC_HOME:-$HOME/.vnc-sandbox}"
WORKSPACE_DIR="${VNC_WORKSPACE:-$BASE_DIR/workspace}"
NOVNC_DIR="${NOVNC_DIR:-$BASE_DIR/novnc}"

DISPLAY_NUM="${DISPLAY_NUM:-:1}"
VNC_RESOLUTION="${VNC_RESOLUTION:-1512x982}"
VNC_RANDR_MODES="${VNC_RANDR_MODES:-1512x982,2560x1600,2560x1440,1920x1080,1680x1050,1440x900,1280x720}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
mkdir -p "$WORKSPACE_DIR"

missing=0
require_cmd Xvfb || missing=1
require_cmd x11vnc || missing=1
require_cmd xrandr || missing=1
require_cmd xfce4-session || missing=1
require_cmd dbus-launch || missing=1

if [ "$missing" -ne 0 ]; then
  log ""
  log "Install required packages (Debian/Ubuntu):"
  log "  sudo apt-get update"
  log "  sudo apt-get install -y xvfb x11-xserver-utils x11vnc xfce4 dbus-x11 xauth x11-utils"
  exit 1
fi

if [ ! -d "$NOVNC_DIR" ]; then
  if command -v git >/dev/null 2>&1; then
    log "Cloning noVNC into $NOVNC_DIR..."
    git clone --depth 1 --branch e2b-desktop https://github.com/e2b-dev/noVNC.git "$NOVNC_DIR"
  else
    log "noVNC not found at $NOVNC_DIR and git is not installed."
    log "Install git or set NOVNC_DIR to an existing noVNC checkout."
    exit 1
  fi
fi

if [ ! -d "$NOVNC_DIR/utils/websockify" ]; then
  if command -v git >/dev/null 2>&1; then
    log "Cloning websockify..."
    git clone --depth 1 --branch v0.12.0 https://github.com/novnc/websockify "$NOVNC_DIR/utils/websockify"
  else
    log "websockify not found and git is not installed."
    exit 1
  fi
fi

ln -sf "$NOVNC_DIR/vnc.html" "$NOVNC_DIR/index.html"

cleanup() {
  kill "$XVFB_PID" "$XFCE_PID" "$VNC_PID" "$NOVNC_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

log "=== Starting VNC Sandbox (VM) ==="
Xvfb "$DISPLAY_NUM" -screen 0 "${VNC_RESOLUTION}x24" &
XVFB_PID=$!
sleep 2

export DISPLAY="$DISPLAY_NUM"
if command -v xrandr >/dev/null 2>&1; then
  OUTPUT=$(xrandr | awk 'NR>1 && $2=="connected"{print $1; exit}')
  if [ -z "$OUTPUT" ]; then
    OUTPUT=$(xrandr | awk 'NR>1 && $1!="Screen"{print $1; exit}')
  fi
  if [ -n "$VNC_RANDR_MODES" ] && command -v cvt >/dev/null 2>&1; then
    IFS=',' read -ra MODES <<< "$VNC_RANDR_MODES"
    for mode in "${MODES[@]}"; do
      if ! xrandr | grep -q " ${mode} "; then
        MODELINE=$(cvt ${mode%x*} ${mode#*x} | sed -n 's/^Modeline //p')
        NAME=$(echo "$MODELINE" | awk '{print $1}')
        xrandr --newmode $MODELINE >/dev/null 2>&1 || true
        if [ -n "$OUTPUT" ]; then
          xrandr --addmode "$OUTPUT" "$NAME" >/dev/null 2>&1 || true
        fi
      fi
    done
  fi
  if [ -n "$OUTPUT" ]; then
    xrandr --output "$OUTPUT" --mode "${VNC_RESOLUTION}" >/dev/null 2>&1 || true
  else
    xrandr --screen 0 -s "${VNC_RESOLUTION}" >/dev/null 2>&1 || true
  fi
fi

dbus-daemon --system --fork >/dev/null 2>&1 || true
eval $(dbus-launch --sh-syntax --exit-with-session) >/dev/null 2>&1 || true

xfce4-session &
XFCE_PID=$!
sleep 2

x11vnc -display "$DISPLAY_NUM" -geometry "${VNC_RESOLUTION}" -xrandr -nopw -listen 0.0.0.0 -xkb -forever -rfbport "$VNC_PORT" &
VNC_PID=$!
sleep 2

cd "$NOVNC_DIR"
./utils/novnc_proxy --vnc "localhost:${VNC_PORT}" --listen "$NOVNC_PORT" --web "$NOVNC_DIR" &
NOVNC_PID=$!

log ""
log "==========================================="
log "  VNC Sandbox (VM) is ready!"
log "  - noVNC Web UI: http://localhost:${NOVNC_PORT}"
log "  - VNC Direct:   localhost:${VNC_PORT}"
log "==========================================="

wait -n "$XVFB_PID" "$XFCE_PID" "$VNC_PID" "$NOVNC_PID"
