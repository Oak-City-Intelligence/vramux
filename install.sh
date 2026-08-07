#!/usr/bin/env bash
#
# install.sh — put the `vramux` client on $PATH, and optionally install the
# router as a user service.
#
# There is nothing to build and nothing to copy into site-packages. The client
# half of vramux is stdlib-only on purpose, so all this installs is a shim that
# points an interpreter at this checkout: `python -m vramux` otherwise only
# resolves from inside it, and every shell client depends on the shim existing.
#
#   ./install.sh                        # the client shim, into ~/.local/bin
#   ./install.sh --service              # the shim, plus the systemd user unit
#   ./install.sh --uninstall            # remove the shim
#   ./install.sh --uninstall --service  # remove the unit too, stopping the router
#
# Removing the unit is opt-in because it stops a router other things on the
# machine are talking to, and a plain `--uninstall` should not.
#
# Overrides:
#   VRAMUX_BIN_DIR   where the shim goes          (default ~/.local/bin)
#   VRAMUX_UNIT_DIR  where the unit goes          (default ~/.config/systemd/user)
#   VRAMUX_PYTHON    interpreter the shim runs    (default .venv/bin/python,
#                                                  else python3 from $PATH)

set -euo pipefail

CHECKOUT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${VRAMUX_BIN_DIR:-$HOME/.local/bin}"
SHIM="$BIN_DIR/vramux"
UNIT_DIR="${VRAMUX_UNIT_DIR:-$HOME/.config/systemd/user}"
UNIT="$UNIT_DIR/vramux.service"

WITH_SERVICE=0
UNINSTALL=0

for arg in "$@"; do
  case "$arg" in
    --service)   WITH_SERVICE=1 ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help)   sed -n '3,25p' "${BASH_SOURCE[0]}" | sed 's|^# \{0,1\}||'; exit 0 ;;
    *)           echo "install.sh: unknown argument '$arg'" >&2; exit 2 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- uninstall ---------------------------------------------------------------

if [ "$UNINSTALL" -eq 1 ]; then
  if [ -e "$SHIM" ]; then
    rm -f "$SHIM"
    say "removed $SHIM"
  else
    say "no shim at $SHIM"
  fi

  if [ "$WITH_SERVICE" -eq 0 ]; then
    if [ -e "$UNIT" ]; then
      say "left $UNIT alone — pass --service as well to remove it"
    fi
  elif [ -e "$UNIT" ]; then
    # Stopping first: a running unit whose file has gone is a confusing state
    # to leave behind, and `disable` alone does not stop it. This stops the
    # router, which is why it takes an explicit --service.
    say "stopping and removing the router service"
    systemctl --user disable --now vramux.service >/dev/null 2>&1 || true
    rm -f "$UNIT"
    systemctl --user daemon-reload || true
    say "removed $UNIT"
  else
    say "no unit at $UNIT"
  fi

  say "the checkout, ~/.cache/vramux and models.yml were left alone"
  exit 0
fi

# --- sanity ------------------------------------------------------------------

[ -f "$CHECKOUT/vramux/__main__.py" ] \
  || die "$CHECKOUT does not look like a vramux checkout (no vramux/__main__.py)"

if [ -n "${VRAMUX_PYTHON:-}" ]; then
  PYTHON="$VRAMUX_PYTHON"
elif [ -x "$CHECKOUT/.venv/bin/python" ]; then
  PYTHON="$CHECKOUT/.venv/bin/python"
else
  PYTHON="$(command -v python3 || true)"
fi

[ -n "$PYTHON" ] && [ -x "$PYTHON" ] || die "no usable python found; set VRAMUX_PYTHON"

"$PYTHON" - <<'EOF' || die "vramux needs python 3.10 or newer"
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

# The client needs none of this; the router needs both. Worth saying now rather
# than at the first failed start.
"$PYTHON" -c 'import aiohttp, yaml' 2>/dev/null \
  || warn "$PYTHON cannot import aiohttp and PyYAML — the client will work, the router will not"

# --- the shim ----------------------------------------------------------------

mkdir -p "$BIN_DIR"

if [ -e "$SHIM" ] && ! grep -q 'VRAMUX_CHECKOUT' "$SHIM" 2>/dev/null; then
  die "$SHIM exists and was not written by this script — move it aside first"
fi

cat > "$SHIM" <<EOF
#!/usr/bin/env bash
# vramux — client CLI for the VRAM broker. Written by install.sh; edits here
# are lost on the next run.
#
# PYTHONPATH rather than an install: the client half of vramux is stdlib-only
# on purpose, so pointing an interpreter at the checkout is enough and there is
# no installed copy to drift from it.
VRAMUX_CHECKOUT="\${VRAMUX_CHECKOUT:-$CHECKOUT}"
VRAMUX_PYTHON="\${VRAMUX_PYTHON:-$PYTHON}"
exec env PYTHONPATH="\$VRAMUX_CHECKOUT\${PYTHONPATH:+:\$PYTHONPATH}" \\
  "\$VRAMUX_PYTHON" -m vramux "\$@"
EOF
chmod +x "$SHIM"
say "installed $SHIM -> $CHECKOUT (python: $PYTHON)"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) warn "$BIN_DIR is not on \$PATH — add it, or the shim is invisible to clients" ;;
esac

# --- the service -------------------------------------------------------------

if [ "$WITH_SERVICE" -eq 1 ]; then
  command -v systemctl >/dev/null 2>&1 || die "--service needs systemd"

  if [ ! -f "$CHECKOUT/models.yml" ]; then
    warn "no models.yml yet — copy models.example.yml and edit it before starting"
  fi

  mkdir -p "$UNIT_DIR"
  # The unit ships with %CHECKOUT% and /usr/bin/python3 as placeholders; both
  # become this installation's real paths, so the venv's interpreter is used
  # when there is one.
  sed -e "s|%CHECKOUT%|$CHECKOUT|" \
      -e "s|^ExecStart=.*|ExecStart=$PYTHON -m vramux|" \
      "$CHECKOUT/systemd/vramux.service" > "$UNIT"
  systemctl --user daemon-reload
  say "installed $UNIT"
  say ""
  say "  systemctl --user enable --now vramux"
  say ""
  say "It binds 127.0.0.1 and has no authentication. Read SECURITY.md before"
  say "changing either."
fi

say ""
say "check it: vramux state"
