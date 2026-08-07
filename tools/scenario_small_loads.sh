#!/usr/bin/env bash
#
# scenario_small_loads.sh — manufacture the traffic shape multi-residency is for.
#
# The workstation this was built on runs one large consumer at a time: a batch
# run measured here peaked at 22 937 MiB of a 24 564 MiB card, from a single
# pipeline. Nothing about that traffic needs a broker to pack it, so nothing
# about it tests one. This makes the other shape on purpose — several modest
# consumers that fit together — and watches whether the budget stays honest
# while they come and go.
#
#   ./tools/scenario_small_loads.sh              # four holders, 90 seconds
#   ./tools/scenario_small_loads.sh 6 120        # six holders, 120 seconds
#
# It allocates real VRAM. Do not run it while something you care about is
# resident.

set -euo pipefail

HOLDERS="${1:-4}"
SECONDS_TO_HOLD="${2:-90}"
URL="${VRAMUX_URL:-http://127.0.0.1:11434}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Roughly an SD1.5-class consumer each: small enough that several coexist,
# large enough that the card notices. The point is the count, not the size.
MB_EACH="${MB_EACH:-2200}"

say() { printf '\n== %s\n' "$*"; }

state() {
  curl -s "$URL/gpu/state" | python3 -c "
import json, sys
d = json.load(sys.stdin)
b = d['budget']
print('  used %5d  free %5d  granted %5d  outstanding %5d  foreign %5d'
      % (b['used_mb'], b['free_mb'], b['granted_mb'], b['outstanding_mb'], b['foreign_mb']))
for l in d['leases']:
    print('    lease %-10s %5d MB  pids %s' % (l['owner'], l['granted_mb'], l['pids']))
r = d.get('residents') or []
if r:
    print('    residents: %s' % ', '.join(str(x) for x in r))
"
}

curl -sf "$URL/gpu/state" >/dev/null || { echo "no router at $URL" >&2; exit 1; }

say "before"
state

pids=()
cleanup() {
  for p in "${pids[@]:-}"; do
    [ -n "$p" ] && kill "$p" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

say "starting $HOLDERS holders of ${MB_EACH} MB each"
for i in $(seq 1 "$HOLDERS"); do
  python3 "$HERE/hold_vram.py" "$MB_EACH" \
    --seconds "$SECONDS_TO_HOLD" --lease "small-$i" --ttl 30 &
  pids+=($!)
  # Stagger them. Simultaneous acquires would test the lock, which is a
  # different question from whether the budget packs correctly; arrivals that
  # interleave with allocations are the realistic case.
  sleep 3
  printf '  holder %d up\n' "$i"
done

say "all holders up"
state

say "probing what is left, twice: once for what fits, once for what cannot"
# The first should be granted — the holders leave real room, and the broker
# knows it. The second is more than the card can ever provide, which is a
# configuration error rather than a wait, so it must fail immediately with 413
# rather than blocking for the timeout.
curl -s "$URL/gpu/lease" \
  -d "{\"mb\": 4000, \"owner\": \"probe-fits\", \"ttl\": 8, \"wait\": 2}" \
  -w '  http %{http_code}\n' | head -c 300
echo
curl -s "$URL/gpu/lease" \
  -d "{\"mb\": 30000, \"owner\": \"probe-never\", \"ttl\": 8, \"wait\": 2}" \
  -w '  http %{http_code}\n' | head -c 300
echo

say "waiting for the holders to finish"
wait "${pids[@]}" 2>/dev/null || true
pids=()

sleep 2
say "after — grants released, card should be back where it started"
state

cat <<'EOF'

What to read here:
  * `granted` should track the holders, and `outstanding` should fall to near
    zero once each holder has actually allocated what it asked for.
  * `free` should be smaller than the card's free memory throughout: the
    reserve and the outstanding grants are both subtracted, which is the whole
    difference between this and reading nvidia-smi.
  * The two probes must differ: the one that fits is granted, and the one
    larger than the card fails 413 immediately rather than waiting out its
    timeout. "Never" and "not yet" are different answers and the broker owes
    the caller the right one.
  * `after` must match `before`. If it does not, a release was missed and the
    grant is sitting there until its TTL runs out.
EOF
