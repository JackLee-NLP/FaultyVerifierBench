#!/usr/bin/env bash
set -euo pipefail

HERE="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"

ROOT="$(
  cd "$HERE/.."
  pwd
)"

RESERVE_GIB="${QWEN38_SYSTEM_RESERVE_GIB:-3}"
HARD_FLOOR_GIB="${QWEN38_RAM_HARD_FLOOR_GIB:-2}"

AVAILABLE_KIB="$(
  awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo
)"

if [[ -z "$AVAILABLE_KIB" ]]; then
  echo "[ERROR] Unable to read MemAvailable." >&2
  exit 1
fi

RESERVE_KIB="$(
  awk -v gib="$RESERVE_GIB" \
    'BEGIN { printf "%.0f", gib * 1024 * 1024 }'
)"

HARD_FLOOR_KIB="$(
  awk -v gib="$HARD_FLOOR_GIB" \
    'BEGIN { printf "%.0f", gib * 1024 * 1024 }'
)"

MINIMUM_START_KIB="$(
  awk \
    -v reserve="$RESERVE_KIB" \
    -v floor="$HARD_FLOOR_KIB" \
    'BEGIN { print reserve + floor }'
)"

if (( AVAILABLE_KIB <= MINIMUM_START_KIB )); then
  AVAILABLE_GIB="$(
    awk -v kib="$AVAILABLE_KIB" \
      'BEGIN { printf "%.2f", kib / 1024 / 1024 }'
  )"

  echo "[ERROR] Insufficient available RAM." >&2
  echo "[ERROR] MemAvailable=${AVAILABLE_GIB}GiB" >&2
  echo "[ERROR] Required reserve=${RESERVE_GIB}GiB" >&2
  echo "[ERROR] Hard floor=${HARD_FLOOR_GIB}GiB" >&2
  exit 1
fi

MEMORY_MAX_KIB="$(
  awk \
    -v available="$AVAILABLE_KIB" \
    -v reserve="$RESERVE_KIB" \
    'BEGIN { print available - reserve }'
)"

ONE_GIB_KIB="$((1024 * 1024))"

MEMORY_HIGH_KIB="$(
  awk \
    -v maximum="$MEMORY_MAX_KIB" \
    -v margin="$ONE_GIB_KIB" \
    'BEGIN {
      value = maximum - margin
      if (value < 1048576) {
        value = 1048576
      }
      print value
    }'
)"

MEMORY_MAX_BYTES="$(
  awk -v kib="$MEMORY_MAX_KIB" \
    'BEGIN { printf "%.0f", kib * 1024 }'
)"

MEMORY_HIGH_BYTES="$(
  awk -v kib="$MEMORY_HIGH_KIB" \
    'BEGIN { printf "%.0f", kib * 1024 }'
)"

AVAILABLE_GIB="$(
  awk -v kib="$AVAILABLE_KIB" \
    'BEGIN { printf "%.2f", kib / 1024 / 1024 }'
)"

MEMORY_MAX_GIB="$(
  awk -v kib="$MEMORY_MAX_KIB" \
    'BEGIN { printf "%.2f", kib / 1024 / 1024 }'
)"

MEMORY_HIGH_GIB="$(
  awk -v kib="$MEMORY_HIGH_KIB" \
    'BEGIN { printf "%.2f", kib / 1024 / 1024 }'
)"

echo "[CGROUP] MemAvailable=${AVAILABLE_GIB}GiB"
echo "[CGROUP] System reserve=${RESERVE_GIB}GiB"
echo "[CGROUP] Watchdog hard floor=${HARD_FLOOR_GIB}GiB"
echo "[CGROUP] MemoryHigh=${MEMORY_HIGH_GIB}GiB"
echo "[CGROUP] MemoryMax=${MEMORY_MAX_GIB}GiB"
echo "[CGROUP] MemorySwapMax=0"

export QWEN38_RAM_RESERVE_GIB="$RESERVE_GIB"
export QWEN38_RAM_HARD_FLOOR_GIB="$HARD_FLOOR_GIB"

cd "$ROOT"

exec systemd-run \
  --user \
  --scope \
  --collect \
  --property="MemoryAccounting=yes" \
  --property="MemoryHigh=${MEMORY_HIGH_BYTES}" \
  --property="MemoryMax=${MEMORY_MAX_BYTES}" \
  --property="MemorySwapMax=0" \
  bash "$HERE/run.sh" "$@"
