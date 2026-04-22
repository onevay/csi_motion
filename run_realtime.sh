#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "Usage: ./run_realtime.sh <output_dir> <window> <overlap> <port1> <port2> <port3> [baud]"
  exit 1
fi

OUTPUT_DIR="$1"
WINDOW_SIZE="$2"
OVERLAP="$3"
PORT1="$4"
PORT2="$5"
PORT3="$6"
BAUD="${7:-921600}"

mkdir -p "$OUTPUT_DIR"
cmake -S "cpp_preprocessor" -B "cpp_preprocessor/build" >/dev/null
cmake --build "cpp_preprocessor/build" -j >/dev/null

cleanup() {
  if [ -n "${RECEIVER_PID:-}" ]; then kill "$RECEIVER_PID" 2>/dev/null || true; fi
  if [ -n "${WATCHER_PID:-}" ]; then kill "$WATCHER_PID" 2>/dev/null || true; fi
}

trap cleanup EXIT INT TERM

./venv/bin/python main.py receiver --output "$OUTPUT_DIR" --window "$WINDOW_SIZE" --overlap "$OVERLAP" --ports "$PORT1" "$PORT2" "$PORT3" --baud "$BAUD" &
RECEIVER_PID=$!

./venv/bin/python main.py watcher --watch-dir "$OUTPUT_DIR" --preprocessor "./cpp_preprocessor/build/preprocessor" --cleanup &
WATCHER_PID=$!

wait -n "$RECEIVER_PID" "$WATCHER_PID"
