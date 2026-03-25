#!/bin/bash
set -e
MODULE_ID="wurl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Building Wurl for Ableton Move (ARM64) ==="

docker build -t wurl-builder "$SCRIPT_DIR"

rm -rf "$ROOT/dist/$MODULE_ID"
mkdir -p "$ROOT/dist/$MODULE_ID"

MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$ROOT:/build" \
  wurl-builder \
  sh -c '\
    dos2unix /build/src/dsp/wurl.c 2>/dev/null; \
    aarch64-linux-gnu-gcc \
      -O2 -shared -fPIC -ffast-math \
      -o /build/dist/wurl/dsp.so \
      /build/src/dsp/wurl.c \
      -lm -Wall -Wno-unused-variable'

cp "$ROOT/src/module.json" "$ROOT/dist/$MODULE_ID/"

tar -czf "$ROOT/dist/$MODULE_ID-module.tar.gz" -C "$ROOT/dist" "$MODULE_ID/"

echo "=== Build complete ==="
ls -la "$ROOT/dist/$MODULE_ID/"
