#!/usr/bin/env bash
set -euo pipefail

MARKER_DIR=/tmp/brev-preflight
mkdir -p "$MARKER_DIR"
date -Is > "$MARKER_DIR/startup-script-ran.txt"
id > "$MARKER_DIR/id.txt" 2>&1 || true
uname -a > "$MARKER_DIR/uname.txt" 2>&1 || true
df -h > "$MARKER_DIR/df.txt" 2>&1 || true
nvidia-smi > "$MARKER_DIR/nvidia-smi.txt" 2>&1 || true
