#!/bin/bash
# BASH_ENV overlay setup script (sourced automatically before every bash -c command).
# Idempotent: runs the actual setup only once, then becomes a no-op.

if [ -f /tmp/.overlay_done ]; then
    return 0 2>/dev/null || exit 0
fi

VOLUME_MOUNT="/hf-cache-volume"
HF_HOME="/hf-home"
WORKSPACE="/home/agent/workspace"

# Set up HuggingFace cache overlay if the volume is mounted and non-empty
if [ -d "$VOLUME_MOUNT" ] && [ "$(ls -A $VOLUME_MOUNT 2>/dev/null)" ]; then
    mkdir -p /tmp/hf-overlay/upper /tmp/hf-overlay/work
    if command -v fuse-overlayfs &>/dev/null; then
        fuse-overlayfs \
            -o "lowerdir=$VOLUME_MOUNT,upperdir=/tmp/hf-overlay/upper,workdir=/tmp/hf-overlay/work" \
            "$HF_HOME" 2>&1
        if [ $? -eq 0 ]; then
            echo "HF cache overlay mounted at $HF_HOME" >&2
        else
            echo "WARNING: fuse-overlayfs failed, falling back to symlinks" >&2
            cp -as "$VOLUME_MOUNT/." "$HF_HOME/" 2>/dev/null || true
        fi
    else
        echo "WARNING: fuse-overlayfs not found, falling back to symlinks" >&2
        cp -as "$VOLUME_MOUNT/." "$HF_HOME/" 2>/dev/null || true
    fi
else
    echo "HF cache volume not mounted or empty at $VOLUME_MOUNT, using empty cache at $HF_HOME" >&2
fi

# Pre-create the .timer_start sentinel so timer.sh counts from right now
# (after overlay setup), not from the first timer.sh invocation.
date +%s > "$WORKSPACE/.timer_start"

# Mark setup as done so subsequent bash -c commands skip this script
touch /tmp/.overlay_done
