#!/usr/bin/env bash
# Stop a running recorder server (the bracket keeps pkill from matching itself).
pkill -f "[b]rainscan.server" && echo "stopped" || echo "not running"
