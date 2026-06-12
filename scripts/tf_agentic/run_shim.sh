#!/bin/bash
source /root/openclaw-venv/bin/activate
cd /tmp
exec env PORT=8021 python -u /tmp/tf_shim.py
