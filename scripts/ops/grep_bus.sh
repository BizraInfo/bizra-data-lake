#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
grep -n '_event_bus\|emit_event\|_emit_breath\|event_bus' core/node0/heartbeat.py
