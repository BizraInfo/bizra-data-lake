#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
grep -n 'class EventBus\|def publish\|def wire_all' core/bus/subscribers.py
