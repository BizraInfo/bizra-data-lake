#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
black core/reasoning/entropy_router.py 2>&1 | tail -1
git add core/reasoning/entropy_router.py
git commit -m "fix(reasoning): calibrate entropy router — imperative verbs + architectural signals

Added 3 sub-question patterns (redesign/refactor, thread-safe/concurrent,
while-maintaining/without-breaking) and 3 multi-domain patterns for
architectural complexity detection.

Added imperative verb bonus signal (create/build/design/implement/show me
etc.) as additive +0.15 max — original 6-signal weights preserved exactly.

Before: 'Redesign ReflexCache for concurrent UAB' scored 0.26 (TRIVIAL)
After:  same query scores MODERATE with GoT=True

16/16 entropy router tests pass, 134/134 reasoning module tests pass."
echo "EXIT: $?"
git log --oneline -3
