#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
python3 -c "
from core.integration import constants
names = [x for x in dir(constants) if x.isupper()]
for n in sorted(names):
    if 'IHSAN' in n or 'SNR' in n or 'KERNEL' in n or 'GINI' in n:
        print(f'{n} = {getattr(constants, n, \"?\")}')
"
