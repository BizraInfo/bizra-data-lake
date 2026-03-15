#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
python3 -c "
from core.integration import constants
names = [x for x in dir(constants) if x.isupper()]
for n in sorted(names)[:40]:
    print(f'{n} = {getattr(constants, n, \"?\")}')
"
