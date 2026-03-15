#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
python3 -u core/pat/data_census.py 2>&1
