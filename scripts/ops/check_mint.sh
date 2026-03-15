#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
echo "=== TOKEN MINTER ==="
grep -n 'class TokenMinter\|def mint\|def compute' core/token/minter.py 2>/dev/null || echo "not at minter.py"
grep -rn 'class TokenMinter' core/token/ 2>/dev/null | head -5
echo ""
echo "=== BLOOM MODULE ==="
grep -n 'class\|def ' core/token/bloom.py | head -20
echo ""
echo "=== SNR ADAPTER ==="
grep -n 'class\|def ' core/iaas/snr_v2_adapter.py | head -15
echo ""
echo "=== EVIDENCE LEDGER ==="
grep -rn 'class EvidenceLedger\|class.*Ledger' core/proof_engine/ | head -10
echo ""
echo "=== GOLD PARQUET SCHEMA ==="
source .venv-linux/bin/activate
python3 -c "
import pyarrow.parquet as pq
for f in ['04_GOLD/sovereign_catalog.parquet', '04_GOLD/documents.parquet']:
    try:
        t = pq.read_metadata(f)
        schema = pq.read_schema(f)
        print(f'{f}: {t.num_rows} rows, {t.num_columns} cols')
        for i, name in enumerate(schema.names[:8]):
            print(f'  {name}: {schema.field(i).type}')
        print()
    except Exception as e:
        print(f'{f}: {e}')
"
