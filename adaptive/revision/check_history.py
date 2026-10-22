#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
manifest=json.loads((root/'adaptive/revision/historical_hashes.json').read_text())
for name,expected in manifest.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print(f'{len(manifest)} historical evidence and fixture files unchanged')
