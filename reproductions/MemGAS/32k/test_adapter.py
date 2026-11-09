import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/paper_repro")
OUT = ROOT / "adapters/memgas_personamem/personamem_memgas.json"

assert OUT.exists(), "EXPECTED_FAIL: adapter output does not exist yet"

data = json.load(open(OUT, encoding="utf-8"))

assert len(data) == 589, f"expected 589 questions, got {len(data)}"

for x in data:
    assert len(x["qa"]) == 1
    assert len(x["sessions"]) == len(x["sessions_ids"])
    assert len(x["sessions"]) == len(x["sessions_dates"])
    assert x["adapter_meta"]["visible_message_count"] == x["adapter_meta"]["end_index"]

print("ADAPTER_TEST_OK")
