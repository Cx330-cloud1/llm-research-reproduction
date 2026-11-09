import sys
sys.path.insert(0, ".")

from build_index_1m import split_sessions, session_hash

msgs = [
    {"role": "system", "content": "persona A"},
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
    {"role": "system", "content": "persona B"},
    {"role": "user", "content": "u2"},
]

sessions = split_sessions(msgs)

assert sessions == [
    ["persona A", "u1", "a1"],
    ["persona B", "u2"],
]

assert session_hash(sessions[0]) == session_hash(
    ["persona A", "u1", "a1"]
)
assert session_hash(sessions[0]) != session_hash(sessions[1])

print("TEST_BUILD_INDEX_1M_OK")
