import json

from web.services.training.log_index import fingerprint, index_for, records, matches_for, search


def test_sparse_index_preserves_nonempty_record_numbers(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text('\n{"line":"first"}\nbroken\n[]\n{"line":"last"}\n')
    key = fingerprint(path)
    assert index_for(key)[1] == 4
    assert list(records(key, 1, 4)) == [(3, {"line": "last"})]
    assert search(key, "last", 0, "forward") == (3, {"line": "last"}, 1, 1)


def test_sparse_index_cache_and_file_changes(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text("".join(json.dumps({"line": f"row {i}"}) + "\n" for i in range(10000)))
    key = fingerprint(path)
    index = index_for(key)
    assert len(index[0]) == 40
    assert list(records(key, 9000, 9002))[0] == (9000, {"line": "row 9000"})
    assert index_for(key) is index
    matches = matches_for(key, "row")
    assert matches_for(key, "row") is matches
    assert search(key, "row", 10000, "forward")[0] == 0
    assert search(key, "row", -1, "backward")[0] == 9999
    with path.open("a") as handle:
        handle.write('{"line":"new"}\n')
    assert index_for(fingerprint(path))[1] == 10001
    path.write_text('{"line":"replacement"}\n')
    assert index_for(fingerprint(path))[1] == 1


def test_utf8_crlf_and_partial_record(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_bytes('{"line":"训练完成"}\r\n{"line":'.encode())
    key = fingerprint(path)
    assert index_for(key)[1] == 2
    assert len(list(records(key, 0, 2))) == 1
    with path.open("ab") as handle:
        handle.write(b'"done"}\n')
    assert list(records(fingerprint(path), 1, 2)) == [(1, {"line": "done"})]
