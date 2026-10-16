from src.dataset.io_utils import read_jsonl, write_jsonl


def test_json_io(tmp_path):
    path = tmp_path / "x.jsonl"
    write_jsonl([{"a": 1}], str(path))
    rows = read_jsonl(str(path))
    assert rows[0]["a"] == 1
