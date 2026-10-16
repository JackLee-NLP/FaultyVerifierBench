import json, os
from typing import Iterable, List, Dict

def ensure_dir(path: str):
    directory = path if os.path.splitext(path)[1] == '' else os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(items: Iterable[Dict], path: str):
    ensure_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def append_jsonl(item: Dict, path: str):
    ensure_dir(path)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
