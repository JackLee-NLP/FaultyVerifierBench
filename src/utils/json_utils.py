import json, re

def extract_json_block(text: str):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?', '', text).strip()
        text = re.sub(r'```$', '', text).strip()
    match = re.search(r'(\{.*\}|\[.*\])', text, re.S)
    return match.group(1) if match else text

def safe_json_loads(text: str):
    try:
        return json.loads(extract_json_block(text))
    except Exception:
        return None
