import re

def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text).strip().lower())

def normalize_answer(answer: str) -> str:
    ans = normalize_text(answer)
    ans = ans.replace(',', '')
    return ans
