def build_question_generation_prompt(domain: str, difficulty: str, n: int):
    system = "You generate high-quality benchmark questions for LLM reasoning evaluation."
    user = f"""
Generate {n} {difficulty} questions for domain: {domain}.
Requirements:
1. Each question must have one objectively correct answer.
2. Avoid subjective, ambiguous, or time-sensitive questions.
3. Include gold answer and gold explanation.
4. Return ONLY a JSON list.
Schema per item:
{{
  "question": "...",
  "gold_answer": "...",
  "gold_explanation": "...",
  "domain": "{domain}",
  "difficulty": "{difficulty}",
  "reasoning_type": "..."
}}
"""
    return system, user
