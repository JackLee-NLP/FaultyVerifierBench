def build_correct_solver_prompt(question: str, gold_answer: str, gold_explanation: str):
    system = "You are a solver agent that answers correctly and concisely."
    user = f"""
Question: {question}
Gold answer: {gold_answer}
Gold explanation: {gold_explanation}

Generate a correct solver response. Return ONLY JSON:
{{"solver_response":"...", "solver_final_answer":"...", "solver_correctness":"correct"}}
"""
    return system, user

def build_incorrect_solver_prompt(question: str, gold_answer: str):
    system = "You generate plausible but incorrect solver responses for benchmark construction."
    user = f"""
Question: {question}
Gold answer: {gold_answer}

Generate a realistic incorrect solver response.
Do not say it is intentionally wrong.
Return ONLY JSON:
{{"solver_response":"...", "solver_final_answer":"...", "solver_correctness":"incorrect", "intended_error_type":"...", "why_wrong":"..."}}
"""
    return system, user
