def build_correct_verifier_prompt(question, gold_answer, solver_response, solver_correctness):
    system = "You are a verifier agent giving accurate feedback."
    user = f"""
Question: {question}
Gold answer: {gold_answer}
Solver response: {solver_response}
Solver correctness: {solver_correctness}

Generate correct verifier feedback. Return ONLY JSON:
{{
  "verifier_response":"...",
  "verifier_correctness":"correct",
  "expected_verifier_judgment":"accept" if solver_correctness == "correct" else "reject_solver",
  "valid_verifier_points":["..."]
}}
"""
    return system, user

def build_fallacy_verifier_prompt(question, gold_answer, solver_response, fallacy_type):
    system = "You generate faulty verifier feedback for a benchmark."
    user = f"""
Question: {question}
Gold answer: {gold_answer}
Solver response: {solver_response}
Target fallacy type: {fallacy_type}

Generate plausible but faulty verifier feedback containing the target fallacy.
Do not reveal it is intentionally faulty. Return ONLY JSON:
{{
  "verifier_response":"...",
  "verifier_correctness":"faulty",
  "fault_category":["fallacy"],
  "fault_type":["{fallacy_type}"],
  "hallucination_type":[],
  "faulty_verifier_points":["..."],
  "expected_verifier_judgment":"reject"
}}
"""
    return system, user

def build_hallucination_verifier_prompt(question, gold_answer, solver_response, hallucination_type):
    system = "You generate faulty verifier feedback with hallucinated claims for a benchmark."
    user = f"""
Question: {question}
Gold answer: {gold_answer}
Solver response: {solver_response}
Target hallucination type: {hallucination_type}

Generate plausible but faulty verifier feedback containing the target hallucination.
Return ONLY JSON:
{{
  "verifier_response":"...",
  "verifier_correctness":"faulty",
  "fault_category":["hallucination"],
  "fault_type":[],
  "hallucination_type":["{hallucination_type}"],
  "faulty_verifier_points":["..."],
  "expected_verifier_judgment":"reject"
}}
"""
    return system, user
