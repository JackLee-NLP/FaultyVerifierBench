def build_question_validation_prompt(item):
    return "You validate benchmark questions.", f"""
Check whether this question and gold answer are valid.
Item: {item}
Return ONLY JSON: {{"question_valid": true, "gold_answer_correct": true, "issues": []}}
"""

def build_solver_validation_prompt(item):
    return "You validate solver labels.", f"""
Check whether solver_correctness is correct.
Item: {item}
Return ONLY JSON: {{"solver_label_correct": true, "correct_label": "correct|incorrect", "issues": []}}
"""

def build_verifier_validation_prompt(item):
    return "You validate verifier feedback labels.", f"""
Check verifier correctness and fault labels.
Item: {item}
Return ONLY JSON: {{"verifier_label_correct": true, "fault_label_correct": true, "issues": []}}
"""
