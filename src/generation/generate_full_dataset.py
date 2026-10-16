from .generate_questions import generate_questions
from .generate_solver_responses import generate_solver_responses
from .generate_verifier_feedback import generate_verifier_feedback

def generate_full_dataset():
    generate_questions()
    # Normally run validation between stages before these two calls.
    generate_solver_responses('data/raw/questions_raw.jsonl')
    generate_verifier_feedback('data/raw/solver_raw.jsonl')
