from pydantic import BaseModel, Field
from typing import List, Optional

class QuestionItem(BaseModel):
    id: str
    question: str
    gold_answer: str
    gold_explanation: str
    domain: str
    difficulty: str
    reasoning_type: str = ""
    question_generator: str = ""

class BenchmarkItem(BaseModel):
    id: str
    domain: str
    difficulty: str
    question: str
    gold_answer: str
    gold_explanation: str
    solver_response: str
    solver_final_answer: str = ""
    solver_correctness: str
    solver_generator: str
    verifier_response: str
    verifier_correctness: str
    verifier_generator: str
    expected_verifier_judgment: str
    fault_category: List[str] = Field(default_factory=list)
    fault_type: List[str] = Field(default_factory=list)
    hallucination_type: List[str] = Field(default_factory=list)
    expected_final_answer: str
    expected_behavior: str = ""
    question_generator: str = ""
    validation_status: str = "unvalidated"
    split: Optional[str] = None
