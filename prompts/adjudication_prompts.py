"""Prompt construction for FaultyVerifierBench adjudication."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from src.config.taxonomy_loader import load_benchmark_taxonomies


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _render_allowed_labels(
    heading: str,
    labels: tuple[str, ...],
) -> str:
    """Render configured taxonomy labels without adding descriptions."""
    rendered_labels = "\n".join(
        f'- "{label}"'
        for label in labels
    )
    return f"{heading}:\n{rendered_labels}"


def build_adjudication_prompt(
    item: Dict[str, Any],
) -> Tuple[str, str]:
    """Build the deterministic zero-shot adjudication prompt.

    Fallacy and hallucination labels are loaded from the benchmark YAML
    configuration files. No taxonomy label or taxonomy description is
    hard-coded in this prompt module.

    The two taxonomy arrays are treated as independent closed sets.
    A selected label must be an exact member of the corresponding
    configured taxonomy.
    """

    required_fields = (
        "question",
        "solver_response",
        "verifier_response",
    )

    missing_fields = [
        field
        for field in required_fields
        if field not in item
        or item[field] is None
        or not str(item[field]).strip()
    ]

    if missing_fields:
        raise KeyError(
            "Adjudication item is missing required non-empty fields: "
            f"{missing_fields}"
        )

    fallacy_labels, hallucination_labels = (
        load_benchmark_taxonomies(PROJECT_ROOT)
    )

    if not fallacy_labels:
        raise ValueError(
            "The configured fallacy taxonomy is empty."
        )

    if not hallucination_labels:
        raise ValueError(
            "The configured hallucination taxonomy is empty."
        )

    allowed_fallacies = _render_allowed_labels(
        "Allowed fallacy labels",
        fallacy_labels,
    )
    allowed_hallucinations = _render_allowed_labels(
        "Allowed hallucination labels",
        hallucination_labels,
    )

    system = """You are an objective adjudicator in a multi-agent reasoning system.

Evaluate the verifier feedback against the original question and solver response.
Determine whether the verifier feedback should be accepted or rejected, identify applicable configured fallacy and hallucination types, decide whether the solver answer requires material revision, and provide the correct final answer.

The verifier judgment must be exactly "accept" or "reject".
Use "accept" only when all substantive verifier feedback is materially correct.
Use "reject" when any substantive verifier feedback contains a material error, hallucination, or fallacious argument.

Return exactly one complete JSON object and no other text.
Follow the requested JSON schema exactly.

Treat fallacy_detected and hallucination_detected as two separate closed-set enum arrays.
A label may be placed in fallacy_detected only if it appears exactly in the provided allowed fallacy-label list.
A label may be placed in hallucination_detected only if it appears exactly in the provided allowed hallucination-label list.
Never copy a hallucination label into fallacy_detected merely because the verifier feedback is logically faulty.
Never copy a fallacy label into hallucination_detected merely because the verifier feedback is false or misleading.
Classify each detected fault independently against the two configured lists.
Do not duplicate a detected fault across both taxonomy arrays unless the same exact configured label explicitly appears in both allowed-label lists and both classifications genuinely apply.

Values in taxonomy arrays must be exact configured labels only.
Never place explanations, claims, sentences, prefixes, suffixes, punctuation, or newly invented labels in taxonomy arrays.
Place natural-language reasoning only in the point lists, explanation arrays, and justification.
Do not force a taxonomy label when no configured label applies.
A verifier response can be faulty even when both taxonomy arrays are empty.
Keep all text concise and do not repeat the full inputs.
The response must fit within the available generation limit."""

    user = f"""Original question:
{item['question']}

Solver response:
{item['solver_response']}

Verifier feedback:
{item['verifier_response']}

{allowed_fallacies}

{allowed_hallucinations}

Return exactly this JSON structure:
{{
  "verifier_judgment": "accept | reject",
  "valid_verifier_points": [],
  "faulty_verifier_points": [],
  "fallacy_detected": [],
  "fallacy_explanations": [],
  "hallucination_detected": [],
  "hallucination_explanations": [],
  "should_revise_solver_answer": false,
  "final_answer": "",
  "justification": ""
}}

JSON requirements:
1. Output one valid JSON object only.
2. Include every required key exactly once.
3. Use exactly "accept" or "reject" for verifier_judgment.
4. Use "accept" only when all substantive verifier feedback is materially correct.
5. Use "reject" when any substantive verifier feedback contains a material error, hallucination, or fallacious argument.
6. Use a JSON boolean, not a quoted string, for should_revise_solver_answer.
7. Set should_revise_solver_answer according to whether the solver answer requires a material correctness change, not a stylistic change.
8. Return final_answer as a JSON string even when the answer is a number, Boolean, formula, list, or short label.
9. Use double quotes for every key and string value.
10. Do not use Markdown fences or add text outside the JSON object.
11. Do not use LaTeX commands, dollar-sign mathematics, or raw backslashes in any string.
12. Write mathematical expressions in plain text using ordinary words and JSON-safe symbols.
13. Do not place literal line breaks inside JSON string values.
14. Keep valid_verifier_points, faulty_verifier_points, explanation entries, and justification concise.
15. Use empty arrays when no valid point, faulty point, fallacy, hallucination, or taxonomy explanation is present.
16. fallacy_detected is a closed-set enum array. Every entry must be copied exactly from the allowed fallacy labels above.
17. hallucination_detected is a separate closed-set enum array. Every entry must be copied exactly from the allowed hallucination labels above.
18. Check membership independently for each taxonomy array.
19. A label that appears only in the allowed hallucination labels must never be placed in fallacy_detected.
20. A label that appears only in the allowed fallacy labels must never be placed in hallucination_detected.
21. Do not copy the same detected fault into both taxonomy arrays unless the same exact configured label appears in both allowed lists and both classifications genuinely apply.
22. Do not alter the spelling, capitalization, underscores, or wording of configured taxonomy labels.
23. Do not combine a taxonomy label with an explanation in the same string.
24. Do not put natural-language descriptions in fallacy_detected or hallucination_detected.
25. If no allowed fallacy label applies, return an empty fallacy_detected array.
26. If no allowed hallucination label applies, return an empty hallucination_detected array.
27. Do not force every faulty verifier response into either or both taxonomies.
28. A factual, mathematical, scientific, or interpretive error is not automatically a logical fallacy.
29. A logical reasoning defect is not automatically a hallucination.
30. Each fallacy_explanations entry must be a JSON object containing exactly "type" and "explanation".
31. The type in each fallacy_explanations entry must exactly match a label present in fallacy_detected.
32. Every label in fallacy_detected must have exactly one corresponding entry in fallacy_explanations.
33. Each hallucination_explanations entry must be a JSON object containing exactly "type" and "explanation".
34. The type in each hallucination_explanations entry must exactly match a label present in hallucination_detected.
35. Every label in hallucination_detected must have exactly one corresponding entry in hallucination_explanations.
36. Do not include an explanation entry for a taxonomy label that is not selected.
37. Do not include duplicate labels in either taxonomy array.
38. Do not include duplicate explanation entries for the same taxonomy label.
39. Before returning, verify each fallacy_detected value against only the allowed fallacy-label list.
40. Before returning, verify each hallucination_detected value against only the allowed hallucination-label list.
41. Remove any taxonomy value that is not an exact member of its corresponding allowed-label list.
42. Verify that the set of fallacy explanation types exactly equals the set of selected fallacy labels.
43. Verify that the set of hallucination explanation types exactly equals the set of selected hallucination labels.
44. Before returning, verify that verifier_judgment is exactly "accept" or "reject".
45. Before returning, verify that the complete response can be parsed directly by a strict JSON parser."""

    return system.strip(), user.strip()
