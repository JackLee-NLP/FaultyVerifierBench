# FaultyVerifierBench

Controlled synthetic benchmark for whole-message verifier-feedback adjudication, fault diagnosis, and operational answer preservation or recovery.

An adjudicator sees a question, a solver response, and a verifier message. It predicts whether to accept the feedback, labels any fallacy or hallucination, decides whether the solver answer needs material revision, and supplies a final answer. Correct and incorrect solver responses are crossed with correct and faulty verifier feedback:

| Case | Solver | Verifier | Gold actions |
| --- | --- | --- | --- |
| C1 | Correct | Correct | Accept; do not materially revise; preserve answer |
| C2 | Correct | Faulty | Reject; do not materially revise; preserve answer |
| C3 | Incorrect | Correct | Accept; materially revise; recover answer |
| C4 | Incorrect | Faulty | Reject; materially revise; recover answer |

The released test partition contains 18,974 records from 360 base questions. Reported comparisons use the 17,085 records completed by all six adjudicators. This repository does not include the human-annotation workspace.

## What is included

- `config/`: domains, fallacy and hallucination taxonomies, and model settings.
- `prompts/`, `scripts/`, `src/`: question, solver, verifier, and adjudicator pipeline.
- `data/splits/dev.jsonl`, `data/splits/test.jsonl`: current development and test partitions. All variants of a question stay in one split.
- `data/results/`: post-correction scores, common-set summaries, clustered-bootstrap intervals, coverage sensitivity, and per-record score rows for the six paper adjudicators.
- `scripts_Ministral/`: local Hugging Face evaluation entry point used for Qwen3-4B, Qwen3-8B, and Ministral-3-14B.

Raw generation dumps, intermediate processed files, pre-correction splits, model-output backups, and unused model runs are omitted. Score rows are the frozen scoring artifacts behind the paper tables. Adjudicator outputs can be regenerated with the evaluation scripts.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Local adjudication also needs the packages in `scripts_Ministral/requirements.txt` and the Hugging Face checkpoints named in `config/models.yaml`. Set those `model_name` paths to your local checkpoint directories before running a local model. Hosted calls read keys from the environment; never commit `.env`.

## Reproduce evaluation

Score the released test split from existing outputs, if you add `data/results/model_outputs_<model>_test.jsonl`:

```bash
python scripts/09_compute_scores.py --split test
```

Run one adjudicator:

```bash
python scripts/08_run_evaluation.py --model deepseek --split test
```

Paper adjudicator keys are `deepseek`, `qwen3_8_flash`, `gpt_5_6_luna`, `qwen3_4b_local`, `qwen3_8b_local`, and `Ministral-3-14B`.

## Reported metrics

- **VJA**: agreement with the benchmark verifier-correctness label.
- **FAA / TRR / ARS**: operational final-answer match, recovery match on incorrect-solver records, and joint judgment-and-answer success.
- **M0–M4**: deterministic answer matchers. M4 is the legacy operational matcher. M1 and M3 completion intervals assign unresolved records to nonmatch or match; they are not bounds on human semantic accuracy.

Common-set analyses use 17,085 shared instance IDs, 360 base-question clusters, 1,000 paired bootstrap resamples, and seed 2027.
