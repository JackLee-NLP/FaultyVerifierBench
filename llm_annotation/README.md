# LLM annotation

This workflow uses three independent LLM judges:

- `gpt_5_6_luna`
- `deepseek`
- `qwen3_8_flash`

For every question, the runner:

1. Deduplicates only byte-for-byte-equivalent answer text after trimming outer
   whitespace. Semantically similar answers remain separate.
2. Keeps every original `model`/`instance_id` pair in the deduplicated record.
3. Groups at most 12 distinct answers in one judge request.
4. Hides source model names and replaces answers with anonymous labels.
5. Randomizes answer order with a reproducible seed.
6. Asks each judge to return `correct` or `incorrect` independently for every
   answer.

Input is JSONL. Each row must contain `question`, `gold_answer`,
`gold_explanation`, and an answer field (default: `verifier_response`). It
must also contain `instance_id` or `id`; `model`/`source_model` is optional.

Example:

```powershell
python llm_annotation\annotate.py `
  --input deepseek=data\results\model_outputs_deepseek_test.jsonl `
  --input qwen=data\results\model_outputs_qwen3_8_flash_test.jsonl `
  --output llm_annotation\results
```

The output directory contains:

- `deduplicated_answers.jsonl`: one row per distinct answer with all original
  mappings.
- `requests.jsonl`: the exact anonymous request payloads, without judge
  responses.
- `judgments_<judge>.jsonl`: one result file per judge, including independent
  judgments and the answer-to-original-ID mapping. This means the GPT judge
  can be run separately in another environment with `--judge gpt_5_6_luna`.
  Each line has a `judgments` array whose entries contain both `label` and
  `judgment`, for example `{"label": "A1", "judgment": "correct"}`.

Results are flushed after every completed request, so a later failed request
does not erase earlier labels. With reasoning enabled, use at least
`--max-output-tokens 4096`; the budget may include hidden reasoning tokens.
DeepSeek retries that end with `finish_reason='length'` and no visible content
automatically double the token budget for the next attempt, up to 32768.
  If all retries for a request fail, the runner writes a `status: "failed"` record
  with `missing_labels` and continues to the next request. A later run retries
  failed requests and skips only requests with `status: "completed"`.
  The output is compacted by `request_id`, so a successful retry replaces the
  older failed record instead of creating duplicate records. Keep a copy of the
  JSONL file if you want to retain the historical failure records.
Rerunning the command resumes from completed request IDs by default. Use
`--no-resume` only when intentionally replacing an output file.
Resume matching also checks the batch's `answer_ids`. Changing `--batch-size`
therefore creates a new request layout and will not reuse a completed request
from the old layout.

For DeepSeek, smaller batches reduce reasoning time and response size:

```powershell
python llm_annotation\annotate.py `
  --input benchmark=data\processed\benchmark_dataset.jsonl `
  --judge deepseek `
  --batch-size 3 `
  --max-output-tokens 8192 `
  --output llm_annotation\results_deepseek
```

Set `--dry-run` to build and validate the deduplicated requests without making
API calls.

To run only GPT-5.6 Luna in a separate environment:

```powershell
python human_annotation\llm_annotation\annotate.py `
  --input benchmark=data\processed\benchmark_dataset.jsonl `
  --judge gpt_5_6_luna `
  --output human_annotation\llm_annotation\results_gpt
```
