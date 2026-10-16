# Local adjudicator runner

Place this folder directly under the repository root:

```text
FaultyVerifierBench/
├── config/
├── data/
├── scripts/
├── src/
└── scripts_Ministral/
```

The launcher reads the project configuration, writes private runtime YAML files inside this folder, patches the client factory only in memory, and then executes `scripts/08_run_evaluation.py`.

## Setup

```bash
bash scripts_Ministral/setup_env.sh
```

Update `scripts_Ministral/models.yaml` if a local checkpoint path differs. The paper checkpoints are `Qwen3-4B-Instruct-2507`, `Qwen3-8B`, and `Ministral-3-14B-Instruct-2512-BF16`.

## Run

```bash
bash scripts_Ministral/run.sh --model qwen3_8b_local --split test
```

## RAM safety

- `QWEN38_RAM_RESERVE_GIB=3` is subtracted from current Linux `MemAvailable`
  when the Accelerate CPU memory ceiling is created.
- A watchdog checks system `MemAvailable` four times per second.
- If available RAM reaches `QWEN38_RAM_HARD_FLOOR_GIB=2`, the launcher exits
  immediately with status 137 to reduce the chance of scheduler deadlock.
- Disk offload is rejected.

This is a user-space safety guard, not a kernel cgroup guarantee. Keep swap
configured and do not run another local model concurrently.
