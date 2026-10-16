import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.generation.generate_verifier_feedback import (
    auto_generate_verifier_feedback,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate multi-model verifier feedback "
            "with automatic resume and concurrent API calls."
        )
    )

    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml.",
    )

    parser.add_argument(
        "--models",
        default="config/models.yaml",
        help="Path to models.yaml.",
    )

    parser.add_argument(
        "--solvers",
        default=None,
        help=(
            "Optional path to solver_validated.jsonl. "
            "If omitted, the path is read from config.yaml."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional path to verifier_raw.jsonl. "
            "If omitted, the path is read from config.yaml."
        ),
    )

    parser.add_argument(
        "--fallacies",
        default="config/fallacies.yaml",
        help="Path to fallacies.yaml.",
    )

    parser.add_argument(
        "--hallucinations",
        default="config/hallucinations.yaml",
        help="Path to hallucinations.yaml.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Optional override for generation.verifier_max_workers. "
            "Example: --workers 6"
        ),
    )

    parser.add_argument(
        "--per-model-workers",
        type=int,
        default=None,
        help=(
            "Optional override for "
            "generation.verifier_max_concurrent_per_model. "
            "Example: --per-model-workers 2"
        ),
    )

    parser.add_argument(
        "--force-full",
        action="store_true",
        help=(
            "Clear the existing verifier output and regenerate "
            "every solver/model/mode combination."
        ),
    )

    args = parser.parse_args()

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "[SCRIPT] Running concurrent multi-model "
        "verifier generation",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    auto_generate_verifier_feedback(
        solver_path=args.solvers,
        output_path=args.output,
        models_path=args.models,
        config_path=args.config,
        fallacies_path=args.fallacies,
        hallucinations_path=args.hallucinations,
        force_full_regenerate=args.force_full,
        max_workers=args.workers,
        max_concurrent_per_model=args.per_model_workers,
    )


if __name__ == "__main__":
    main()
