import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.generation.generate_solver_responses import (
    generate_solver_responses,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate correct and incorrect solver responses "
            "using multiple configured models."
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
        "--questions",
        default=None,
        help=(
            "Optional validated-question JSONL path. "
            "Defaults to paths.processed_data_dir/"
            "questions_validated.jsonl."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional solver output JSONL path. "
            "Defaults to paths.raw_data_dir/solver_raw.jsonl."
        ),
    )

    parser.add_argument(
        "--force-full",
        action="store_true",
        help=(
            "Clear the existing solver output and regenerate "
            "every question/model/mode combination."
        ),
    )

    args = parser.parse_args()

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "[SCRIPT] Running multi-model solver generation",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    generate_solver_responses(
        question_path=args.questions,
        output_path=args.output,
        models_path=args.models,
        config_path=args.config,
        force_full_regenerate=args.force_full,
    )


if __name__ == "__main__":
    main()