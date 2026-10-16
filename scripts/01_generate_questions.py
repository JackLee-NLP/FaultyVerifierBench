import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.generate_questions import generate_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--models", default="config/models.yaml")
    parser.add_argument("--domains", default="config/domains.yaml")
    parser.add_argument("--output", default=None)

    # If used, it clears old questions_raw.jsonl and regenerates everything.
    parser.add_argument("--force-full", action="store_true")

    args = parser.parse_args()

    print("[SCRIPT] Running multi-model question generation", flush=True)

    generate_questions(
        config_path=args.config,
        models_path=args.models,
        domains_path=args.domains,
        output_path=args.output,
        force_full_regenerate=args.force_full,
    )