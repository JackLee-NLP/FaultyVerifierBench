import sys
from pathlib import Path


# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.validation.validate_verifier import validate_verifier


def main() -> None:
    print("=" * 80, flush=True)
    print("[SCRIPT] Running verifier validation", flush=True)
    print("[SCRIPT] Taxonomy sources: config/fallacies.yaml, config/hallucinations.yaml", flush=True)
    print("=" * 80, flush=True)

    validate_verifier()


if __name__ == "__main__":
    main()
