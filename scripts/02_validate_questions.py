import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.validation.validate_questions import validate_questions


if __name__ == "__main__":
    print(
        "[SCRIPT] Running question validation",
        flush=True,
    )

    validate_questions()