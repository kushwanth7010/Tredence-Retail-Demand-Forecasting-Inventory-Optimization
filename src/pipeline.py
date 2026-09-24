from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "src" / script)], check=True)


def main() -> None:
    run("generate_data.py")
    run("train.py")
    run("explain.py")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
