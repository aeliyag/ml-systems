#!/usr/bin/env python3
"""Execute all code cells in final-project-merged.ipynb sequentially.

Run from repo root with your course venv active, e.g.:
  .venv/bin/python "final project/run_notebook.py"
"""

import json
import sys
import traceback
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "final-project-merged.ipynb"


def main() -> int:
    nb = json.loads(NOTEBOOK.read_text())
    namespace: dict = {"__name__": "__main__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", [])).strip()
        if not source:
            continue
        print(f"\n--- Cell {i} ---")
        try:
            exec(compile(source, f"cell_{i}", "exec"), namespace)
        except Exception:
            traceback.print_exc()
            return 1
    print("\nAll code cells executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
