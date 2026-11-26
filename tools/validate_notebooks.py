"""Validate the generated notebooks: JSON structure, code-cell syntax, and
that every name imported from ppq.* actually exists."""
import ast
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "moshi"))

ok = True
for name in ("ppq_data_pipeline.ipynb", "ppq_training.ipynb",
             "ppq_huggingface_data_pipeline.ipynb"):
    path = ROOT / "notebooks" / name
    nb = json.loads(path.read_text(encoding="utf-8"))
    n_code = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            print(f"{name} cell {i}: SYNTAX ERROR {e}")
            ok = False
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ppq"):
                try:
                    mod = importlib.import_module(node.module)
                except Exception as e:
                    print(f"{name} cell {i}: cannot import {node.module}: {e}")
                    ok = False
                    continue
                for alias in node.names:
                    if not hasattr(mod, alias.name):
                        print(f"{name} cell {i}: {node.module} has no '{alias.name}'")
                        ok = False
    print(f"{name}: {len(nb['cells'])} cells, {n_code} code cells parsed")
print("ALL OK" if ok else "FAILURES FOUND")
sys.exit(0 if ok else 1)
