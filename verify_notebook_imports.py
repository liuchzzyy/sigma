import json
import os
import sys
import glob


def check_imports_in_notebook(notebook_path):
    print(f"Checking {notebook_path}...")
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Failed to load notebook JSON: {e}")
        return False

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]

    imports_to_test = []

    for cell in code_cells:
        source = cell.get("source", [])
        if isinstance(source, str):
            lines = source.split("\n")
        else:
            lines = source

        for line in lines:
            line = line.strip()
            # Simple heuristic to grab import statements
            if line.startswith("import ") or line.startswith("from "):
                # Skip comments
                if "#" in line:
                    line = line.split("#")[0].strip()
                if line:
                    imports_to_test.append(line)

    if not imports_to_test:
        print("  [WARN] No imports found.")
        return True

    print(f"  Found {len(imports_to_test)} import statements.")

    # Try to execute them
    failed_count = 0
    for imp in imports_to_test:
        try:
            exec(imp, {"__name__": "__main__"})
        except Exception as e:
            print(f"  [FAIL] {imp}")
            print(f"         Error: {e}")
            failed_count += 1

    if failed_count == 0:
        print("  [PASS] All imports successful.")
        return True
    else:
        print(f"  [FAIL] {failed_count} imports failed.")
        return False


def main():
    notebooks = glob.glob("tutorial/local/*.ipynb")
    notebooks.extend(glob.glob("tutorial/colab/*.ipynb"))

    all_passed = True
    for nb in notebooks:
        if not check_imports_in_notebook(nb):
            all_passed = False
        print("-" * 40)

    if all_passed:
        print("\nAll notebooks passed import checks.")
        sys.exit(0)
    else:
        print("\nSome notebooks failed import checks.")
        sys.exit(1)


if __name__ == "__main__":
    main()
