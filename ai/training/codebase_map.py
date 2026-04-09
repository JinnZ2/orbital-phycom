"""
Codebase Map - Programmatic discovery of the Orbital PHYCOM project structure.

Scans the repository to produce a machine-readable inventory of all modules,
classes, functions, and their relationships.

Usage:
    python ai/training/codebase_map.py              # Full scan
    python ai/training/codebase_map.py --json        # Output as JSON
    python ai/training/codebase_map.py --module core  # Scan specific module
"""

import ast
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def scan_python_file(filepath):
    """Extract classes, functions, and imports from a Python file.

    Args:
        filepath: Path to the Python file.

    Returns:
        Dict with file analysis results.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        return {"error": str(e)}

    result = {
        "path": str(filepath.relative_to(PROJECT_ROOT)),
        "lines": len(source.splitlines()),
        "classes": [],
        "functions": [],
        "imports": [],
        "constants": [],
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            result["classes"].append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
                "method_count": len(methods),
            })

        elif isinstance(node, ast.FunctionDef) and isinstance(
            getattr(node, "_parent", None), ast.Module
        ) or (isinstance(node, ast.FunctionDef) and node.col_offset == 0):
            # Top-level functions only
            if node.col_offset == 0:
                result["functions"].append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": [a.arg for a in node.args.args if a.arg != "self"],
                })

        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)

        elif isinstance(node, ast.ImportFrom) and node.module:
            result["imports"].append(node.module)

        elif isinstance(node, ast.Assign) and node.col_offset == 0:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    result["constants"].append({
                        "name": target.id,
                        "line": node.lineno,
                    })

    return result


def scan_directory(dirpath, module_name=None):
    """Scan a directory for Python files and produce an inventory.

    Args:
        dirpath: Path to the directory to scan.
        module_name: Optional filter for a specific module.

    Returns:
        Dict with directory scan results.
    """
    results = {
        "project_root": str(PROJECT_ROOT),
        "modules": {},
    }

    # Define scannable directories
    scan_dirs = ["core", "atmospheric", "nutrient_cycling", "robotics", "simulations", "ai"]

    if module_name:
        scan_dirs = [d for d in scan_dirs if d == module_name]

    for dirname in scan_dirs:
        module_path = dirpath / dirname
        if not module_path.is_dir():
            continue

        module_info = {
            "path": dirname,
            "files": [],
            "total_classes": 0,
            "total_functions": 0,
            "total_lines": 0,
        }

        for pyfile in sorted(module_path.rglob("*.py")):
            file_info = scan_python_file(pyfile)
            if "error" not in file_info:
                module_info["files"].append(file_info)
                module_info["total_classes"] += len(file_info["classes"])
                module_info["total_functions"] += len(file_info["functions"])
                module_info["total_lines"] += file_info["lines"]

        results["modules"][dirname] = module_info

    # Summary statistics
    results["summary"] = {
        "total_modules": len(results["modules"]),
        "total_files": sum(len(m["files"]) for m in results["modules"].values()),
        "total_classes": sum(m["total_classes"] for m in results["modules"].values()),
        "total_functions": sum(m["total_functions"] for m in results["modules"].values()),
        "total_lines": sum(m["total_lines"] for m in results["modules"].values()),
    }

    return results


def print_map(results):
    """Print a human-readable codebase map.

    Args:
        results: Dict from scan_directory.
    """
    s = results["summary"]
    print()
    print("=" * 60)
    print("  ORBITAL PHYCOM - CODEBASE MAP")
    print("=" * 60)
    print(f"\n  Modules: {s['total_modules']}  |  Files: {s['total_files']}  |  "
          f"Classes: {s['total_classes']}  |  Functions: {s['total_functions']}  |  "
          f"Lines: {s['total_lines']}")

    for name, module in results["modules"].items():
        print(f"\n  --- {name}/ ({module['total_lines']} lines) ---")

        for f in module["files"]:
            print(f"\n    {f['path']} ({f['lines']} lines)")

            for cls in f["classes"]:
                print(f"      class {cls['name']} (line {cls['line']}, {cls['method_count']} methods)")
                for method in cls["methods"][:5]:
                    print(f"        .{method}()")
                if len(cls["methods"]) > 5:
                    print(f"        ... +{len(cls['methods'])-5} more")

            for func in f["functions"]:
                args_str = ", ".join(func["args"][:4])
                if len(func["args"]) > 4:
                    args_str += ", ..."
                print(f"      def {func['name']}({args_str})  (line {func['line']})")

            if f["constants"]:
                const_names = [c["name"] for c in f["constants"][:5]]
                print(f"      constants: {', '.join(const_names)}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map the PHYCOM codebase structure")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--module", type=str, help="Scan specific module only")
    parser.add_argument("--save", action="store_true", help="Save map to file")
    args = parser.parse_args()

    results = scan_directory(PROJECT_ROOT, module_name=args.module)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_map(results)

    if args.save:
        outpath = Path(__file__).parent / "codebase_map.json"
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Map saved to {outpath}")
