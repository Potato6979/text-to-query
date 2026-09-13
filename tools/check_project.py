from __future__ import annotations

import ast
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
REQUIRED_INDEXES = (
    "faiss.index",
    "metadata.pkl",
    "cypher_faiss.index",
    "cypher_metadata.pkl",
    "paired_benchmark_resources.json",
)
REQUIRED_PACKAGES = {
    "faiss": "faiss-cpu",
    "fastapi": "fastapi",
    "huggingface_hub": "huggingface-hub",
    "neo4j": "neo4j",
    "numpy": "numpy",
    "openai": "openai",
    "pydantic": "pydantic",
    "sentence_transformers": "sentence-transformers",
    "uvicorn": "uvicorn",
}
OPTIONAL_PACKAGES = {"tiktoken"}


def report(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def check_python_sources() -> int:
    failures = 0
    files = sorted(BACKEND.glob("*.py"))
    local_modules = {path.stem for path in files}
    referenced_local: set[str] = set()

    required = (BACKEND / "__init__.py", BACKEND / "api_server.py", BACKEND / "main.py")
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        report("FAIL", "Missing backend entry files: " + ", ".join(missing))
        failures += 1

    root_modules = sorted(path.name for path in ROOT.glob("*.py"))
    if root_modules:
        report("FAIL", "Python modules must be placed under backend/: " + ", ".join(root_modules))
        failures += 1

    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            report("FAIL", f"Cannot parse {path.name}: {type(exc).__name__}")
            failures += 1
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                referenced_local.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                referenced_local.add(node.module.split(".")[0])

    missing_local = sorted(
        name for name in referenced_local if name not in local_modules and name not in sys.stdlib_module_names
        and name not in REQUIRED_PACKAGES and name not in OPTIONAL_PACKAGES
    )
    if missing_local:
        report("FAIL", "Unclassified imports: " + ", ".join(missing_local))
        failures += 1
    else:
        report("OK", f"Parsed {len(files)} backend Python files; local imports are closed.")
    return failures


def check_dependencies() -> int:
    missing = [dist for module, dist in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing:
        report("FAIL", "Missing Python packages: " + ", ".join(missing))
        return 1
    report("OK", "Required Python packages are importable.")
    return 0


def check_indexes_and_data() -> int:
    failures = 0
    index_dir = ROOT / "schema_index"
    missing_indexes = [name for name in REQUIRED_INDEXES if not (index_dir / name).is_file()]
    if missing_indexes:
        report("FAIL", "Missing index files: " + ", ".join(missing_indexes))
        return 1
    report("OK", "Four retrieval indexes and the paired-resource catalog are present.")

    try:
        sql_rows = pickle.loads((index_dir / "metadata.pkl").read_bytes())
        missing_sql = set()
        for row in sql_rows:
            raw_path = Path(str(row.get("db_path", "")))
            path = raw_path if raw_path.is_absolute() else ROOT / raw_path
            if not path.is_file():
                missing_sql.add(str(row.get("db_id", "unknown")))
        if missing_sql:
            report("WARN", f"SQL data is absent for {len(missing_sql)} indexed databases; see data/README.md.")
        else:
            report("OK", "All indexed SQLite files are present.")
    except Exception as exc:
        report("FAIL", f"Cannot inspect SQL metadata: {type(exc).__name__}")
        failures += 1

    try:
        paired = json.loads((index_dir / "paired_benchmark_resources.json").read_text(encoding="utf-8"))
        resources = paired.get("resources", {})
        missing_paired: list[str] = []

        def paired_path(raw_path: object) -> Path:
            return ROOT / Path(str(raw_path or "").replace("\\", "/"))

        for resource in resources.values():
            keys = ("resource_path",) if resource.get("query_type") == "sql" else ("graph_records", "cypher_import")
            for key in keys:
                path = paired_path(resource.get(key))
                if not path.is_file():
                    missing_paired.append(str(path.relative_to(ROOT)))

        paired_dir = ROOT / "data" / "paired_benchmark"
        contract_files = (
            paired_dir / "manifest.json",
            paired_dir / "bridge_gold_mappings.json",
            paired_dir / "case_schema_v1.json",
        )
        missing_paired.extend(str(path.relative_to(ROOT)) for path in contract_files if not path.is_file())
        if missing_paired:
            report("FAIL", "Missing paired benchmark files: " + ", ".join(sorted(set(missing_paired))))
            failures += 1
        else:
            mappings = json.loads(contract_files[1].read_text(encoding="utf-8"))
            json.loads(contract_files[2].read_text(encoding="utf-8"))
            report("OK", f"All paired runtime resources and {len(mappings)} bridge mappings are present.")
    except (OSError, TypeError, ValueError) as exc:
        report("FAIL", f"Cannot inspect paired resource index: {type(exc).__name__}")
        failures += 1

    return failures


def check_frontend() -> int:
    required = (
        ROOT / "web" / "package.json",
        ROOT / "web" / "package-lock.json",
        ROOT / "web" / "src" / "main.tsx",
        ROOT / "web" / "src" / "App.tsx",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        report("FAIL", "Missing frontend files: " + ", ".join(missing))
        return 1
    report("OK", "Frontend entry files and lockfile are present.")
    return 0


def check_license_files() -> int:
    required = (
        ROOT / "LICENSE",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "licenses" / "Apache-2.0.txt",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        report("FAIL", "Missing license files: " + ", ".join(missing))
        return 1
    report("OK", "MIT and third-party license files are present.")
    return 0


def check_documentation() -> int:
    required = (
        ROOT / "README.md",
        ROOT / "docs" / "images" / "system-architecture.png",
        ROOT / "docs" / "images" / "sql-agent-flow.png",
        ROOT / "docs" / "images" / "cypher-agent-flow.png",
        ROOT / "docs" / "images" / "multi-step-runtime.png",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        report("FAIL", "Missing documentation files: " + ", ".join(missing))
        return 1
    report("OK", "README and architecture diagrams are present.")
    return 0


def main() -> int:
    failures = 0
    failures += check_python_sources()
    failures += check_dependencies()
    failures += check_indexes_and_data()
    failures += check_frontend()
    failures += check_license_files()
    failures += check_documentation()

    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        report("WARN", "DEEPSEEK_API_KEY is not set; LLM-backed queries will not run.")
    if not os.getenv("NEO4J_PASSWORD", "").strip():
        report("WARN", "NEO4J_PASSWORD is not set; Cypher execution may not connect.")

    if failures:
        report("FAIL", f"Project check found {failures} blocking issue(s).")
        return 1
    report("OK", "Project structure is valid. Warnings describe external runtime resources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
