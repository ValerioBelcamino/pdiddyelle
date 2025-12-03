import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_tools_module(path: Path) -> Tuple[str, Dict[str, int]]:
    """Load a generated tools module and return domain name and action arities."""
    spec = importlib.util.spec_from_file_location("tools_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    domain_name = getattr(mod, "DOMAIN_NAME", "").strip()
    arities: Dict[str, int] = {}
    tools = getattr(mod, "TOOLS", [])
    for tool in tools:
        fn = tool.get("function", {})
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        params = fn.get("parameters", {})
        required = params.get("required", [])
        arities[name] = len(required)
    return domain_name, arities


def extract_domain_name(domain_text: str) -> Optional[str]:
    match = re.search(r"\(domain\s+([^\s)]+)", domain_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def parse_plan_line(line: str) -> Optional[Tuple[str, List[str]]]:
    """Parse a single plan line into (action, args)."""
    raw = line.strip()
    if not raw or raw.startswith(";"):
        return None
    # strip leading timestamp if present
    if ":" in raw:
        parts = raw.split(":", 1)
        if len(parts) == 2 and parts[0].replace(".", "", 1).replace("-", "", 1).isdigit():
            raw = parts[1].strip()
    raw = raw.strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
    tokens = raw.replace("(", " ").replace(")", " ").split()
    if not tokens:
        return None
    return tokens[0].lower(), tokens[1:]


def load_dataset(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list.")
    return data


def filter_examples_by_domain(
    examples: List[Dict[str, str]],
    domain_name: str,
) -> List[Dict[str, str]]:
    target = domain_name.lower().replace("-", "_").strip()
    filtered = []
    for ex in examples:
        instr = str(ex.get("instruction", ""))
        dn = extract_domain_name(instr)
        if dn and dn.lower().replace("-", "_").strip() == target:
            filtered.append(ex)
    return filtered


def analyze_plans(examples: List[Dict[str, str]], arities: Dict[str, int]) -> Dict[str, int]:
    stats = {
        "plans_checked": 0,
        "lines_checked": 0,
        "actions_missing": 0,
        "arity_mismatches": 0,
    }
    missing_actions: Dict[str, int] = {}
    mismatch_actions: Dict[str, int] = {}

    for ex in examples:
        plan_text = str(ex.get("output", "")).strip()
        if not plan_text:
            continue
        stats["plans_checked"] += 1
        for line in plan_text.splitlines():
            parsed = parse_plan_line(line)
            if not parsed:
                continue
            action, args = parsed
            stats["lines_checked"] += 1
            expected_arity = arities.get(action)
            if expected_arity is None:
                stats["actions_missing"] += 1
                missing_actions[action] = missing_actions.get(action, 0) + 1
                continue
            if expected_arity != len(args):
                stats["arity_mismatches"] += 1
                mismatch_actions[action] = mismatch_actions.get(action, 0) + 1

    # Attach details
    stats_detail = {
        "missing_actions_detail": missing_actions,
        "arity_mismatches_detail": mismatch_actions,
    }
    stats.update(stats_detail)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Check if generated tools can represent reference plans.")
    parser.add_argument("--tools", required=True, help="Path to generated tools module (e.g., generated_tools/satellite_tools.py).")
    parser.add_argument("--dataset", default="test_500.json", help="Path to dataset with reference plans.")
    parser.add_argument("--max-plans", type=int, default=None, help="Limit number of plans checked.")
    args = parser.parse_args()

    tools_path = Path(args.tools)
    dataset_path = Path(args.dataset)
    if not tools_path.exists():
        raise SystemExit(f"Tools module not found: {tools_path}")
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    domain_name, arities = load_tools_module(tools_path)
    if not domain_name:
        print("[WARN] DOMAIN_NAME missing in tools module; will not filter by domain.")

    examples = load_dataset(dataset_path)
    if domain_name:
        filtered = filter_examples_by_domain(examples, domain_name)
        if filtered:
            examples = filtered
        else:
            print("[WARN] No dataset entries matched domain; checking against all examples.")

    if args.max_plans is not None:
        examples = examples[: args.max_plans]

    stats = analyze_plans(examples, arities)
    print(f"Domain: {domain_name or 'unknown'}")
    print(f"Plans checked: {stats['plans_checked']}")
    print(f"Plan lines checked: {stats['lines_checked']}")
    print(f"Missing actions: {stats['actions_missing']}")
    if stats.get("missing_actions_detail"):
        print("Missing actions detail:")
        for name, count in sorted(stats["missing_actions_detail"].items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}")
    print(f"Arity mismatches: {stats['arity_mismatches']}")
    if stats.get("arity_mismatches_detail"):
        print("Arity mismatches detail:")
        for name, count in sorted(stats["arity_mismatches_detail"].items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
