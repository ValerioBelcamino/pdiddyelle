import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = (
    "You are a PDDL planner. Given a PDDL domain and problem, reply with the plan only.\n"
    "Return a valid PDDL plan using standard syntax (one action per line, optionally timestamped "
    "like `0.00100: (action parameters)`).\n"
    "Do not include explanations, comments, or any text before or after the plan."
)

DEFAULT_VALIDATE_PATH = (
    Path(__file__).resolve().parent
    / "Gideon"
    / "Gideon"
    / "planner"
    / "planners_and_val"
    / "VAL"
    / "build"
    / "linux64"
    / "Release"
    / "bin"
    / "Validate"
)


def load_env_file(env_path: str) -> None:
    """Load a simple KEY=VALUE env file without overwriting existing env vars."""
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_examples(dataset_path: str) -> List[Dict[str, str]]:
    """Load the JSON dataset containing domain/problem/plan tuples."""
    with open(dataset_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of objects.")
    return data


def build_user_prompt(domain: str, problem: str) -> str:
    """Combine domain and problem into the user prompt."""
    return f"Domain:\n{domain}\n\nProblem:\n{problem}\n\nReturn only the plan."


def request_plan(
    domain: str,
    problem: str,
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
) -> str:
    """Call DeepSeek via LangChain and return the plan text."""
    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - dependency/setup issues
        raise RuntimeError(f"Failed to initialise ChatOpenAI: {exc}") from exc

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(domain, problem)),
    ]

    result = llm.invoke(messages)
    content = getattr(result, "content", None)
    if not content:
        raise RuntimeError(f"Unexpected LangChain response: {result!r}")
    return content.strip()


def _write_temp(content: str, suffix: str) -> str:
    """Persist content to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        delete=False,
        encoding="utf-8",
        newline="\n",
    ) as handle:
        normalised = content.replace("\r\n", "\n").replace("\r", "\n")
        handle.write(normalised)
        return handle.name


def validate_with_val(
    domain_text: str,
    problem_text: str,
    plan_text: str,
    validate_path: Path,
    timeout: int = 30,
) -> str:
    """
    Run VAL's Validate binary against the generated plan.

    Returns a status string: "valid", "invalid", or "error".
    """
    plan_path = _write_temp(plan_text, ".plan")
    domain_path = _write_temp(domain_text, ".pddl")
    problem_path = _write_temp(problem_text, ".pddl")

    cmd = [
        str(validate_path),
        "-v",
        domain_path,
        problem_path,
        plan_path,
    ]
    print(f"[Validate] Running: {' '.join(cmd)}")

    status = "invalid"
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        status = "valid" if result.returncode == 0 else "invalid"
        print(f"[Validate] Return code: {result.returncode}")
        if result.stdout.strip():
            print("[Validate] stdout:")
            print(result.stdout.strip())
        if result.stderr.strip():
            print("[Validate] stderr:")
            print(result.stderr.strip())
    except Exception as exc:
        status = "error"
        print(f"[Validate] Failed to run validation: {exc}")
    finally:
        for path in (plan_path, domain_path, problem_path):
            try:
                os.remove(path)
            except OSError:
                pass

    return status


def print_progress_bar(completed: int, total: int, status: str) -> None:
    """Render a simple ASCII loading bar with status."""
    width = 20
    filled = int(width * completed / total) if total else 0
    bar = "#" * filled + "-" * (width - filled)
    print(f"[{bar}] {completed}/{total} - {status}")


def select_items(
    examples: List[Dict[str, str]],
    index: int = None,
    count: int = None,
) -> List[Tuple[int, Dict[str, str]]]:
    """Pick which dataset rows to solve."""
    if index is not None:
        if index < 0 or index >= len(examples):
            raise IndexError(f"Index {index} out of range for dataset of length {len(examples)}.")
        return [(index, examples[index])]

    if count is None or count < 0:
        count = len(examples)
    return list(zip(range(count), examples[:count]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query DeepSeek to solve PDDL problems.")
    parser.add_argument(
        "--dataset",
        default="test_500.json",
        help="Path to the JSON file containing domain/problem/plan tuples.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="DeepSeek model name (overrides DEEPSEEK_MODEL or .env).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="DeepSeek API base URL (overrides DEEPSEEK_API_BASE or .env).",
    )
    parser.add_argument(
        "--api-key",
        help="DeepSeek API key (falls back to DEEPSEEK_API_KEY env var).",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Solve only the example at this zero-based index.",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Solve the first N examples from the dataset.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to a .env file with credentials and config.",
    )
    parser.add_argument(
        "--langsmith-project",
        default=None,
        help="LangSmith project name for tracing (overrides LANGSMITH_PROJECT or .env).",
    )
    parser.add_argument(
        "--validate-path",
        default=None,
        help="Path to the VAL Validate binary (overrides VALIDATE_PATH or default).",
    )
    parser.add_argument(
        "--validate-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for the Validate subprocess.",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY or provide --api-key.")

    model = args.model or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
    base_url = args.base_url or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    langsmith_project = args.langsmith_project or os.getenv("LANGSMITH_PROJECT")
    if langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = langsmith_project

    validate_path: Optional[Path]
    validate_override = args.validate_path or os.getenv("VALIDATE_PATH")
    if validate_override:
        validate_path = Path(validate_override)
    else:
        validate_path = DEFAULT_VALIDATE_PATH

    if not validate_path.exists():
        print(f"[WARN] Validate binary not found at {validate_path}; skipping validation.")
        validate_path = None

    examples = load_examples(args.dataset)
    targets = select_items(examples, args.index, args.count)
    results: List[Dict[str, str]] = []
    total_targets = len(targets)

    for loop_idx, (display_idx, example) in enumerate(targets, start=1):
        domain = str(example.get("instruction", ""))
        problem = str(example.get("input", ""))
        plan: str = ""
        request_error: Optional[str] = None
        validation_status = "skipped"
        try:
            plan = request_plan(
                domain=domain,
                problem=problem,
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout=args.timeout,
            )
        except Exception as exc:
            request_error = str(exc)
            print(f"[ERROR] Failed to generate plan for example {display_idx}: {request_error}")
        else:
            print(f"### Plan for example {display_idx}")
            print(plan)
            if validate_path:
                if plan.strip():
                    validation_status = validate_with_val(
                        domain_text=domain,
                        problem_text=problem,
                        plan_text=plan,
                        validate_path=validate_path,
                        timeout=args.validate_timeout,
                    )
                else:
                    validation_status = "skipped_empty_plan"
                    print("[Validate] Skipping validation; received empty plan.")
            else:
                validation_status = "skipped"

        if request_error:
            results.append(
                {
                    "example": str(display_idx),
                    "validation": "request_failed",
                }
            )
            print_progress_bar(loop_idx, total_targets, "REQUEST_FAILED")
            continue

        status_label = "PASSED" if validation_status == "valid" else "FAILED"
        if validation_status == "skipped":
            status_label = "SKIPPED"
        elif validation_status == "skipped_empty_plan":
            status_label = "SKIPPED (EMPTY PLAN)"
        elif validation_status == "error":
            status_label = "ERROR"
        elif validation_status == "request_failed":
            status_label = "FAILED"

        print(f"[Result] Example {display_idx}: {status_label}")
        print_progress_bar(loop_idx, total_targets, status_label)
        results.append(
            {
                "example": str(display_idx),
                "validation": validation_status,
            }
        )

    if results:
        print("\n=== Validation Summary ===")
        summary_counts: Dict[str, int] = {}
        for item in results:
            summary_counts[item["validation"]] = summary_counts.get(item["validation"], 0) + 1

        print(f"Total problems: {total_targets}")
        for key, count in sorted(summary_counts.items()):
            print(f"{key}: {count}")
        print("Per-example outcomes:")
        for item in results:
            print(f"  Example {item['example']}: {item['validation']}")
        print()


if __name__ == "__main__":
    main()
