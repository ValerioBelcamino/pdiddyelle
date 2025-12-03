import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


DEFAULT_VAL_BIN_DIR = Path(
    os.environ.get(
        "VAL_BIN_DIR",
        "/home/acarfi/grpo-valerio/VAL/build/linux64/Release/bin",
    )
)


@dataclass
class ToolRunResult:
    """
    Container for the output of an external VAL-related tool invocation.
    """

    command: Tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str


@dataclass
class PlanValidationResult:
    """
    Summarises the outcome of validating a plan (complete or partial).
    """

    status: str
    executed_instructions: int
    good_plan_prefix: str
    validate: ToolRunResult
    plan_to_valstep: Optional[ToolRunResult] = None
    valstep: Optional[ToolRunResult] = None
    updated_problem_text: Optional[str] = None


@dataclass
class InitEntry:
    """
    Representation of a single literal/function assignment inside :init.
    """

    kind: str  # "bool" or "number"
    key: str
    atom: str
    function: Optional[str] = None
    value: Optional[str] = None

    def serialize(self) -> str:
        if self.kind == "bool":
            return self.atom
        if self.function is None:
            raise ValueError("Numeric init entry requires a function.")
        if self.value is None:
            raise ValueError("Numeric init entry requires a value.")
        return f"(= {self.function} {self.value})"


def write_temp_file(content: str, suffix: str) -> str:
    """
    Persist content to a temporary file and return the filesystem path.
    """
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


def extract_plan_prefix(plan_text: str, instruction_count: int) -> str:
    """
    Return the plan prefix that includes only the first `instruction_count`
    timestamped instructions.
    """
    if instruction_count <= 0 or not plan_text:
        return ""

    collected_lines = []
    seen = 0
    for line in plan_text.splitlines(keepends=True):
        collected_lines.append(line)
        if ":" in line:
            seen += 1
            if seen == instruction_count:
                break
    return "".join(collected_lines)


def count_executed_instructions(plan_text: str) -> Tuple[int, str]:
    """
    Count the timestamped instructions in the plan and return both the count
    and the corresponding plan segment.
    """
    if not plan_text:
        return 0, ""

    executed = 0
    for line in plan_text.splitlines():
        if ":" in line:
            executed += 1
    return executed, extract_plan_prefix(plan_text, executed)


def infer_correct_instructions_from_stdout(
    stdout: str,
    plan_text: str,
    timestamp_regex: str = r"unsatisfied precondition.*time\s+([\d.]+)",
) -> Tuple[int, str]:
    """
    Infer how many instructions are correct based on the failure timestamp
    reported by VAL, and return the matching plan prefix.
    """
    if not stdout:
        return 0, ""

    match = re.search(timestamp_regex, stdout, flags=re.IGNORECASE)
    if not match:
        return 0, ""

    try:
        timestamp = float(match.group(1))
    except ValueError:
        return 0, ""

    # Plan timestamps start at 0.001 and progress in steps of 0.002.
    correct_lines = (timestamp - 0.001) / 0.002
    executed = max(math.floor(correct_lines + 1e-9), 0)
    return executed, extract_plan_prefix(plan_text, executed)


def normalize_atom(atom: str) -> str:
    """
    Convert an atom (or functional expression) into a canonical lowercase form
    with single spaces between tokens. This aids stable comparisons between
    different sources.
    """
    tokens = re.findall(r"[()]|[^\s()]+", atom)
    result: List[str] = []
    prev: Optional[str] = None
    for token in tokens:
        if token == "(":
            if prev and prev not in ("(", None):
                result.append(" ")
            result.append("(")
        elif token == ")":
            result.append(")")
        else:
            if prev and prev not in ("(", None):
                result.append(" ")
            result.append(token.lower())
        prev = token
    return "".join(result)


def normalize_numeric_key(atom: str) -> str:
    """
    Normalise numeric atoms which may be represented either as (= ...) forms
    or directly as function expressions in ValStep output.
    """
    stripped = atom.strip()
    if stripped.startswith("(="):
        inner = stripped[1:-1].strip()
        if inner.startswith("="):
            inner = inner[1:].strip()
        if inner.startswith("("):
            depth = 0
            for idx, char in enumerate(inner):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        function_expr = inner[: idx + 1]
                        return normalize_atom(function_expr)
    return normalize_atom(atom)


def split_init_expressions(init_content: str) -> List[str]:
    """
    Split the raw :init content into top-level S-expressions.
    """
    expressions: List[str] = []
    depth = 0
    start: Optional[int] = None

    for idx, char in enumerate(init_content):
        if char == "(":
            if depth == 0:
                start = idx
            depth += 1
        elif char == ")":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    expr = init_content[start : idx + 1].strip()
                    if expr:
                        expressions.append(expr)
                    start = None
    return expressions


def parse_numeric_init_entry(expr: str) -> InitEntry:
    """
    Parse a numeric assignment of the form (= (function ...) value).
    """
    trimmed = expr.strip()
    inner = trimmed[1:-1].strip()  # remove outer parentheses

    if inner.startswith("="):
        inner = inner[1:].strip()
    else:
        # Support formats like ( = ... )
        eq_idx = inner.find("=")
        if eq_idx != -1:
            inner = inner[eq_idx + 1 :].strip()

    if not inner.startswith("("):
        raise ValueError(f"Unexpected numeric init format: {expr}")

    depth = 0
    split_idx = None
    for idx, char in enumerate(inner):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                split_idx = idx + 1
                break

    if split_idx is None:
        raise ValueError(f"Could not parse numeric init entry: {expr}")

    function_expr = inner[:split_idx].strip()
    value_part = inner[split_idx:].strip()

    if not value_part:
        raise ValueError(f"Missing value in numeric init entry: {expr}")

    return InitEntry(
        kind="number",
        key=normalize_atom(function_expr),
        atom=expr,
        function=function_expr,
        value=value_part,
    )


def parse_init_entries(init_content: str) -> List[InitEntry]:
    """
    Convert :init content into structured entries.
    """
    entries: List[InitEntry] = []
    for expr in split_init_expressions(init_content):
        stripped = expr.strip()
        if not stripped:
            continue
        if stripped.startswith("(=") or stripped.startswith("( ="):
            try:
                entries.append(parse_numeric_init_entry(stripped))
            except ValueError:
                # If parsing fails, treat as opaque boolean literal to avoid data loss.
                entries.append(
                    InitEntry(kind="bool", key=normalize_atom(stripped), atom=stripped)
                )
        else:
            entries.append(
                InitEntry(kind="bool", key=normalize_atom(stripped), atom=stripped)
            )
    return entries


def parse_valstep_updates(output: str) -> Dict[str, Dict[str, str]]:
    """
    Parse ValStep output into a mapping of atom/function keys to their latest
    truth values or numeric values.
    """
    updates: Dict[str, Dict[str, str]] = {}
    if not output:
        return updates

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if "now" not in line.lower() or "(" not in line or ")" not in line:
            continue

        atom_start = line.find("(")
        atom_end = line.rfind(")")
        if atom_end <= atom_start:
            continue

        atom_repr = line[atom_start : atom_end + 1]
        tail = line[atom_end + 1 :]

        bool_match = re.search(r"now\s+(true|false)\b", tail, flags=re.IGNORECASE)
        if bool_match:
            key = normalize_atom(atom_repr)
            if key not in updates:
                updates[key] = {
                    "kind": "bool",
                    "value": str(bool_match.group(1)).lower(),
                    "atom": atom_repr,
                }
            continue

        number_match = re.search(
            r"now(?:\s+\w+)*\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", tail
        )
        if number_match:
            key = normalize_numeric_key(atom_repr)
            if key not in updates:
                updates[key] = {
                    "kind": "number",
                    "value": number_match.group(1),
                    "atom": atom_repr,
                }

    return updates


def locate_init_section(problem_text: str) -> Optional[Tuple[int, int, int, str]]:
    """
    Locate the :init section within the problem text and return positional data.
    """
    match = re.search(r"\(\s*:init\b", problem_text, flags=re.IGNORECASE)
    if not match:
        return None

    start_idx = match.start()
    depth = 0
    end_idx: Optional[int] = None

    for idx in range(start_idx, len(problem_text)):
        char = problem_text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break

    if end_idx is None:
        raise ValueError("Unbalanced parentheses while locating :init section.")

    content_start = match.end()
    while content_start < end_idx and problem_text[content_start] in " \t\r\n":
        content_start += 1

    init_content = problem_text[content_start:end_idx]
    return start_idx, end_idx, content_start, init_content


def update_problem_with_valstep(problem_text: str, valstep_output: str) -> str:
    """
    Apply ValStep updates to the problem's :init section.
    """
    if not valstep_output:
        return problem_text

    location = locate_init_section(problem_text)
    if location is None:
        return problem_text

    start_idx, end_idx, content_start, init_content = location
    entries = parse_init_entries(init_content)
    updates = parse_valstep_updates(valstep_output)

    if not updates:
        return problem_text

    updated_entries: List[InitEntry] = []
    handled_keys: Set[str] = set()

    for entry in entries:
        update = updates.get(entry.key)
        if entry.kind == "bool":
            if update and update["kind"] == "bool":
                handled_keys.add(entry.key)
                if update["value"] == "true":
                    updated_entries.append(entry)
                # Drop entry when it becomes false.
            else:
                updated_entries.append(entry)
        else:  # numeric
            if update and update["kind"] == "number":
                handled_keys.add(entry.key)
                entry.value = update["value"]
            updated_entries.append(entry)

    for key, update in updates.items():
        if key in handled_keys:
            continue
        if update["kind"] == "bool" and update["value"] == "true":
            updated_entries.append(
                InitEntry(
                    kind="bool",
                    key=key,
                    atom=normalize_atom(update["atom"]),
                )
            )

    indent = "    "
    init_lines = ["(:init"]
    for entry in updated_entries:
        init_lines.append(f"{indent}{entry.serialize()}")
    init_lines.append(")")
    new_section = "\n".join(init_lines)

    updated_text = (
        problem_text[:start_idx] + new_section + problem_text[end_idx + 1 :]
    )

    return updated_text


def build_command(binary: Path, *operands: str) -> Tuple[str, ...]:
    return (str(binary), *operands)


def run_tool(
    binary: Path, *operands: str, timeout: int = 10
) -> ToolRunResult:
    """
    Execute a VAL binary and capture its output.
    """
    cmd = build_command(binary, *operands)
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return ToolRunResult(
        command=cmd,
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def ensure_binaries_exist(val_bin_dir: Path) -> Tuple[Path, Path, Path]:
    """
    Resolve the VAL binaries and verify that they exist on disk.
    """
    validate_bin = val_bin_dir / "Validate"
    plantovalstep_bin = val_bin_dir / "PlanToValStep"
    valstep_bin = val_bin_dir / "ValStep"

    missing = [
        str(binary)
        for binary in (validate_bin, plantovalstep_bin, valstep_bin)
        if not binary.exists()
    ]

    if missing:
        missing_str = "\n".join(missing)
        raise FileNotFoundError(
            f"Missing VAL executable(s):\n{missing_str}\n"
            "Set the VAL_BIN_DIR environment variable if they are located elsewhere."
        )

    return validate_bin, plantovalstep_bin, valstep_bin


def validate_plan_text(
    domain_text: str,
    problem_text: str,
    plan_text: str,
    timeout: int = 10,
    val_bin_dir: Optional[Path] = None,
) -> PlanValidationResult:
    """
    Validate a (possibly incomplete) plan and, when necessary, produce the
    executable prefix that succeeds before failure.
    """
    if val_bin_dir is None:
        val_bin_dir = DEFAULT_VAL_BIN_DIR
    else:
        val_bin_dir = Path(val_bin_dir)

    validate_bin, plantovalstep_bin, valstep_bin = ensure_binaries_exist(val_bin_dir)

    plan_path = write_temp_file(plan_text, ".pddl")
    domain_path = write_temp_file(domain_text, ".pddl")
    problem_path = write_temp_file(problem_text, ".pddl")

    temp_paths = [plan_path, domain_path, problem_path]
    try:
        validate_result = run_tool(
            validate_bin,
            "-v",
            domain_path,
            problem_path,
            plan_path,
            timeout=timeout,
        )

        status, executed_instructions, prefix = classify_plan_outcome(
            validate_result, plan_text
        )

        plan_to_valstep_result: Optional[ToolRunResult] = None
        valstep_result: Optional[ToolRunResult] = None
        updated_problem_text: Optional[str] = None

        if status in {"unsatisfied_precondition", "goal_not_satisfied"} and prefix:
            plan_to_valstep_result, valstep_result = analyse_partial_plan(
                prefix,
                plantovalstep_bin,
                valstep_bin,
                domain_path,
                problem_path,
                timeout=timeout,
            )
            if valstep_result and valstep_result.stdout:
                updated_problem_text = update_problem_with_valstep(
                    problem_text, valstep_result.stdout
                )

        return PlanValidationResult(
            status=status,
            executed_instructions=executed_instructions,
            good_plan_prefix=prefix,
            validate=validate_result,
            plan_to_valstep=plan_to_valstep_result,
            valstep=valstep_result,
            updated_problem_text=updated_problem_text,
        )
    finally:
        for path in temp_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def classify_plan_outcome(
    validate_result: ToolRunResult, plan_text: str
) -> Tuple[str, int, str]:
    """
    Interpret the output of VAL's Validate binary and determine the plan status.
    """
    stdout_lower = validate_result.stdout.lower()

    if (
        validate_result.return_code == 0
        and "plan executed successfully" in stdout_lower
        and "plan valid" in stdout_lower
    ):
        executed, prefix = count_executed_instructions(plan_text)
        return "plan_valid", executed, prefix

    if "has an unsatisfied precondition" in stdout_lower:
        executed, prefix = infer_correct_instructions_from_stdout(
            validate_result.stdout, plan_text
        )
        return "unsatisfied_precondition", executed, prefix

    if "goal not satisfied" in stdout_lower:
        executed, prefix = count_executed_instructions(plan_text)
        return "goal_not_satisfied", executed, prefix

    return "invalid", 0, ""


def analyse_partial_plan(
    partial_plan: str,
    plantovalstep_bin: Path,
    valstep_bin: Path,
    domain_path: str,
    problem_path: str,
    timeout: int = 10,
) -> Tuple[Optional[ToolRunResult], Optional[ToolRunResult]]:
    """
    Convert a partial plan using PlanToValStep and subsequently run ValStep.
    """
    if not partial_plan.strip():
        return None, None

    partial_plan_path = write_temp_file(partial_plan, ".pddl")
    temp_paths = [partial_plan_path]
    plan_vs_path: Optional[str] = None

    try:
        plan_to_valstep_result = run_tool(
            plantovalstep_bin, partial_plan_path, timeout=timeout
        )

        valstep_result: Optional[ToolRunResult] = None

        if plan_to_valstep_result.return_code == 0 and plan_to_valstep_result.stdout:
            plan_vs_path = write_temp_file(plan_to_valstep_result.stdout, ".vs")
            temp_paths.append(plan_vs_path)

            valstep_result = run_tool(
                valstep_bin,
                '-i',
                plan_vs_path,
                domain_path,
                problem_path,
                timeout=timeout,
            )

        return plan_to_valstep_result, valstep_result
    finally:
        for path in temp_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def _load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _print_tool_summary(tool_name: str, result: Optional[ToolRunResult]) -> None:
    if result is None:
        print(f"{tool_name}: not executed")
        return

    print(f"{tool_name} command : {' '.join(result.command)}")
    print(f"{tool_name} return  : {result.return_code}")
    if result.stdout:
        print(f"{tool_name} stdout:\n{result.stdout}")
    if result.stderr:
        print(f"{tool_name} stderr:\n{result.stderr}")


def main() -> None:
    """
    Simple CLI for manual validation:
        python partial_plan_validator.py plan.pddl domain.pddl problem.pddl
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate PDDL plans and extract executable prefixes."
    )
    parser.add_argument("plan", help="Path to the plan file.")
    parser.add_argument("domain", help="Path to the domain file.")
    parser.add_argument("problem", help="Path to the problem file.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout (in seconds) for each VAL invocation.",
    )
    parser.add_argument(
        "--val-bin-dir",
        type=str,
        default=None,
        help="Directory containing Validate, PlanToValStep, and ValStep.",
    )
    parser.add_argument(
        "--write-updated-problem",
        type=str,
        default=None,
        help="If provided, write the updated problem text to this path when available.",
    )

    args = parser.parse_args()

    plan_text = _load_file(args.plan)
    domain_text = _load_file(args.domain)
    problem_text = _load_file(args.problem)

    result = validate_plan_text(
        domain_text,
        problem_text,
        plan_text,
        timeout=args.timeout,
        val_bin_dir=Path(args.val_bin_dir) if args.val_bin_dir else None,
    )

    print(f"Plan status           : {result.status}")
    print(f"Executed instructions : {result.executed_instructions}")
    if result.good_plan_prefix:
        print("Good plan prefix:")
        print(result.good_plan_prefix)

    _print_tool_summary("Validate", result.validate)
    _print_tool_summary("PlanToValStep", result.plan_to_valstep)
    _print_tool_summary("ValStep", result.valstep)

    if result.updated_problem_text:
        if args.write_updated_problem:
            with open(args.write_updated_problem, "w", encoding="utf-8") as handle:
                handle.write(result.updated_problem_text)
            print(f"Updated problem written to {args.write_updated_problem}")
        else:
            print("Updated problem text:")
            print(result.updated_problem_text)


if __name__ == "__main__":
    main()
