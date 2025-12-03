import argparse
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
 
 
 
DEFAULT_VAL_BIN_DIR = Path(
    os.environ.get(
        "VAL_BIN_DIR",
        "/home/jovyan/Gideon/Gideon/planner/planners_and_val/VAL/build/linux64/Release/bin/",
    )
)
 
 
@dataclass
class ToolRun:
    command: Tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
 
 
@dataclass
class PlanEvaluation:
    status: str
    executed_instructions: int
    good_plan_prefix: str
    validate: ToolRun
    plan_to_valstep: Optional[ToolRun] = None
    valstep: Optional[ToolRun] = None
    updated_problem_text: Optional[str] = None
 
 
def _normalise_newlines(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")
 
 
def _write_temp(content: str, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        delete=False,
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(_normalise_newlines(content))
        return Path(handle.name)
 
 
def _run(command: Tuple[str, ...], timeout: int, cwd: Optional[Path] = None) -> ToolRun:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    return ToolRun(
        command=command,
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
 
 
def _extract_plan_prefix(plan_text: str, instruction_count: int) -> str:
    if instruction_count <= 0 or not plan_text:
        return ""
    collected = []
    seen = 0
    for line in plan_text.splitlines(keepends=True):
        collected.append(line)
        if ":" in line:
            seen += 1
            if seen == instruction_count:
                break
    return "".join(collected)
 
 
def _count_instructions(plan_text: str) -> Tuple[int, str]:
    if not plan_text:
        return 0, ""
    executed = 0
    for line in plan_text.splitlines():
        if ":" in line:
            executed += 1
    return executed, _extract_plan_prefix(plan_text, executed)
 
 
def _infer_from_stdout(stdout: str, plan_text: str) -> Tuple[int, str]:
    if not stdout:
        return 0, ""
    match = re.search(
        r"unsatisfied precondition.*time\s+([\d.]+)",
        stdout,
        flags=re.IGNORECASE,
    )
    if not match:
        return 0, ""
    try:
        timestamp = float(match.group(1))
    except ValueError:
        return 0, ""
    correct = (timestamp - 0.001) / 0.002
    executed = max(math.floor(correct + 1e-9), 0)
    return executed, _extract_plan_prefix(plan_text, executed)
 
 
def _ensure_val_bin(bin_dir: Optional[Path]) -> Path:
    return Path(bin_dir) if bin_dir else DEFAULT_VAL_BIN_DIR
 
 
def _call_validate(
    domain_text: str,
    problem_text: str,
    plan_text: str,
    val_bin_dir: Path,
    timeout: int,
) -> Tuple[ToolRun, Path, Path, Path]:
    plan_path = _write_temp(plan_text, ".pddl")
    domain_path = _write_temp(domain_text, ".pddl")
    problem_path = _write_temp(problem_text, ".pddl")
    command = (
        str(val_bin_dir / "Validate"),
        "-v",
        str(domain_path),
        str(problem_path),
        str(plan_path),
    )
    result = _run(command, timeout=timeout, cwd=plan_path.parent)
    return result, plan_path, domain_path, problem_path
 
 
def _evaluate_validate_output(validate: ToolRun, plan_text: str) -> Tuple[str, int, str]:
    stdout = validate.stdout
    if (
        validate.return_code == 0
        and "Plan executed successfully" in stdout
        and "Plan valid" in stdout
    ):
        exit()
        executed, prefix = _count_instructions(plan_text)
        return "plan_valid", executed, prefix or plan_text
    if "Goal not satisfied" in stdout:
        executed, prefix = _count_instructions(plan_text)
        return "goal_not_satisfied", executed, prefix
    if "has an unsatisfied precondition" in stdout:
        executed, prefix = _infer_from_stdout(stdout, plan_text)
        return "unsatisfied_precondition", executed, prefix
    return "invalid", 0, ""
 
 
def _run_plan_to_valstep_and_valstep(
    partial_plan_text: str,
    domain_path: Path,
    problem_path: Path,
    val_bin_dir: Path,
    timeout: int,
    progress_filename: str,
) -> Tuple[Optional[ToolRun], Optional[ToolRun], Optional[str]]:
    if not partial_plan_text.strip():
        return None, None, None
 
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        plan_path = temp_dir / "partial-plan.pddl"
        plan_path.write_text(_normalise_newlines(partial_plan_text), encoding="utf-8")
 
        temp_domain = temp_dir / "domain.pddl"
        temp_problem = temp_dir / "problem.pddl"
        temp_domain.write_text(domain_path.read_text(encoding="utf-8"), encoding="utf-8")
        temp_problem.write_text(problem_path.read_text(encoding="utf-8"), encoding="utf-8")
 
        plan_vs_path = temp_dir / "plan.vs"
        progress_path = temp_dir / progress_filename
 
        planar = _run(
            (str(val_bin_dir / "PlanToValStep"), str(plan_path)),
            timeout=timeout,
            cwd=temp_dir,
        )
 
        valstep_result: Optional[ToolRun] = None
        updated_problem_text: Optional[str] = None
 
        if planar.return_code == 0 and planar.stdout:
            vs_content = _normalise_newlines(planar.stdout)
            with plan_vs_path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(vs_content)
                if not vs_content.endswith("\n"):
                    handle.write("\n")
                handle.write(f"w {progress_filename}\n")
                handle.write("q\n")
 
            valstep_result = _run(
                (
                    str(val_bin_dir / "ValStep"),
                    "-i",
                    str(plan_vs_path),
                    str(temp_domain),
                    str(temp_problem),
                ),
                timeout=timeout,
                cwd=temp_dir,
            )
            if progress_path.exists():
                updated_problem_text = progress_path.read_text(encoding="utf-8")
 
        return planar, valstep_result, updated_problem_text
 
 
def validate_plan_text(
    domain_text: str,
    problem_text: str,
    plan_text: str,
    *,
    timeout: int = 10,
    val_bin_dir: Optional[Path] = None,
    progress_filename: str = "progress.pddl",
) -> PlanEvaluation:
    val_bin_dir = _ensure_val_bin(val_bin_dir)
 
    validate_result, plan_path, domain_path, problem_path = _call_validate(
        domain_text, problem_text, plan_text, val_bin_dir, timeout
    )
 
    status, executed_instructions, good_plan_prefix = _evaluate_validate_output(
        validate_result, plan_text
    )
 
    plan_to_valstep_result: Optional[ToolRun] = None
    valstep_result: Optional[ToolRun] = None
    updated_problem_text: Optional[str] = None
 
    try:
        if good_plan_prefix:
            plan_to_valstep_result, valstep_result, updated_problem_text = (
                _run_plan_to_valstep_and_valstep(
                    good_plan_prefix,
                    domain_path,
                    problem_path,
                    val_bin_dir,
                    timeout,
                    progress_filename,
                )
            )
    finally:
        for path in (plan_path, domain_path, problem_path):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
 
    return PlanEvaluation(
        status=status,
        executed_instructions=executed_instructions,
        good_plan_prefix=good_plan_prefix,
        validate=validate_result,
        plan_to_valstep=plan_to_valstep_result,
        valstep=valstep_result,
        updated_problem_text=updated_problem_text,
    )
 
 
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Validate plans, extract executable prefixes, and progress problems."
    )
    parser.add_argument("plan", help="Path to the plan file.")
    parser.add_argument("domain", help="Path to the domain file.")
    parser.add_argument("problem", help="Path to the problem file.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds for each VAL invocation.",
    )
    parser.add_argument(
        "--val-bin-dir",
        type=str,
        default=None,
        help="Directory containing Validate, PlanToValStep, and ValStep.",
    )
    parser.add_argument(
        "--progress-filename",
        type=str,
        default="progress.pddl",
        help="Filename written by ValStep when progressing the problem.",
    )
    parser.add_argument(
        "--write-updated-problem",
        type=str,
        default=None,
        help="If set, write the progressed problem to this path when available.",
    )
 
    args = parser.parse_args()
 
    plan_path = Path(args.plan)
    domain_path = Path(args.domain)
    problem_path = Path(args.problem)
 
    plan_text = plan_path.read_text(encoding="utf-8")
    domain_text = domain_path.read_text(encoding="utf-8")
    problem_text = problem_path.read_text(encoding="utf-8")
 
    result = validate_plan_text(
        domain_text,
        problem_text,
        plan_text,
        timeout=args.timeout,
        val_bin_dir=Path(args.val_bin_dir) if args.val_bin_dir else None,
        progress_filename=args.progress_filename,
    )
 
    print(f"Plan status           : {result.status}")
    print(f"Executed instructions : {result.executed_instructions}")
    if result.good_plan_prefix:
        print("Good plan prefix:")
        print(result.good_plan_prefix)
 
    print("Validate command      :", " ".join(result.validate.command))
    print("Validate return code  :", result.validate.return_code)
    if result.validate.stdout:
        print("Validate stdout:")
        print(result.validate.stdout)
    if result.validate.stderr:
        print("Validate stderr:")
        print(result.validate.stderr)
 
    if result.plan_to_valstep:
        print("PlanToValStep command :", " ".join(result.plan_to_valstep.command))
        print("PlanToValStep return  :", result.plan_to_valstep.return_code)
        if result.plan_to_valstep.stdout:
            print("PlanToValStep stdout:")
            print(result.plan_to_valstep.stdout)
        if result.plan_to_valstep.stderr:
            print("PlanToValStep stderr:")
            print(result.plan_to_valstep.stderr)
    else:
        print("PlanToValStep not executed.")
 
    if result.valstep:
        print("ValStep command       :", " ".join(result.valstep.command))
        print("ValStep return        :", result.valstep.return_code)
        if result.valstep.stdout:
            print("ValStep stdout:")
            print(result.valstep.stdout)
        if result.valstep.stderr:
            print("ValStep stderr:")
            print(result.valstep.stderr)
    else:
        print("ValStep not executed.")
 
    if result.updated_problem_text:
        if args.write_updated_problem:
            Path(args.write_updated_problem).write_text(
                result.updated_problem_text, encoding="utf-8"
            )
            print(f"Updated problem written to {args.write_updated_problem}")
        else:
            print("Updated problem text:")
            print(result.updated_problem_text)
 
 
if __name__ == "__main__":
    _cli()