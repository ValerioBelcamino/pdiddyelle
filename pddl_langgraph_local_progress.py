"""
Local-model planner with tool-call style, per-step validation, and mid-run progressor.

Behavior:
- Exactly one tool call per turn; multiple tool_calls trigger a retry message.
- Per-step validation; on goal_not_reached with END we drop END; otherwise continue.
- On other failures we drop the last action, append error feedback, bump temp/seed.
- Mid-run: on hard errors we keep history; retries start fresh from validated prefix/problem
  returned by `validate_plan_text` (PlanToValStep/ValStep).
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from local_chat_model import create_local_chat_model

from partial_plan_validator import validate_plan_text

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

AGENT_SYSTEM_PROMPT = """You are a PDDL planning router that must emit plan steps via tools.
- Exactly ONE tool call per turn (native tool_calls or a single JSON code block with one entry).
- Do NOT return prose or plain text plans. No lists, no bullets, no multiple tool calls.
- Use the tool names exactly as given and keep argument order.
- Use the END tool only when the goal is satisfied."""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_env_file(env_path: str) -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_tools_module(path: Path):
    spec = importlib.util.spec_from_file_location("tool_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load tools module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def build_tools(module) -> List:
    tools: List = []
    seen: set = set()
    for tool_def in getattr(module, "TOOLS", []):
        fn = tool_def.get("function", {})
        name = fn.get("name")
        if not name:
            continue
        if name in seen:
            continue
        if name == "end":
            seen.add(name)
            continue
        description = fn.get("description", "")
        params = fn.get("parameters", {})
        required = params.get("required", [])

        def make_tool(t_name: str, t_desc: str, required_params: List[str]):
            @tool(t_name, description=t_desc)
            def _dynamic(**kwargs):
                parts = [str(kwargs.get(param, "")) for param in required_params]
                return f"{t_name} " + " ".join(parts)

            return _dynamic

        dynamic_tool = make_tool(name, description, required)
        setattr(dynamic_tool, "_pddl_required_params", required)
        tools.append(dynamic_tool)
        seen.add(name)

    @tool("end", description="Signal the plan is complete and stop emitting further actions.")
    def end_tool():
        return "end"

    setattr(end_tool, "_pddl_required_params", [])
    tools.append(end_tool)
    return tools


def build_tool_metadata(tools: List) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for tool in tools:
        name = getattr(tool, "name", getattr(tool, "__name__", "")).strip()
        if not name:
            continue
        required = list(getattr(tool, "_pddl_required_params", []))
        meta[name.lower()] = {"name": name, "required": required}
    return meta


def _tokens_to_args(required_params: List[str], tokens: List[str]) -> Dict[str, str]:
    args: Dict[str, str] = {}
    for idx, param in enumerate(required_params):
        args[param] = tokens[idx] if idx < len(tokens) else ""
    if tokens and len(tokens) > len(required_params) and required_params:
        args[required_params[-1]] = " ".join(tokens[len(required_params) - 1 :])
    return args


def parse_tool_call_text(text: str, tool_meta: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Parse text (plain, fenced, or XML-ish) into a tool call."""
    if not text:
        return None

    def from_json_block(raw: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        name = obj.get("name")
        obj_args = obj.get("arguments") or obj.get("args", {})
        if not name and isinstance(obj.get("function"), dict):
            name = obj["function"].get("name")
            obj_args = obj["function"].get("arguments") or obj["function"].get("args", {})
        if not name:
            return None
        meta = tool_meta.get(str(name).lower())
        if not meta:
            return None
        if isinstance(obj_args, dict):
            args = {param: str(obj_args.get(param, "")) for param in meta.get("required", [])}
        elif isinstance(obj_args, str):
            args = _tokens_to_args(meta.get("required", []), obj_args.split())
        else:
            args = _tokens_to_args(meta.get("required", []), [])
        return {"name": meta["name"], "args": args}

    def parse_tokens(tokens: List[str]) -> Optional[Dict[str, Any]]:
        if not tokens:
            return None
        action_key = tokens[0].lower()
        meta = tool_meta.get(action_key)
        if not meta:
            return None
        args_tokens = tokens[1:]
        args = _tokens_to_args(meta.get("required", []), args_tokens)
        return {"name": meta["name"], "args": args}

    def clean_line(raw_line: str) -> str:
        line = raw_line.strip()
        if not line:
            return line
        line = re.sub(r"^[\-*\d\.]+\s*", "", line)
        if ":" in line:
            prefix, rest = line.split(":", 1)
            if prefix.strip().lower() in {"action", "tool", "step", "call"}:
                line = rest.strip()
        if line.startswith("`") and line.endswith("`"):
            line = line[1:-1].strip()
        if line.startswith("(") and line.endswith(")"):
            line = line[1:-1].strip()
        line = re.sub(r"</?[^>]+>", "", line)
        return line

    fenced_blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, flags=re.DOTALL)
    for block in fenced_blocks:
        block = block.strip()
        if not block:
            continue
        parsed_json = from_json_block(block)
        if parsed_json:
            return parsed_json
        for raw_line in block.splitlines():
            candidate = clean_line(raw_line)
            parsed = parse_tokens(candidate.split())
            if parsed:
                return parsed

    xml_blocks = re.findall(r"<tool_call[^>]*>(.*?)</tool_call>", text, flags=re.DOTALL | re.IGNORECASE)
    for block in xml_blocks:
        block = block.strip()
        parsed_json = from_json_block(block)
        if parsed_json:
            return parsed_json
        for raw_line in block.splitlines():
            candidate = clean_line(raw_line)
            parsed = parse_tokens(candidate.split())
            if parsed:
                return parsed

    for raw_line in text.splitlines():
        candidate = clean_line(raw_line)
        parsed = parse_tokens(candidate.split())
        if parsed:
            return parsed

    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        parsed_json = from_json_block(json_match.group(0))
        if parsed_json:
            return parsed_json

    condensed = re.sub(r"\s+", " ", text).strip()
    for key, meta in tool_meta.items():
        match = re.search(rf"\b{re.escape(key)}\b", condensed, flags=re.IGNORECASE)
        if not match:
            continue
        tail = condensed[match.end() :].strip()
        parsed = parse_tokens([key] + tail.split())
        if parsed:
            return parsed
    return None


def coerce_tool_call(response: AIMessage, tool_meta: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    """Return a single tool-call dict and normalized plan line."""
    if not isinstance(response, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(response)}")

    existing_calls = list(getattr(response, "tool_calls", []) or [])
    parsed: Optional[Dict[str, Any]] = None
    call_id = None
    if existing_calls:
        call = existing_calls[0]
        call_id = call.get("id")
        name = call.get("name") or call.get("function", {}).get("name")
        args_raw = call.get("args")
        if args_raw is None and isinstance(call.get("function"), dict):
            args_raw = call["function"].get("arguments") or call["function"].get("args")
        if name:
            meta = tool_meta.get(name.lower())
            if meta:
                if isinstance(args_raw, str):
                    try:
                        args_raw = json.loads(args_raw)
                    except Exception:
                        args_raw = args_raw.strip()
                if isinstance(args_raw, dict):
                    args = {param: str(args_raw.get(param, "")) for param in meta.get("required", [])}
                elif isinstance(args_raw, str):
                    args = _tokens_to_args(meta.get("required", []), args_raw.split())
                else:
                    args = _tokens_to_args(meta.get("required", []), [])
                parsed = {"name": meta["name"], "args": args}

    if parsed is None:
        parsed = parse_tool_call_text(str(response.content or ""), tool_meta)

    if parsed is None:
        raw_content = str(response.content or "")
        line = ""
        for raw in raw_content.splitlines():
            candidate = raw.strip()
            if not candidate:
                continue
            candidate = candidate.strip("`")
            if candidate.startswith("(") and candidate.endswith(")"):
                candidate = candidate[1:-1].strip()
            candidate = re.sub(r"</?[^>]+>", "", candidate)
            if candidate:
                line = candidate
                break
        tokens = line.split()
        if not tokens or tokens[0].lower() not in tool_meta:
            tokens = ["end"]
        name = tokens[0]
        args = _tokens_to_args(tool_meta.get(name.lower(), {}).get("required", []), tokens[1:])
        parsed = {"name": name, "args": args}
        call_id = f"call_{uuid4().hex}"

    name = parsed["name"]
    args = parsed.get("args", {})
    required = tool_meta.get(name.lower(), {}).get("required", [])
    arg_list = [str(args.get(param, "")).strip() for param in required]
    plan_line = f"{name} {' '.join(arg_list)}".strip()
    return {"name": name, "args": args, "id": call_id or f"call_{uuid4().hex}"}, plan_line


def run_validate(
    domain_text: str,
    problem_text: str,
    plan_lines: List[str],
    validate_path: Path,
    timeout: int,
) -> Tuple[str, str, str, int]:
    plan_text = "\n".join(plan_lines)
    plan_path = Path(os.path.join("/tmp", "tmp_plan.plan"))
    domain_path = Path(os.path.join("/tmp", "tmp_domain.pddl"))
    problem_path = Path(os.path.join("/tmp", "tmp_problem.pddl"))
    plan_path.write_text(plan_text, encoding="utf-8")
    domain_path.write_text(domain_text, encoding="utf-8")
    problem_path.write_text(problem_text, encoding="utf-8")

    cmd = [
        str(validate_path),
        "-v",
        str(domain_path),
        str(problem_path),
        str(plan_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        status = "valid" if result.returncode == 0 else "invalid"
        return status, result.stdout, result.stderr, result.returncode
    except Exception as exc:
        return "error", "", str(exc), -1
    finally:
        for path in (plan_path, domain_path, problem_path):
            try:
                path.unlink()
            except OSError:
                pass


def classify_val_error(status: str, stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if status == "invalid":
        if "unsatisfied precondition" in text:
            return "unsatisfied_precondition"
        if "goal not satisfied" in text:
            return "goal_not_satisfied"
        return "invalid_other"
    if status == "error":
        if "parse" in text:
            return "parse_error"
        return "validate_error"
    return status or "unknown"


def plan_lines_to_text(plan_lines: List[str]) -> str:
    lines: List[str] = []
    for idx, line in enumerate(plan_lines):
        timestamp = 0.001 + 0.002 * idx
        stripped = line.strip()
        if not stripped.startswith("("):
            stripped = f"({stripped})"
        lines.append(f"{timestamp:.3f}: {stripped}")
    return "\n".join(lines)


def strip_plan_prefix(prefix: str) -> List[str]:
    results: List[str] = []
    for raw in prefix.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        if line.startswith("(") and line.endswith(")"):
            line = line[1:-1].strip()
        results.append(line)
    return results


# ---------------------------------------------------------------------------
# Agent run
# ---------------------------------------------------------------------------
def run_planner(
    domain_text: str,
    problem_text: str,
    tools: List,
    tool_meta: Dict[str, Dict[str, Any]],
    llm,
    validate_path: Path,
    args,
    dataset_label: Optional[str] = None,
):
    def log(msg: str) -> None:
        print(msg, flush=True)

    def agent(messages: List, temp: float, seed: Optional[int]):
        llm.temperature = temp if temp > 0 else 1.0
        llm.do_sample = temp > 0
        return llm.invoke(messages)

    def rebuild_state(prefix_lines: List[str], problem_text: str, feedback: Optional[str]) -> Tuple[List, List[str]]:
        messages: List = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Domain:\n{domain_text}\n\nProblem:\n{problem_text}\n\n"
                f"Existing validated prefix (keep and continue after it):\n{plan_lines_to_text(prefix_lines) if prefix_lines else '(none)'}\n"
                "Emit the next steps via tools. Finish with END when done."
            ),
        ]
        if feedback:
            messages.append(HumanMessage(content=feedback))
        return messages, prefix_lines.copy()

    def attempt_one(
        attempt_idx: int,
        prefix_lines: Optional[List[str]] = None,
        feedback: Optional[str] = None,
        base_seed: Optional[int] = None,
        base_temperature: float = 0.0,
        problem_text_current: Optional[str] = None,
    ):
        prefix_lines = prefix_lines or []
        problem_text_local = problem_text_current or problem_text
        messages, plan_lines = rebuild_state(prefix_lines, problem_text_local, feedback)

        dynamic_temp = base_temperature
        dynamic_seed = base_seed or (int(time.time() * 1000) % 10_000_000)
        temp_schedule = [0, 1, 2, 4, 6]
        temp_idx = 0

        start_time = time.perf_counter()
        error_count = 0
        for step in range(args.max_steps):
            response = agent(messages, dynamic_temp, dynamic_seed)
            messages.append(response)
            tool_calls_raw = list(getattr(response, "tool_calls", []) or [])
            if tool_calls_raw and len(tool_calls_raw) != 1:
                messages.append(
                    HumanMessage(
                        content="Error: you must return exactly ONE tool call per turn. Resend with a single tool call."
                    )
                )
                continue

            tool_call, plan_line = coerce_tool_call(response, tool_meta)
            action = plan_line.split()[0].lower() if plan_line else ""

            # Execute tool if available; otherwise use normalized plan line.
            tool_output = plan_line
            tool_obj = next((t for t in tools if getattr(t, "name", "") == tool_call["name"]), None)
            if tool_obj:
                try:
                    res = tool_obj.invoke(tool_call.get("args", {}))
                    if isinstance(res, ToolMessage):
                        tool_output = str(res.content)
                    else:
                        tool_output = str(res)
                except Exception as exc:
                    tool_output = plan_line
                    log(f"[tool error] {exc}")

            messages.append(
                ToolMessage(
                    content=tool_output,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
            plan_lines.append(tool_output)

            status, stdout, stderr, rc = run_validate(
                domain_text=domain_text,
                problem_text=problem_text_local,
                plan_lines=plan_lines,
                validate_path=validate_path,
                timeout=args.validate_timeout,
            )
            error_category = classify_val_error(status, stdout, stderr)
            if action == "end" and len(plan_lines) == 1 and status != "valid":
                error_category = "goal_not_satisfied"
            if status == "valid":
                val_summary = "ok:passed"
            elif error_category == "goal_not_satisfied":
                val_summary = "ok:goal not reached"
            else:
                val_summary = f"error:{error_category}"

            log(
                f"[attempt {attempt_idx} step {step+1}] temp={dynamic_temp} seed={dynamic_seed} action={plan_line} val={val_summary}"
            )

            if status == "valid":
                break

            if error_category == "goal_not_satisfied":
                if action == "end" and plan_lines:
                    plan_lines.pop()
                    messages.append(
                        HumanMessage(
                            content="Goal not reached; do not call END yet. Continue planning with a single tool call."
                        )
                    )
                dynamic_seed = int(time.time() * 1000) % 10_000_000
                continue

            if plan_lines:
                plan_lines.pop()
            error_count += 1
            err = f"VAL status={status} rc={rc}\nstdout:{stdout}\nstderr:{stderr}"
            # Use progressor to recover last good prefix/problem.
            plan_text_full = plan_lines_to_text(plan_lines)
            prog_result = validate_plan_text(
                domain_text=domain_text,
                problem_text=problem_text_local,
                plan_text=plan_text_full,
                timeout=args.validate_timeout,
                val_bin_dir=validate_path.parent,
            )
            prefix_lines = strip_plan_prefix(prog_result.good_plan_prefix) if prog_result.good_plan_prefix else []
            if prog_result.updated_problem_text:
                problem_text_local = prog_result.updated_problem_text
            messages, plan_lines = rebuild_state(prefix_lines, problem_text_local, err)

            temp_idx = min(temp_idx + 1, len(temp_schedule) - 1) if args.grow_temperature else temp_idx
            dynamic_temp = temp_schedule[temp_idx] if args.grow_temperature else dynamic_temp + 1.0
            dynamic_seed = int(time.time() * 1000) % 10_000_000

            if step + 1 >= args.max_steps or error_count >= args.max_errors:
                log("[warn] reached stopping condition (steps or errors)")
                break

        total_time = time.perf_counter() - start_time

        plan_text = plan_lines_to_text(plan_lines)
        val_result = validate_plan_text(
            domain_text=domain_text,
            problem_text=problem_text_local,
            plan_text=plan_text,
            timeout=args.validate_timeout,
            val_bin_dir=validate_path.parent,
        )

        print("\n=== Run Summary ===")
        if dataset_label:
            print(f"Dataset: {dataset_label}")
        print(f"Steps: {len(plan_lines)}")
        print(f"Plan length (lines): {len(plan_lines)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Validation status: {val_result.status}")
        if val_result.good_plan_prefix:
            print("Good prefix:")
            print(val_result.good_plan_prefix.strip())
        if val_result.updated_problem_text:
            print("Problem updated by ValStep.")

        if args.output:
            record = {
                "dataset": dataset_label,
                "validate_mode": "per-step",
                "steps": len(plan_lines),
                "plan_length": len(plan_lines),
                "plan_lines": plan_lines,
                "plan_text": plan_text,
                "total_time_sec": total_time,
                "validation_status": val_result.status,
                "executed_instructions": val_result.executed_instructions,
                "good_plan_prefix": val_result.good_plan_prefix,
                "updated_problem": val_result.updated_problem_text,
            }
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("a", encoding="utf-8") as fh:
                json.dump(record, fh)
                fh.write("\n")
            log(f"[output] appended record to {out_path}")

        return val_result, plan_lines, val_result.updated_problem_text

    max_attempts = args.max_attempts
    prefix_lines: List[str] = []
    feedback: Optional[str] = None
    current_problem = problem_text
    for attempt in range(1, max_attempts + 1):
        base_temperature = args.temperature
        if args.grow_temperature:
            schedule = [0, 1, 2, 4, 6]
            idx = min(attempt, len(schedule)) - 1
            base_temperature = float(schedule[idx])
        log(f"\n[RUN] {dataset_label or 'problem'} attempt {attempt} temp={base_temperature}")
        val_result, plan_lines, updated_problem_text = attempt_one(
            attempt_idx=attempt,
            prefix_lines=prefix_lines,
            feedback=feedback,
            base_seed=int(time.time() * 1000) % 10_000_000,
            base_temperature=base_temperature,
            problem_text_current=current_problem,
        )
        if val_result.status == "valid":
            break
        prefix_lines = strip_plan_prefix(val_result.good_plan_prefix) if val_result.good_plan_prefix else []
        if updated_problem_text:
            current_problem = updated_problem_text
        feedback = (
            f"Previous plan invalid ({val_result.status}). "
            f"Use the validated prefix of length {len(prefix_lines)} (if any) and continue."
        )
        if not prefix_lines and not updated_problem_text:
            print("[WARN] No progress from validation; stopping retries.")
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_dataset(dataset_path: Path) -> List[Dict[str, str]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list.")
    return data


def extract_domain_name(domain_text: str) -> Optional[str]:
    match = re.search(r"\(domain\s+([^\s)]+)", domain_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-model planner (tool-call style, with progressor).")
    parser.add_argument("--domain-file", required=True, help="Path to PDDL domain file.")
    parser.add_argument("--problem-file", help="Path to PDDL problem file.")
    parser.add_argument("--dataset", help="Optional dataset JSON; uses --index or --scan-all.")
    parser.add_argument("--index", type=int, default=0, help="Index into dataset.")
    parser.add_argument("--scan-all", action="store_true", help="Iterate through dataset from --index onward.")
    parser.add_argument("--tools", help="Path to generated tools module; defaults to generated_tools/<domain>_tools.py")
    parser.add_argument("--model-path", required=True, help="Local HF model path/name.")
    parser.add_argument("--device", default="auto", help="Device map for HF model (e.g., auto, cpu, cuda:0).")
    parser.add_argument("--max-steps", type=int, default=50, help="Max agent/tool iterations.")
    parser.add_argument("--max-errors", type=int, default=5, help="Max validation errors before stopping an attempt.")
    parser.add_argument("--validate-path", help="Path to VAL Validate binary.")
    parser.add_argument("--validate-timeout", type=int, default=30, help="Validate timeout seconds.")
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum validate/replan attempts per problem.")
    parser.add_argument("--grow-temperature", action="store_true", help="Increase temperature per retry (0,1,2,4,6).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Base sampling temperature.")
    parser.add_argument("--output", help="Optional path to append JSONL run results.")
    parser.add_argument("--env-file", default=".env", help="Path to .env file.")
    args = parser.parse_args()

    load_env_file(args.env_file)

    domain_path = Path(args.domain_file)
    if not domain_path.exists():
        raise SystemExit(f"Domain file not found: {domain_path}")
    domain_text = domain_path.read_text(encoding="utf-8")

    examples: Optional[List[Dict[str, str]]] = None
    problem_text_single: Optional[str] = None
    dataset_label_single: Optional[str] = None

    if args.dataset:
        examples = load_dataset(Path(args.dataset))

    if args.problem_file:
        problem_text_single = Path(args.problem_file).read_text(encoding="utf-8")
    elif examples and not args.scan_all:
        if args.index < 0 or args.index >= len(examples):
            raise SystemExit(f"Index {args.index} out of range for dataset length {len(examples)}.")
        problem_text_single = str(examples[args.index].get("input", ""))
        dataset_label_single = f"{Path(args.dataset).name}[{args.index}]"
    elif not args.scan_all:
        raise SystemExit("Provide either --problem-file or --dataset with --index (or use --scan-all).")

    domain_name = extract_domain_name(domain_text) or domain_path.stem.lower()
    if problem_text_single:
        problem_domain = extract_domain_name(problem_text_single)
        if problem_domain and domain_name and problem_domain != domain_name:
            raise SystemExit(
                f"Domain mismatch: domain file='{domain_name}' vs problem domain='{problem_domain}'. "
                "Choose a dataset/problem that matches the domain."
            )
    tools_path = Path(args.tools) if args.tools else Path("generated_tools") / f"{domain_name}_tools.py"
    if not tools_path.exists():
        raise SystemExit(f"Tools module not found: {tools_path}")
    tools_module = load_tools_module(tools_path)
    tools = build_tools(tools_module)
    tool_meta = build_tool_metadata(tools)

    validate_path = Path(args.validate_path) if args.validate_path else DEFAULT_VALIDATE_PATH
    if not validate_path.exists():
        print(f"[WARN] Validate not found at {validate_path}; validation will fail if run.")

    llm = create_local_chat_model(
        model_name=args.model_path,
        device=args.device,
        temperature=args.temperature,
        do_sample=True,
    )
    llm = llm.bind_tools(tools)

    if args.scan_all and examples:
        ds_name = extract_domain_name(domain_text)
        for idx in range(args.index, len(examples)):
            instr = str(examples[idx].get("instruction", ""))
            ex_domain = extract_domain_name(instr or "")
            if ds_name and ex_domain and ex_domain != ds_name:
                continue
            problem_text = str(examples[idx].get("input", ""))
            label = f"{Path(args.dataset).name}[{idx}]"
            run_planner(
                domain_text=domain_text,
                problem_text=problem_text,
                tools=tools,
                tool_meta=tool_meta,
                llm=llm,
                validate_path=validate_path,
                args=args,
                dataset_label=label,
            )
    else:
        run_planner(
            domain_text=domain_text,
            problem_text=problem_text_single,
            tools=tools,
            tool_meta=tool_meta,
            llm=llm,
            validate_path=validate_path,
            args=args,
            dataset_label=dataset_label_single if args.dataset else None,
        )


if __name__ == "__main__":
    main()
