"""
vLLM-friendly LangGraph planner that keeps dataset iteration, VAL validation,
and retry logic while being tolerant to non-JSON tool calling.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

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
- Call one tool per step to build the plan in order.
- Use the tool names exactly as given. Each tool call should correspond to a single plan line: "action param1 param2".
- Use the END tool to finish when goals are achieved.
- If JSON/tool calling is unavailable, reply with exactly one line: either "action param1 param2" or "end", no explanations."""


class PlannerState(TypedDict):
    messages: Annotated[List, add_messages]
    plan_lines: List[str]
    last_error: Optional[str]
    last_status: Optional[str]
    last_action: Optional[str]
    steps: int
    fail_streak: int
    first_try_success: bool


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
    """Create a lookup of tool names to required parameter order."""
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
    """Parse a free-form line like `move a b` into a tool call dict."""
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
        if not name and isinstance(obj.get("function"), dict):
            name = obj["function"].get("name")
            obj_args = obj["function"].get("arguments") or obj["function"].get("args", {})
        else:
            obj_args = obj.get("arguments") or obj.get("args", {})
        if not name:
            return None
        meta = tool_meta.get(str(name).lower())
        if not meta:
            return None
        args: Dict[str, str] = {}
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
        line = re.sub(r"</?[^>]+>", "", line)  # strip simple XML-ish tags
        return line

    # Prefer fenced code blocks if present
    fenced_blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, flags=re.DOTALL)
    for block in fenced_blocks:
        block = block.strip()
        if not block:
            continue
        json_candidate = block
        parsed_json = from_json_block(json_candidate)
        if parsed_json:
            return parsed_json
        for raw_line in block.splitlines():
            candidate = clean_line(raw_line)
            parsed = parse_tokens(candidate.split())
            if parsed:
                return parsed

    # XML-like blocks
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


def coerce_tool_call_message(
    response: AIMessage, tool_meta: Dict[str, Dict[str, Any]], allow_fallback: bool = True
) -> AIMessage:
    """Ensure the LLM response always contains a usable tool call."""
    if not isinstance(response, AIMessage):
        raise TypeError(f"Expected AIMessage from model, got {type(response)}")

    existing_calls = list(getattr(response, "tool_calls", []) or [])
    for call in existing_calls:
        name = call.get("name") or call.get("function", {}).get("name")
        if not name:
            continue
        meta = tool_meta.get(name.lower())
        if not meta:
            continue
        raw_args = call.get("args")
        if raw_args is None and isinstance(call.get("function"), dict):
            raw_args = call["function"].get("arguments")
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except Exception:
                raw_args = raw_args.strip()
        if isinstance(raw_args, dict):
            args = {param: str(raw_args.get(param, "")) for param in meta.get("required", [])}
        elif isinstance(raw_args, str):
            args = _tokens_to_args(meta.get("required", []), raw_args.split())
        else:
            args = _tokens_to_args(meta.get("required", []), [])
        response.tool_calls = [
            {
                "name": meta["name"],
                "args": args,
                "id": call.get("id") or f"call_{uuid4().hex}",
                "type": call.get("type") or "tool_call",
            }
        ]
        response.content = ""
        return response

    parsed = parse_tool_call_text(str(response.content or ""), tool_meta) if allow_fallback else None
    if not parsed:
        end_meta = tool_meta.get("end")
        if end_meta:
            parsed = {"name": end_meta["name"], "args": {}}
        else:
            raise ValueError("Model did not return a tool call and no tool could be parsed from text.")

    response.tool_calls = [
        {
            "name": parsed["name"],
            "args": parsed.get("args", {}),
            "id": f"call_{uuid4().hex}",
            "type": "tool_call",
        }
    ]
    response.content = ""
    return response


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
    """Map VAL status/output to coarse error categories."""
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
    """Convert plan lines (action arg1 arg2) into timestamped VAL plan text."""
    lines: List[str] = []
    for idx, line in enumerate(plan_lines):
        timestamp = 0.001 + 0.002 * idx
        stripped = line.strip()
        if not stripped.startswith("("):
            stripped = f"({stripped})"
        lines.append(f"{timestamp:.3f}: {stripped}")
    return "\n".join(lines)


def strip_plan_prefix(prefix: str) -> List[str]:
    """Convert a timestamped prefix into plain plan lines."""
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


def build_graph(
    tools: List,
    llm: ChatOpenAI,
    validate_path: Path,
    domain_text: str,
    problem_text: str,
    validate_timeout: int,
    max_steps: int,
    validate_mode: str = "per-step",
    max_failures_per_step: int = 3,
    tool_calling_mode: str = "auto",
    tool_meta: Optional[Dict[str, Dict[str, Any]]] = None,
):
    tool_meta = tool_meta or build_tool_metadata(tools)
    tool_node = ToolNode(tools)

    def agent(state: PlannerState):
        messages = state["messages"]
        response = None
        if tool_calling_mode in {"auto", "json"}:
            try:
                response = llm.bind_tools(tools, tool_choice="required").invoke(messages)
            except Exception:
                if tool_calling_mode == "json":
                    raise
        if response is None:
            response = llm.invoke(messages)
        coerced = coerce_tool_call_message(response, tool_meta, allow_fallback=True)
        return {"messages": [coerced], "steps": state["steps"] + 1}

    def accumulate(state: PlannerState):
        plan_lines = list(state["plan_lines"])
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        new_tool_msgs = tool_msgs[len(plan_lines):]
        last_action = state.get("last_action")
        for msg in new_tool_msgs:
            content = str(msg.content).strip()
            if not content:
                continue
            action = content.split()[0].lower() if content else ""
            last_action = action
            if action != "end":
                plan_lines.append(content)
        return {"plan_lines": plan_lines, "last_action": last_action}

    def validate(state: PlannerState):
        if validate_mode == "final" and state.get("last_action") != "end":
            return {
                "last_error": None,
                "last_status": "pending",
                "fail_streak": state.get("fail_streak", 0),
                "first_try_success": False,
                "error_category": None,
            }
        if state.get("last_action") == "end":
            return {
                "last_error": None,
                "last_status": "pending" if validate_mode == "final" else "skipped",
                "fail_streak": state.get("fail_streak", 0),
                "first_try_success": False,
                "error_category": None,
            }

        status, stdout, stderr, rc = run_validate(
            domain_text=domain_text,
            problem_text=problem_text,
            plan_lines=state["plan_lines"],
            validate_path=validate_path,
            timeout=validate_timeout,
        )
        prev_streak = state.get("fail_streak", 0)
        if status == "valid":
            first_try = prev_streak == 0
            return {
                "last_error": None,
                "last_status": status,
                "fail_streak": 0,
                "first_try_success": first_try,
                "error_category": None,
            }

        streak = prev_streak + 1
        error_category = classify_val_error(status, stdout, stderr)
        trimmed_err = f"VAL status={status} rc={rc}\nstdout:{stdout}\nstderr:{stderr}"
        return {
            "messages": [HumanMessage(content=trimmed_err)],
            "plan_lines": state["plan_lines"],
            "last_error": trimmed_err,
            "last_status": status,
            "fail_streak": streak,
            "first_try_success": False,
            "error_category": error_category,
        }

    def should_continue(state: PlannerState) -> str:
        if state.get("last_status") == "valid":
            return END
        if state.get("fail_streak", 0) >= max_failures_per_step:
            return END
        if state.get("last_action") == "end":
            return END
        if state["steps"] >= max_steps:
            return END
        return "agent"

    workflow = StateGraph(PlannerState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("accumulate", accumulate)
    workflow.add_node("validate", validate)

    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "accumulate")
    workflow.add_edge("accumulate", "validate")
    workflow.add_conditional_edges("validate", should_continue, {END: END, "agent": "agent"})

    workflow.set_entry_point("agent")
    return workflow.compile()


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
    parser = argparse.ArgumentParser(description="LangGraph tool-calling planner with VAL validation (vLLM-friendly).")
    parser.add_argument("--domain-file", required=True, help="Path to PDDL domain file.")
    parser.add_argument("--problem-file", help="Path to PDDL problem file.")
    parser.add_argument("--dataset", help="Optional dataset (instruction/input/output) JSON; uses --index to select problem.")
    parser.add_argument("--index", type=int, default=0, help="Index into dataset.")
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="If set, iterate through dataset from --index onward, skipping non-matching domains.",
    )
    parser.add_argument("--tools", help="Path to generated tools module; defaults to generated_tools/<domain>_tools.py")
    parser.add_argument("--model", default=None, help="Model name (defaults to VLLM_MODEL or 'local-vllm').")
    parser.add_argument("--base-url", default=None, help="API base URL (defaults to VLLM_API_BASE or http://localhost:8000/v1).")
    parser.add_argument("--api-key", help="API key (defaults to VLLM_API_KEY or 'not-needed').")
    parser.add_argument("--validate-path", help="Path to VAL Validate binary.")
    parser.add_argument("--validate-timeout", type=int, default=30, help="Validate timeout seconds.")
    parser.add_argument("--max-steps", type=int, default=50, help="Max agent/tool iterations.")
    parser.add_argument(
        "--validate-mode",
        choices=["per-step", "final"],
        default="final",
        help="Validation strategy: per-step or only once at the end (default).",
    )
    parser.add_argument(
        "--max-failures-per-step",
        type=int,
        default=3,
        help="Stop if the same plan length fails validation this many times in a row (per-step mode).",
    )
    parser.add_argument(
        "--final-validate",
        action="store_true",
        help="Alias for --validate-mode final.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=200,
        help="LangGraph recursion limit (iterations) before aborting.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, stream graph events; otherwise only summary is printed.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to append JSONL run results (one JSON object per problem).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum validate/replan attempts per problem.",
    )
    parser.add_argument(
        "--grow-temperature",
        action="store_true",
        help="If set, increase temperature per retry (0,1,2,4,6).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--tool-calling",
        choices=["auto", "json", "text"],
        default="auto",
        help="Tool calling strategy: json forces OpenAI/tool schema, text expects plain 'action arg1 arg2' for local models, auto tries json then falls back.",
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file.")
    args = parser.parse_args()

    load_env_file(args.env_file)

    api_key = args.api_key or os.getenv("VLLM_API_KEY") or "not-needed"
    model = args.model or os.getenv("VLLM_MODEL") or "local-vllm"
    base_url = args.base_url or os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")

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
    tools_path = Path(args.tools) if args.tools else Path("generated_tools") / f"{domain_name}_tools.py"
    if not tools_path.exists():
        raise SystemExit(f"Tools module not found: {tools_path}")
    tools_module = load_tools_module(tools_path)
    tools = build_tools(tools_module)
    tool_meta = build_tool_metadata(tools)

    validate_path = Path(args.validate_path) if args.validate_path else DEFAULT_VALIDATE_PATH
    if not validate_path.exists():
        print(f"[WARN] Validate not found at {validate_path}; validation will fail if run.")

    def make_llm(seed: Optional[int] = None, temperature: float = 0.0) -> ChatOpenAI:
        mk: Dict[str, object] = {}
        if seed is not None:
            mk["seed"] = seed
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            timeout=60,
            model_kwargs=mk if mk else None,
        )

    validate_mode = "final" if args.final_validate else args.validate_mode
    recursion_limit = args.recursion_limit
    tool_calling_mode = args.tool_calling

    output_path: Optional[Path] = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def run_problem(
        problem_text: str,
        dataset_label_local: Optional[str],
        prefix_lines: Optional[List[str]] = None,
        feedback: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: float = 0.0,
    ):
        llm = make_llm(seed, temperature=temperature)
        graph = build_graph(
            tools=tools,
            llm=llm,
            validate_path=validate_path,
            domain_text=domain_text,
            problem_text=problem_text,
            validate_timeout=args.validate_timeout,
            max_steps=args.max_steps,
            validate_mode=validate_mode,
            max_failures_per_step=args.max_failures_per_step,
            tool_calling_mode=tool_calling_mode,
            tool_meta=tool_meta,
        )

        prefix_lines = prefix_lines or []
        initial_messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Domain:\n{domain_text}\n\nProblem:\n{problem_text}\n\n"
                f"Existing validated prefix (keep and continue after it):\n{plan_lines_to_text(prefix_lines) if prefix_lines else '(none)'}\n"
                "Emit the next steps via tools. Finish with END when done."
            ),
        ]
        if feedback:
            initial_messages.append(HumanMessage(content=f"Validator feedback:\n{feedback}"))
        state: PlannerState = {
            "messages": initial_messages,
            "plan_lines": prefix_lines.copy(),
            "last_error": None,
            "last_status": None,
            "last_action": None,
            "steps": 0,
            "fail_streak": 0,
            "first_try_success": False,
        }

        start_time = time.perf_counter()
        current_plan_lines: List[str] = []

        for event in graph.stream(state, config={"recursion_limit": recursion_limit}):
            if args.verbose:
                print(event)
            if "accumulate" in event and "plan_lines" in event["accumulate"]:
                current_plan_lines = event["accumulate"]["plan_lines"]

        total_time = time.perf_counter() - start_time

        plan_text = plan_lines_to_text(current_plan_lines)
        val_result = validate_plan_text(
            domain_text=domain_text,
            problem_text=problem_text,
            plan_text=plan_text,
            timeout=args.validate_timeout,
            val_bin_dir=validate_path.parent,
        )

        print("\n=== Run Summary ===")
        print(f"Domain: {domain_name}")
        if dataset_label_local:
            print(f"Dataset: {dataset_label_local}")
        print(f"Steps: {len(current_plan_lines)}")
        print(f"Plan length (lines): {len(current_plan_lines)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Validation status: {val_result.status}")
        if val_result.good_plan_prefix:
            print("Good prefix:")
            print(val_result.good_plan_prefix.strip())
        if val_result.updated_problem_text:
            print("Problem updated by ValStep.")

        if output_path:
            record = {
                "domain": domain_name,
                "dataset": dataset_label_local,
                "validate_mode": validate_mode,
                "steps": len(current_plan_lines),
                "plan_length": len(current_plan_lines),
                "plan_lines": current_plan_lines,
                "plan_text": plan_text,
                "total_time_sec": total_time,
                "validation_status": val_result.status,
                "executed_instructions": val_result.executed_instructions,
                "good_plan_prefix": val_result.good_plan_prefix,
                "updated_problem": val_result.updated_problem_text,
            }
            with output_path.open("a", encoding="utf-8") as fh:
                json.dump(record, fh)
                fh.write("\n")

        return val_result, current_plan_lines

    max_attempts = args.max_attempts

    def run_with_retries(problem_text: str, label: Optional[str]) -> None:
        current_problem = problem_text
        prefix_lines: List[str] = []
        feedback: Optional[str] = None
        for attempt in range(1, max_attempts + 1):
            seed = int(time.time() * 1000) % 10_000_000
            temperature = args.temperature
            if args.grow_temperature:
                temp_schedule = [0, 1, 2, 4, 6]
                idx = min(attempt, len(temp_schedule)) - 1
                temperature = float(temp_schedule[idx])
            print(f"\n[RUN] {label or 'problem'} attempt {attempt}")
            val_result, plan_lines = run_problem(
                current_problem, label, prefix_lines, feedback, seed, temperature
            )
            if val_result.status == "valid":
                break
            prefix_lines = strip_plan_prefix(val_result.good_plan_prefix) if val_result.good_plan_prefix else []
            if val_result.updated_problem_text:
                current_problem = val_result.updated_problem_text
            feedback = (
                f"Previous plan invalid ({val_result.status}). "
                f"Use the validated prefix of length {len(prefix_lines)} (if any) and continue."
            )
            if not prefix_lines and not val_result.updated_problem_text:
                print("[WARN] No progress from validation; stopping retries.")
                break

    if args.scan_all and examples:
        ds_name = extract_domain_name(domain_text)
        for idx in range(args.index, len(examples)):
            instr = str(examples[idx].get("instruction", ""))
            ex_domain = extract_domain_name(instr or "")
            if ds_name and ex_domain and ex_domain != ds_name:
                continue
            problem_text = str(examples[idx].get("input", ""))
            label = f"{Path(args.dataset).name}[{idx}]"
            run_with_retries(problem_text, label)
    else:
        run_with_retries(problem_text_single, dataset_label_single if args.dataset else None)


if __name__ == "__main__":
    main()
