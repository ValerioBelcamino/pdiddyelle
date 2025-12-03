import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You generate tool/function specifications from a PDDL domain file.
Tools correspond to executable plan steps; a call will emit a plan line like "action param1 param2" (or "(action param1 param2)").
For each action in the domain:
- Emit a JSON schema style tool with: name, description (embed key preconditions/effects/constraints in text), and parameters.
- Parameter types must be JSON Schema primitive types: string, number, integer, boolean.
- Keep names concise and aligned with the domain action names.
Return only the structured object requested."""


class ToolParam(BaseModel):
    name: str
    description: str
    type: str = Field(description="JSON schema primitive type.")
    required: bool = True


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: List[ToolParam]


class ToolSpecResponse(BaseModel):
    tools: List[ToolSpec]


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


def normalize_param_type(raw_type: str) -> str:
    """Map loose types to JSON Schema primitives."""
    lowered = raw_type.strip().lower()
    if lowered in {"int", "integer"}:
        return "integer"
    if lowered in {"float", "double", "number"}:
        return "number"
    if lowered in {"bool", "boolean"}:
        return "boolean"
    return "string"


def generate_tool_specs(
    domain_text: str,
    problems_text: Optional[str],
    model: str,
    api_key: str,
    base_url: str,
    timeout: int,
) -> List[ToolSpec]:
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=timeout,
    )

    human_parts = [f"DOMAIN:\n{domain_text}"]
    if problems_text:
        human_parts.append(f"REPRESENTATIVE PROBLEMS:\n{problems_text}")
    base_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(human_parts)),
    ]

    # First try structured output (OpenAI-compatible). If the endpoint rejects response_format,
    # fall back to plain JSON parsing.
    try:
        structured_llm = llm.with_structured_output(ToolSpecResponse)
        result: ToolSpecResponse = structured_llm.invoke(base_messages)
        return result.tools
    except Exception as exc:
        print(f"[WARN] Structured output failed ({exc}); falling back to JSON parsing.")

    json_only_messages = base_messages + [
        HumanMessage(
            content=(
                "Return ONLY compact JSON with shape: "
                '{"tools":[{"name":"","description":"","parameters":[{"name":"","description":"","type":"","required":true}]}]}. '
                "Description must include key preconditions/effects/constraints."
            )
        )
    ]
    raw = llm.invoke(json_only_messages)
    content = getattr(raw, "content", "")
    try:
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean[3:].strip()
            if clean.lower().startswith("json"):
                clean = clean[4:].strip()
            if clean.endswith("```"):
                clean = clean[:-3].strip()
        parsed = json.loads(clean)
        tools_raw = parsed.get("tools", [])
        return [ToolSpec(**tool) for tool in tools_raw]
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON tool specs: {exc}; content was: {content!r}") from exc


def tool_specs_to_functions(tools: List[ToolSpec]) -> List[Dict]:
    """Convert ToolSpec list into OpenAI function-call style tool definitions."""
    functions: List[Dict] = []
    for spec in tools:
        properties: Dict[str, Dict[str, str]] = {}
        required: List[str] = []
        for param in spec.parameters:
            param_type = normalize_param_type(param.type)
            properties[param.name] = {
                "type": param_type,
                "description": param.description,
            }
            if param.required:
                required.append(param.name)

        functions.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )

    # Universal END tool to let the router signal completion.
    functions.append(
        {
            "type": "function",
            "function": {
                "name": "end",
                "description": "Signal the plan is complete and stop emitting further actions.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    return functions


def validate_functions(functions: List[Dict]) -> None:
    """Basic sanity checks on generated tool definitions."""
    names = set()
    for fn in functions:
        name = fn.get("function", {}).get("name")
        if not name:
            raise ValueError("Tool missing name.")
        if name in names:
            raise ValueError(f"Duplicate tool name: {name}")
        names.add(name)
        params = fn.get("function", {}).get("parameters", {})
        if params.get("type") != "object":
            raise ValueError(f"Parameters for {name} must be an object.")
        properties = params.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(f"Properties for {name} must be a dict.")


def load_examples(dataset_path: Path) -> List[Dict[str, str]]:
    """Load a JSON list of objects containing 'instruction' and 'input' keys."""
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list.")
    return data


def filter_examples_by_domain(examples: List[Dict[str, str]], domain_text: str) -> List[Dict[str, str]]:
    """Keep only examples whose domain matches the provided domain text."""
    target = domain_text.strip()
    return [ex for ex in examples if str(ex.get("instruction", "")).strip() == target]


def sample_problems(examples: List[Dict[str, str]], count: int, seed: int = 42) -> List[str]:
    """Sample problems from the dataset."""
    if count <= 0 or not examples:
        return []
    rnd = random.Random(seed)
    chosen = rnd.sample(examples, k=min(count, len(examples)))
    problems: List[str] = []
    for idx, item in enumerate(chosen, start=1):
        problem_text = str(item.get("input", "")).strip()
        problems.append(f"Problem {idx}:\n{problem_text}")
    return problems


def write_tools_module(functions: List[Dict], output_path: Path, domain_name: str) -> None:
    """Persist the generated tools into a Python module."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tools_json = json.dumps(functions, indent=4, ensure_ascii=True)
    content_lines = [
        "# Auto-generated tool definitions for domain: {}".format(domain_name),
        "# Edit with care; regenerate when the domain changes.",
        f'DOMAIN_NAME = "{domain_name}"',
        f"TOOLS = {tools_json}",
        "__all__ = ['DOMAIN_NAME', 'TOOLS']",
    ]
    output_path.write_text("\n".join(content_lines) + "\n", encoding="utf-8")


def infer_domain_name(domain_file: Path) -> str:
    """Derive a stable domain name from the file stem."""
    return domain_file.stem.lower().replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tool schemas from a PDDL domain.")
    parser.add_argument("--domain-file", required=True, help="Path to the PDDL domain file.")
    parser.add_argument("--output", help="Output Python module path for the tools.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (overrides DEEPSEEK_MODEL or .env).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (overrides DEEPSEEK_API_BASE or .env).",
    )
    parser.add_argument(
        "--api-key",
        help="API key (falls back to DEEPSEEK_API_KEY env var).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to a .env file with credentials and config.",
    )
    parser.add_argument(
        "--dataset",
        help="Optional JSON dataset containing domain/problem pairs (e.g., test_500.json).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="Number of problems to sample from the dataset for context.",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY/OPENAI_API_KEY or provide --api-key.")

    model = args.model or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
    base_url = args.base_url or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

    domain_file = Path(args.domain_file)
    if not domain_file.exists():
        raise SystemExit(f"Domain file not found: {domain_file}")

    domain_text = domain_file.read_text(encoding="utf-8")
    domain_name = infer_domain_name(domain_file)
    output_path = Path(args.output) if args.output else Path(f"{domain_name}_tools.py")

    problems_text: Optional[str] = None
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise SystemExit(f"Dataset file not found: {dataset_path}")
        examples = load_examples(dataset_path)
        domain_examples = filter_examples_by_domain(examples, domain_text)
        source_examples = domain_examples if domain_examples else examples
        if not domain_examples:
            print("[WARN] No matching domain entries found in dataset; sampling across all examples.")
        sampled = sample_problems(source_examples, args.sample_count)
        if sampled:
            problems_text = "\n\n".join(sampled)

    print(f"[INFO] Generating tools for domain: {domain_name}")
    tool_specs = generate_tool_specs(
        domain_text=domain_text,
        problems_text=problems_text,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout,
    )
    functions = tool_specs_to_functions(tool_specs)
    validate_functions(functions)
    write_tools_module(functions, output_path, domain_name)
    print(f"[INFO] Wrote tools to {output_path}")


if __name__ == "__main__":
    main()
