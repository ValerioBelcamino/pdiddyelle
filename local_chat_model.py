import json
import re
from typing import Any, Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import inspect
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
import torch
from pydantic import BaseModel

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

@dataclass
class ToolCall:
    """Represents a tool call made by the model."""
    id: str
    name: str
    args: Dict[str, Any]
    type: str = "tool_call"


class ToolCallParser:
    """Parses model responses to extract tool calls with robust handling."""

    @staticmethod
    def extract_tool_calls(text: str) -> List[ToolCall]:
        """Extract tool calls from model response text with improved robustness."""
        tool_calls = []

        # Try multiple extraction strategies in order of preference
        strategies = [
            ToolCallParser._extract_from_complete_json_blocks,
            ToolCallParser._extract_from_incomplete_json_blocks,
            ToolCallParser._extract_from_json_anywhere,
            ToolCallParser._extract_from_function_calls
        ]

        for strategy in strategies:
            try:
                calls = strategy(text)
                if calls:
                    tool_calls.extend(calls)
                    # If we found calls with a strategy, we can break or continue to find more
                    # Continue to catch cases where multiple formats exist
            except Exception as e:
                continue

        # Remove duplicates based on name and args
        unique_calls = []
        seen = set()
        for call in tool_calls:
            signature = (call.name, str(call.args))
            if signature not in seen:
                seen.add(signature)
                unique_calls.append(call)

        return unique_calls

    @staticmethod
    def _extract_from_complete_json_blocks(text: str) -> List[ToolCall]:
        """Extract from properly formatted ```json...``` blocks."""
        tool_calls = []
        pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            calls = ToolCallParser._parse_json_content(match)
            tool_calls.extend(calls)

        return tool_calls

    @staticmethod
    def _extract_from_incomplete_json_blocks(text: str) -> List[ToolCall]:
        """Extract from incomplete JSON blocks (```json... without closing)."""
        tool_calls = []

        # Look for ```json or ```JSON followed by content
        pattern = r'```(?:json|JSON)\s*(.*?)(?=```|\Z)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            # Try to find where JSON likely ends
            json_content = ToolCallParser._extract_json_from_text(match)
            if json_content:
                calls = ToolCallParser._parse_json_content(json_content)
                tool_calls.extend(calls)

        return tool_calls

    @staticmethod
    def _extract_from_json_anywhere(text: str) -> List[ToolCall]:
        """Extract JSON that looks like tool calls from anywhere in text."""
        tool_calls = []

        # Look for JSON-like structures that contain "tool_calls" or tool patterns
        # This handles cases where there are no markdown code blocks at all
        json_patterns = [
            r'\{[^{}]*"tool_calls"[^{}]*\[[^\]]*\][^{}]*\}',  # Simple tool_calls pattern
            r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}',        # Single tool call pattern
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                # Try to expand the match to get complete JSON
                expanded = ToolCallParser._expand_json_match(text, match)
                calls = ToolCallParser._parse_json_content(expanded)
                tool_calls.extend(calls)

        return tool_calls

    @staticmethod
    def _extract_from_function_calls(text: str) -> List[ToolCall]:
        """Extract from direct function call format like multiply(2, 6)."""
        tool_calls = []

        # Look for function call patterns
        func_pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(func_pattern, text)

        for func_name, args_str in matches:
            # Skip common English words that might match the pattern
            if func_name.lower() in ['to', 'in', 'on', 'at', 'by', 'for', 'with', 'from']:
                continue

            args = ToolCallParser._parse_function_args(args_str)
            tool_calls.append(ToolCall(
                id=f"call_{hash(func_name + args_str)%10000:04d}",
                name=func_name,
                args=args
            ))

        return tool_calls

    @staticmethod
    def _parse_json_content(content: str) -> List[ToolCall]:
        """Parse JSON content and extract tool calls."""
        tool_calls = []

        try:
            parsed = json.loads(content)

            if isinstance(parsed, dict):
                if "tool_calls" in parsed:
                    # Multiple tool calls format
                    for call in parsed["tool_calls"]:
                        if isinstance(call, dict) and "name" in call:
                            tool_calls.append(ToolCall(
                                id=f"call_{hash(str(call))%10000:04d}",
                                name=call["name"],
                                args=call.get("arguments", call.get("args", {}))
                            ))
                elif "name" in parsed:
                    # Single tool call format
                    tool_calls.append(ToolCall(
                        id=f"call_{hash(str(parsed))%10000:04d}",
                        name=parsed["name"],
                        args=parsed.get("arguments", parsed.get("args", {}))
                    ))

        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed_content = ToolCallParser._fix_json_issues(content)
            if fixed_content != content:
                return ToolCallParser._parse_json_content(fixed_content)

        return tool_calls

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[str]:
        """Extract the most likely JSON content from text."""
        # Find the first { and try to find matching }
        start = text.find('{')
        if start == -1:
            return None

        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]

        # If we can't find matching braces, try to take until end of reasonable JSON
        # Look for common JSON ending patterns
        json_text = text[start:]
        for end_pattern in ['\n\n', '```', '\n}', '}']:
            if end_pattern in json_text:
                potential_end = json_text.find(end_pattern)
                if end_pattern == '}':
                    potential_end += 1
                potential_json = json_text[:potential_end]
                if potential_json.count('{') <= potential_json.count('}'):
                    return potential_json

        return json_text

    @staticmethod
    def _expand_json_match(text: str, match: str) -> str:
        """Expand a partial JSON match to try to get complete JSON."""
        start_pos = text.find(match)
        if start_pos == -1:
            return match

        # Look backwards for opening brace
        actual_start = start_pos
        for i in range(start_pos - 1, -1, -1):
            if text[i] == '{':
                actual_start = i
                break
            elif text[i] == '}':
                break

        # Look forwards for closing brace
        actual_end = start_pos + len(match)
        brace_count = 0
        for i, char in enumerate(text[actual_start:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    actual_end = actual_start + i + 1
                    break

        return text[actual_start:actual_end]

    @staticmethod
    def _fix_json_issues(content: str) -> str:
        """Try to fix common JSON formatting issues."""
        fixed = content.strip()

        # Remove trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)

        # Add missing closing braces/brackets if obvious
        open_braces = fixed.count('{') - fixed.count('}')
        open_brackets = fixed.count('[') - fixed.count(']')

        if open_braces > 0:
            fixed += '}' * open_braces
        if open_brackets > 0:
            fixed += ']' * open_brackets

        # Fix unquoted keys (simple cases)
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)

        return fixed

    @staticmethod
    def _parse_function_args(args_str: str) -> Dict[str, Any]:
        """Parse function arguments from string format."""
        args = {}
        if not args_str.strip():
            return args

        try:
            # Try to handle as JSON-like object
            if args_str.strip().startswith('{'):
                return json.loads(args_str)
        except:
            pass

        # Handle positional and named arguments
        if '=' not in args_str:
            # Positional arguments
            values = [v.strip().strip('"\'') for v in args_str.split(',')]
            for i, value in enumerate(values):
                try:
                    # Try to convert to appropriate type
                    if value.isdigit():
                        args[f'arg_{i}'] = int(value)
                    elif value.replace('.', '').isdigit():
                        args[f'arg_{i}'] = float(value)
                    else:
                        args[f'arg_{i}'] = value
                except:
                    args[f'arg_{i}'] = value
        else:
            # Named arguments
            for arg in args_str.split(','):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    try:
                        if value.isdigit():
                            args[key] = int(value)
                        elif value.replace('.', '').isdigit():
                            args[key] = float(value)
                        elif value.lower() in ['true', 'false']:
                            args[key] = value.lower() == 'true'
                        else:
                            args[key] = value
                    except:
                        args[key] = value

        return args

class DynamicPromptBuilder:
    """Dynamically builds prompts using tokenizer's chat template."""

    @staticmethod
    def supports_system_prompt(tokenizer: PreTrainedTokenizer) -> bool:
        """Check if the tokenizer's chat template supports system prompts."""
        try:
            # Test with a simple system message
            test_messages = [
                {"role": "system", "content": "Test system message"},
                {"role": "user", "content": "Test user message"}
            ]

            # Try to apply the chat template
            prompt = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Check if system content appears in the prompt
            # If it's properly handled, the system message should be in the prompt
            return "Test system message" in prompt

        except Exception as e:
            # If there's an error (like role alternation error), system is not supported
            return False

    @staticmethod
    def create_tool_instruction(tools: List[Dict]) -> str:
        """Create tool instruction text from tool definitions."""
        if not tools:
            return ""

        tool_descriptions = []
        for tool in tools:
            func_info = tool.get('function', {})
            name = func_info.get('name', '')
            description = func_info.get('description', '')
            parameters = func_info.get('parameters', {}).get('properties', {})

            param_desc = []
            for param_name, param_info in parameters.items():
                param_type = param_info.get('type', 'any')
                param_description = param_info.get('description', '')
                param_desc.append(f"  - {param_name} ({param_type}): {param_description}")

            tool_desc = f"**{name}**: {description}"
            if param_desc:
                tool_desc += f"\n  Parameters:\n" + "\n".join(param_desc)

            tool_descriptions.append(tool_desc)

        return f"""You have access to the following tools:

{chr(10).join(tool_descriptions)}

When you need to use a tool, respond with a JSON code block in this exact format:
```json
{{
    "tool_calls": [
        {{
            "name": "tool_name",
            "arguments": {{"param1": "value1", "param2": "value2"}}
        }}
    ]
}}
```

If you don't need to use any tools, respond normally without the JSON block."""

    @staticmethod
    def build_prompt(
        tokenizer: PreTrainedTokenizer,
        tools: List[Dict],
        messages: List[BaseMessage]
    ) -> str:
        """Build prompt using tokenizer's chat template dynamically."""

        # Convert LangChain messages to chat format
        chat_messages = []
        tool_instruction = DynamicPromptBuilder.create_tool_instruction(tools)

        # Check if system prompts are supported
        supports_system = DynamicPromptBuilder.supports_system_prompt(tokenizer)

        system_content = ""
        first_user_message = True

        for message in messages:
            if isinstance(message, SystemMessage):
                system_content = message.content
            elif isinstance(message, HumanMessage):
                if supports_system:
                    # Add system message separately if supported
                    if system_content and not any(msg.get("role") == "system" for msg in chat_messages):
                        if tool_instruction:
                            full_system_content = system_content + "\n\n" + tool_instruction
                        else:
                            full_system_content = system_content
                        chat_messages.append({
                            "role": "system",
                            "content": full_system_content
                        })
                    elif tool_instruction and not any(msg.get("role") == "system" for msg in chat_messages):
                        # Add tool instruction as system message
                        chat_messages.append({
                            "role": "system",
                            "content": tool_instruction
                        })

                    chat_messages.append({
                        "role": "user",
                        "content": message.content
                    })
                else:
                    # For models without system support, prepend to first user message
                    if first_user_message:
                        combined_content = ""
                        if system_content:
                            combined_content += system_content + "\n\n"
                        if tool_instruction:
                            combined_content += tool_instruction + "\n\n"
                        combined_content += message.content

                        chat_messages.append({
                            "role": "user",
                            "content": combined_content
                        })
                        first_user_message = False
                    else:
                        chat_messages.append({
                            "role": "user",
                            "content": message.content
                        })

            elif isinstance(message, AIMessage):
                chat_messages.append({
                    "role": "assistant",
                    "content": message.content
                })

        # Apply the tokenizer's chat template
        try:
            prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt

        except Exception as e:
            print(f"Error applying chat template: {e}")
            print(f"Model: {getattr(tokenizer, 'name_or_path', 'unknown')}")
            print(f"Supports system: {supports_system}")

            # Fallback to simple concatenation
            fallback_prompt = ""
            for msg in chat_messages:
                role = msg["role"].title()
                content = msg["content"]
                fallback_prompt += f"{role}: {content}\n\n"
            fallback_prompt += "Assistant:"
            return fallback_prompt


class LocalChatModel(BaseChatModel):
    """Local chat model that supports tool calling through LangChain interface."""

    # Declare all Pydantic fields
    model_name: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    bound_tools: List[Dict] = []
    device: str = "auto"
    torch_dtype_str: str = "float16"  # Store as string, convert when needed
    model_kwargs: Dict[str, Any] = {}  # Store additional model loading kwargs

    # Use Any type for complex objects and exclude from serialization
    model: Any = None
    tokenizer: Any = None

    class Config:
        arbitrary_types_allowed = True
        # Don't try to serialize these complex objects
        exclude = {"model", "tokenizer"}

    def __init__(
        self,
        model_name: str,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: str = "auto",
        torch_dtype = torch.float16,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        bound_tools: Optional[List[Dict]] = None,
        quantization_config = None,  # BitsAndBytesConfig or other quantization
        attn_implementation: Optional[str] = None,  # e.g., "flash_attention_2"
        **model_kwargs  # Additional model loading arguments
    ):
        # Convert torch_dtype to string for Pydantic
        torch_dtype_str = str(torch_dtype).split('.')[-1] if torch_dtype else "float16"

        # Prepare model_kwargs for storage
        stored_model_kwargs = model_kwargs.copy()
        if quantization_config is not None:
            stored_model_kwargs['quantization_config'] = 'provided'  # Placeholder
        if attn_implementation is not None:
            stored_model_kwargs['attn_implementation'] = attn_implementation

        # Initialize with Pydantic fields
        super().__init__(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            bound_tools=bound_tools or [],
            device=device,
            torch_dtype_str=torch_dtype_str,
            model_kwargs=stored_model_kwargs,
            model=None,  # Will be set below
            tokenizer=None,  # Will be set below
        )

        # Load model and tokenizer if not provided
        if model is None or tokenizer is None:
            print(f"Loading model: {model_name}")

            # Convert string back to torch dtype
            if torch_dtype_str == "float16":
                actual_dtype = torch.float16
            elif torch_dtype_str == "bfloat16":
                actual_dtype = torch.bfloat16
            elif torch_dtype_str == "float32":
                actual_dtype = torch.float32
            else:
                actual_dtype = torch.float16

            # Prepare model loading arguments
            model_load_kwargs = {
                'torch_dtype': actual_dtype,
                'device_map': device,
                'trust_remote_code': True,
                **model_kwargs
            }

            # Add quantization config if provided
            if quantization_config is not None:
                model_load_kwargs['quantization_config'] = quantization_config
                print(f"Using quantization config: {type(quantization_config).__name__}")

            # Add attention implementation if provided
            if attn_implementation is not None:
                model_load_kwargs['attn_implementation'] = attn_implementation
                print(f"Using attention implementation: {attn_implementation}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load model with all configurations
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_load_kwargs
            )

            # Store the actual quantization config for bind_tools
            self._quantization_config = quantization_config
            self._attn_implementation = attn_implementation
            self._original_model_kwargs = model_kwargs

        else:
            self.model = model
            self.tokenizer = tokenizer
            self._quantization_config = quantization_config
            self._attn_implementation = attn_implementation
            self._original_model_kwargs = model_kwargs

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check and log system prompt support
        supports_system = DynamicPromptBuilder.supports_system_prompt(self.tokenizer)
        print(f"Model '{model_name}' system prompt support: {supports_system}")

    def bind_tools(self, tools: List[Union[Dict, BaseTool, Callable, BaseModel]]) -> "LocalChatModel":
        """Bind tools to the model for function calling."""
        converted_tools = []

        for tool in tools:
            if isinstance(tool, dict):
                # Already in OpenAI format
                converted_tools.append(tool)
            elif hasattr(tool, '__annotations__') and callable(tool):
                # Function with type annotations
                converted_tools.append(convert_to_openai_tool(tool))
            elif isinstance(tool, type) and issubclass(tool, BaseModel):
                # Pydantic model
                converted_tools.append(self._pydantic_to_openai_tool(tool))
            elif isinstance(tool, BaseTool):
                # LangChain tool
                converted_tools.append(convert_to_openai_tool(tool))
            else:
                # Try to convert anyway
                try:
                    converted_tools.append(convert_to_openai_tool(tool))
                except:
                    print(f"Warning: Could not convert tool {tool}")

        # Create new instance with bound tools, reusing existing model/tokenizer
        new_instance = self.__class__(
            model_name=self.model_name,
            model=self.model,  # Reuse loaded model
            tokenizer=self.tokenizer,  # Reuse loaded tokenizer
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            device=self.device,
            bound_tools=converted_tools,
            quantization_config=getattr(self, '_quantization_config', None),
            attn_implementation=getattr(self, '_attn_implementation', None),
            **getattr(self, '_original_model_kwargs', {})
        )
        return new_instance

    def _pydantic_to_openai_tool(self, pydantic_model: BaseModel) -> Dict:
        """Convert Pydantic model to OpenAI tool format."""
        schema = pydantic_model.model_json_schema()

        return {
            "type": "function",
            "function": {
                "name": pydantic_model.__name__,
                "description": pydantic_model.__doc__ or schema.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", [])
                }
            }
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from the model."""

        # Build prompt using dynamic approach
        prompt = DynamicPromptBuilder.build_prompt(
            self.tokenizer,
            self.bound_tools,
            messages
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4000  # Adjust based on model's context length
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Parse tool calls
        tool_calls = ToolCallParser.extract_tool_calls(response)

        # Create AI message
        if tool_calls:
            # Convert to LangChain format
            langchain_tool_calls = []
            for call in tool_calls:
                langchain_tool_calls.append({
                    "name": call.name,
                    "args": call.args,
                    "id": call.id,
                    "type": call.type
                })

            ai_message = AIMessage(
                content=response,
                tool_calls=langchain_tool_calls
            )
        else:
            ai_message = AIMessage(content=response)

        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    @property
    def _llm_type(self) -> str:
        return "local_chat_model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


# Convenience function to create a local chat model
def create_local_chat_model(
    model_name: str,
    device: str = "auto",
    quantization_config = None,
    attn_implementation: Optional[str] = None,
    **kwargs
) -> LocalChatModel:
    """Create a local chat model with tool calling support."""
    return LocalChatModel(
        model_name=model_name,
        device=device,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        **kwargs
    )