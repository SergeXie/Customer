"""Adapt the audited ``@agent_tool`` registry to LangChain tools.

Authority fields (channel_type / shop_id / user_id / recipient_uid) always
come from the trusted per-request dependencies dict, never from the model —
the same contract the previous ``execute_tool`` enforced.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from langchain_core.tools import StructuredTool

from Agent.CustomerAgent.custom.tool_decorator import (
    TOOL_REGISTRY,
    ToolEntry,
    get_tool_entry,
)
from utils.logger_loguru import get_logger

logger = get_logger("LCTools")

# Identity / permission fields that must never be taken from model output.
AUTHORITY_FIELDS = frozenset(
    {
        "channel_type",
        "shop_id",
        "user_id",
        "recipient_uid",
    }
)


def build_tool_params(
    entry: ToolEntry,
    arguments: Any,
    dependencies: Dict[str, Any],
) -> Any:
    """Merge LLM arguments with trusted dependencies and validate.

    Returns the validated Pydantic instance, or an error string when a
    required authority field is missing from the trusted session.
    """
    if isinstance(arguments, str):
        try:
            args_dict: Dict[str, Any] = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return "[工具参数格式错误，请重试]"
    elif isinstance(arguments, dict):
        args_dict = arguments
    else:
        args_dict = {}

    if not isinstance(args_dict, dict):
        return "[工具参数格式错误，请重试]"

    params: Dict[str, Any] = {}
    for field_name in entry.param_model.model_fields:
        if field_name in AUTHORITY_FIELDS:
            trusted_value = dependencies.get(field_name)
            if trusted_value is None or str(trusted_value).strip() == "":
                return "[工具无法执行：缺少可信会话身份]"
            params[field_name] = trusted_value
        elif field_name in args_dict:
            params[field_name] = args_dict[field_name]
        elif field_name in dependencies:
            params[field_name] = dependencies[field_name]

    return entry.param_model(**params)


def execute_registry_tool(
    name: str,
    arguments: Any,
    dependencies: Dict[str, Any],
) -> str:
    """Run one registry tool with authority injection and safe errors."""
    entry = get_tool_entry(name)
    if entry is None:
        return f"[工具不存在: {name}]"

    try:
        validated = build_tool_params(entry, arguments, dependencies)
        if isinstance(validated, str):
            return validated
        result = entry.func(validated)
        if result is None:
            return "[工具执行无返回]"
        return str(result)
    except json.JSONDecodeError:
        logger.error(f"工具 {name} 参数解析失败: JSONDecodeError")
        return "[工具参数格式错误，请重试]"
    except Exception as exc:
        logger.error(f"工具 {name} 执行异常: {type(exc).__name__}")
        return "[工具执行失败，请稍后重试]"


def _make_runner(entry: ToolEntry, dependencies: Dict[str, Any]) -> Callable[..., str]:
    def _run(**kwargs: Any) -> str:
        return execute_registry_tool(entry.name, kwargs, dependencies)

    _run.__name__ = entry.name
    _run.__doc__ = entry.description
    return _run


def build_langchain_tools(dependencies: Dict[str, Any]) -> List[StructuredTool]:
    """Build LangChain tools bound to this request's trusted dependencies."""
    tools: List[StructuredTool] = []
    for entry in TOOL_REGISTRY.values():
        tools.append(
            StructuredTool.from_function(
                func=_make_runner(entry, dependencies),
                name=entry.name,
                description=entry.description,
                args_schema=entry.param_model,
            )
        )
    return tools


__all__ = [
    "AUTHORITY_FIELDS",
    "build_langchain_tools",
    "build_tool_params",
    "execute_registry_tool",
]
