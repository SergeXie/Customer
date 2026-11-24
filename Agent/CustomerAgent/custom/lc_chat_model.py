"""LangChain chat model backed by the shared LiteLLM transport.

Preserves provider profiles, endpoint trust checks, tool policy and the
single-shot vision fallback of the previous hand-rolled agent loop.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from Agent.CustomerAgent.custom.multimodal import (
    has_image_blocks,
    strip_image_blocks,
)
from utils.llm_provider import LLMProfile
from utils.llm_transport import (
    LLMErrorCategory,
    LLMTransportError,
    NormalizedResponse,
    async_completion,
)
from utils.logger_loguru import get_logger

logger = get_logger("LCChatModel")


def openai_tool_calls_to_ai(tool_calls: Sequence[Any]) -> List[Dict[str, Any]]:
    """Convert transport-normalized tool calls into LangChain tool_call dicts."""
    converted: List[Dict[str, Any]] = []
    for call in tool_calls:
        raw_arguments = call.function.arguments
        if isinstance(raw_arguments, str):
            try:
                args = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_arguments}
        elif isinstance(raw_arguments, dict):
            args = raw_arguments
        else:
            args = {}
        if not isinstance(args, dict):
            args = {"_raw": str(args)}
        converted.append(
            {
                "name": call.function.name,
                "args": args,
                "id": call.id,
                "type": "tool_call",
            }
        )
    return converted


def message_to_openai(message: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain message into the portable Chat Completions dict."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.text()}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.text(),
        }
    if isinstance(message, AIMessage):
        payload: Dict[str, Any] = {
            "role": "assistant",
            "content": message.text() if message.text() else "",
        }
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.get("id") or "",
                    "function": {
                        "name": tc.get("name") or "",
                        "arguments": json.dumps(
                            tc.get("args") or {}, ensure_ascii=False
                        ),
                    },
                }
                for tc in message.tool_calls
            ]
        return payload
    # HumanMessage and anything else: keep multimodal content blocks intact.
    content = message.content
    if isinstance(content, list):
        return {"role": "user", "content": content}
    return {"role": "user", "content": message.text()}


def openai_to_message(payload: Dict[str, Any]) -> BaseMessage:
    """Convert a Chat Completions message dict into a LangChain message."""
    role = payload.get("role", "user")
    content = payload.get("content")
    if role == "system":
        return SystemMessage(content=content or "")
    if role == "tool":
        return ToolMessage(
            content=content or "",
            tool_call_id=str(payload.get("tool_call_id") or ""),
        )
    if role == "assistant":
        tool_calls = openai_tool_calls_to_ai(payload.get("tool_calls") or [])
        return AIMessage(content=content or "", tool_calls=tool_calls)
    if isinstance(content, list):
        return HumanMessage(content=content)
    return HumanMessage(content=content or "")


def normalized_to_ai_message(normalized: NormalizedResponse) -> AIMessage:
    tool_calls = openai_tool_calls_to_ai(normalized.tool_calls)
    return AIMessage(
        content=normalized.content or "",
        tool_calls=tool_calls,
        usage_metadata=(
            {
                "input_tokens": int(normalized.usage.get("prompt_tokens") or 0),
                "output_tokens": int(normalized.usage.get("completion_tokens") or 0),
                "total_tokens": int(normalized.usage.get("total_tokens") or 0),
            }
            if normalized.usage
            else None
        ),
    )


class LiteLLMChatModel(BaseChatModel):
    """LangChain chat model that delegates to ``async_completion``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile: LLMProfile
    temperature: float = 0.3
    bound_tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: str = "auto"
    request_timeout: float = 60.0

    @property
    def _llm_type(self) -> str:
        return "customer-litellm"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> "LiteLLMChatModel":
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.model_copy(
            update={
                "bound_tools": openai_tools,
                "tool_choice": tool_choice or "auto",
            }
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            "LiteLLMChatModel is async-only; use ainvoke/ainvoke_stream"
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        openai_messages = [message_to_openai(message) for message in messages]
        normalized = await self._complete_with_vision_fallback(openai_messages)
        return ChatResult(
            generations=[ChatGeneration(message=normalized_to_ai_message(normalized))]
        )

    async def _complete_with_vision_fallback(
        self,
        messages: List[Dict[str, Any]],
    ) -> NormalizedResponse:
        """Call the transport; strip images once if the provider rejects them."""
        if not has_image_blocks(messages):
            return await self._complete(messages)

        try:
            return await self._complete(messages)
        except LLMTransportError as exc:
            if exc.category is not LLMErrorCategory.PARAMETER:
                raise

            stripped = strip_image_blocks(messages)
            response = await self._complete(stripped)
            messages[:] = stripped
            logger.warning(
                "图片内容被模型拒绝，本轮已降级为纯文本；"
                f"provider={self.profile.provider.value} "
                f"model={self.profile.model_name}。"
                "若该模型支持视觉，请检查图片地址是否可被供应商访问。"
            )
            return response

    async def _complete(self, messages: List[Dict[str, Any]]) -> NormalizedResponse:
        use_tools = bool(self.bound_tools)
        return await async_completion(
            self.profile,
            messages,
            temperature=self.temperature,
            tools=self.bound_tools,
            tool_choice=self.tool_choice,
            use_tools=use_tools,
            timeout=self.request_timeout,
        )


__all__ = [
    "LiteLLMChatModel",
    "message_to_openai",
    "normalized_to_ai_message",
    "openai_to_message",
    "openai_tool_calls_to_ai",
]
