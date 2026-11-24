"""Convert SessionManager / MessageBuilder OpenAI dicts to LangChain messages."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from Agent.CustomerAgent.custom.lc_chat_model import (
    message_to_openai,
    openai_to_message,
)


def openai_messages_to_langchain(
    messages: List[Dict[str, Any]],
) -> List[BaseMessage]:
    return [openai_to_message(message) for message in messages]


def langchain_messages_to_openai(
    messages: List[BaseMessage],
) -> List[Dict[str, Any]]:
    return [message_to_openai(message) for message in messages]


def split_system_prompt(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Peel the leading system message off a MessageBuilder payload."""
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        return str(content), list(messages[1:])
    return "", list(messages)


def persist_new_messages(
    new_messages: List[BaseMessage],
) -> List[Dict[str, Any]]:
    """Turn post-invoke LangChain messages into SessionManager rows.

    Assistant tool-call turns are stored as the same JSON envelope the
    MessageBuilder already understands for replay.
    """
    rows: List[Dict[str, Any]] = []
    for message in new_messages:
        if isinstance(message, ToolMessage):
            rows.append(
                {
                    "role": "tool",
                    "content": message.text(),
                    "tool_call_id": message.tool_call_id,
                }
            )
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                payload = {
                    "content": message.text(),
                    "tool_calls": [
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
                    ],
                }
                rows.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                )
            else:
                rows.append(
                    {
                        "role": "assistant",
                        "content": message.text(),
                    }
                )
        elif isinstance(message, HumanMessage):
            # Catalog / injected user turns from MessageBuilder are plain text
            # or multimodal blocks; persist text only (images ride the envelope
            # on the buyer turn, which async_reply already stored).
            rows.append({"role": "user", "content": message.text()})
        elif isinstance(message, SystemMessage):
            continue
    return rows


def final_ai_text(messages: List[BaseMessage]) -> str:
    """Prefer the last text-only AI reply; fall back to the last AI content."""
    fallback = ""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        if not fallback:
            fallback = message.text()
        if not message.tool_calls and message.text():
            return message.text()
    return fallback


__all__ = [
    "final_ai_text",
    "langchain_messages_to_openai",
    "openai_messages_to_langchain",
    "persist_new_messages",
    "split_system_prompt",
]
