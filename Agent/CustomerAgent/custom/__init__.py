"""
Agent/CustomerAgent/custom/ - LangGraph 客服 Agent 实现

包含：
- agent_config: Agent 配置管理
- lc_chat_model: LangChain ChatModel（LiteLLM 传输 + 视觉降级）
- lc_tools: @agent_tool 注册表 → LangChain 工具（可信身份注入）
- lc_messages: OpenAI 消息 dict ↔ LangChain 消息
- llm_client: LLM 客户端封装（上下文压缩摘要仍使用）
- message_builder: 消息和 Prompt 构建器
- tool_executor: 工具执行器（兼容保留）
- tool_decorator: 工具系统（装饰器 + 注册表）
- session_manager: 会话历史管理（SQLite + 上下文压缩）
- customer_agent: 主入口（对外接口不变，内部走 create_agent）
"""

from Agent.CustomerAgent.custom.agent_config import (
    AgentConfig,
    DEFAULT_DB_PATH,
    DEFAULT_TOKEN_WINDOW,
    DEFAULT_COMPRESS_RATIO,
    DEFAULT_RETAIN_COUNT,
    DEFAULT_MAX_LOOPS,
    DEFAULT_TEMPERATURE,
)

from Agent.CustomerAgent.custom.lc_chat_model import (
    LiteLLMChatModel,
)

from Agent.CustomerAgent.custom.lc_tools import (
    build_langchain_tools,
    execute_registry_tool,
)

from Agent.CustomerAgent.custom.lc_messages import (
    final_ai_text,
    openai_messages_to_langchain,
    split_system_prompt,
)

from Agent.CustomerAgent.custom.llm_client import (
    LLMClient,
    LLMResponse,
)

from Agent.CustomerAgent.custom.message_builder import MessageBuilder

from Agent.CustomerAgent.custom.tool_executor import (
    ToolExecutor,
    ToolResult,
)

from Agent.CustomerAgent.custom.tool_decorator import (
    agent_tool,
    TOOL_REGISTRY,
    get_tools_for_llm,
    get_tool_entry,
    execute_tool,
    ToolEntry,
)

from Agent.CustomerAgent.custom.session_manager import (
    SessionManager,
    AgentMessage,
    TokenEstimator,
)

from Agent.CustomerAgent.custom.customer_agent import CustomerAgent

__all__ = [
    # 配置模块
    "AgentConfig",
    "DEFAULT_DB_PATH",
    "DEFAULT_TOKEN_WINDOW",
    "DEFAULT_COMPRESS_RATIO",
    "DEFAULT_RETAIN_COUNT",
    "DEFAULT_MAX_LOOPS",
    "DEFAULT_TEMPERATURE",
    # LangChain 集成
    "LiteLLMChatModel",
    "build_langchain_tools",
    "execute_registry_tool",
    "final_ai_text",
    "openai_messages_to_langchain",
    "split_system_prompt",
    # LLM 客户端
    "LLMClient",
    "LLMResponse",
    # 消息构建器
    "MessageBuilder",
    # 工具执行器
    "ToolExecutor",
    "ToolResult",
    # 工具装饰器
    "agent_tool",
    "TOOL_REGISTRY",
    "get_tools_for_llm",
    "get_tool_entry",
    "execute_tool",
    "ToolEntry",
    # 会话管理
    "SessionManager",
    "AgentMessage",
    "TokenEstimator",
    # 主 Agent
    "CustomerAgent",
]

