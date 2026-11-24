"""
自定义 CustomerAgent 实现

对外接口保持不变（async_reply / initialize_async / close / record_context），
内部 Agent 循环改为 LangGraph ``create_agent``：

- lc_chat_model: LiteLLM 传输之上的 LangChain ChatModel（含视觉降级）
- lc_tools: 将 @agent_tool 注册表适配为 LangChain 工具（可信身份注入）
- lc_messages: OpenAI 消息 dict ↔ LangChain 消息转换

会话持久化、上下文压缩、Prompt 构建仍由 SessionManager / MessageBuilder 负责。
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

from Agent.bot import Bot

# 导入工具模块，触发 @agent_tool 装饰器注册
from Agent.CustomerAgent.tools import (
    send_goods_link,                  # noqa: F401  — 注册 send_goods_link 工具
    move_conversation,                 # noqa: F401  — 注册 transfer_conversation 工具
    get_product_list,                 # noqa: F401  — 注册 get_shop_products 工具
    get_product_knowledge,             # noqa: F401  — 注册 get_product_knowledge 工具
    search_customer_service_knowledge,  # noqa: F401  — 注册 search_customer_service_knowledge 工具
)
from bridge.context import (
    Context,
    _context_value,
    context_scope,
    make_conversation_key,
)
from bridge.reply import Reply, ReplyType
from Agent.CustomerAgent.custom.session_manager import SessionManager
from Agent.CustomerAgent.custom.tool_decorator import TOOL_REGISTRY
from utils.logger_loguru import get_logger

# 导入重构后的模块
from Agent.CustomerAgent.custom.agent_config import (
    AgentConfig,
    DEFAULT_DB_PATH,
    DEFAULT_TOKEN_WINDOW,
    DEFAULT_COMPRESS_RATIO,
    DEFAULT_RETAIN_COUNT,
    DEFAULT_MAX_LOOPS,
    DEFAULT_TEMPERATURE,
)
from Agent.CustomerAgent.custom.lc_chat_model import LiteLLMChatModel
from Agent.CustomerAgent.custom.lc_messages import (
    final_ai_text,
    openai_messages_to_langchain,
    split_system_prompt,
)
from Agent.CustomerAgent.custom.lc_tools import build_langchain_tools
from Agent.CustomerAgent.custom.llm_client import LLMClient, LLMResponse
from Agent.CustomerAgent.custom.message_builder import MessageBuilder
from Agent.CustomerAgent.custom.multimodal import (
    decode_history_content,
    encode_history_content,
    extract_context_images,
    has_image_blocks,
    strip_image_blocks,
)
from utils.llm_transport import LLMErrorCategory

logger = get_logger("CustomerAgent")


class CustomerAgent(Bot):
    """
    LangGraph 客服 Agent（对外接口与旧版一致）

    核心循环：
    1. 加载历史消息
    2. 检查上下文压缩
    3. 构建 messages 列表（MessageBuilder）
    4. create_agent → 模型与工具循环（LangGraph）
    5. 持久化中间 tool 轨迹，返回最终回复
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        token_window: int = DEFAULT_TOKEN_WINDOW,
        compress_ratio: float = DEFAULT_COMPRESS_RATIO,
        retain_count: int = DEFAULT_RETAIN_COUNT,
        max_loops: int = DEFAULT_MAX_LOOPS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        super().__init__()
        self._is_initialized = False

        # 配置参数
        self._config = AgentConfig(
            db_path=db_path or DEFAULT_DB_PATH,
            token_window=token_window,
            compress_ratio=compress_ratio,
            retain_count=retain_count,
            max_loops=max_loops,
            temperature=temperature,
        )

        # 子组件（延迟初始化）
        self._llm_client: Optional[LLMClient] = None
        self._active_profile = None
        self._message_builder: Optional[MessageBuilder] = None
        self._session_manager: Optional[SessionManager] = None
        self._initialize_lock = asyncio.Lock()
        self._conversation_locks: Dict[str, asyncio.Lock] = {}
        self._fallback_session_id = f"fallback_{uuid.uuid4().hex}"

        logger.info("CustomerAgent 实例创建成功")

    async def initialize_async(self) -> bool:
        """Initialize once, even when the first messages arrive concurrently."""
        if self._is_initialized:
            return True
        async with self._initialize_lock:
            if self._is_initialized:
                return True
            return await self._initialize_async_unlocked()

    async def _initialize_async_unlocked(self) -> bool:
        """异步初始化 Agent"""
        if self._is_initialized:
            return True

        try:
            # 1. 从配置文件加载配置
            self._config = AgentConfig.load_from_config()

            # 2. 验证配置
            if not self._config.validate():
                return False

            # Snapshot the complete validated profile.  A later settings save
            # must not mutate this account's in-flight client.
            self._active_profile = self._config.profile

            # 3. 初始化 LLM 客户端（上下文压缩摘要仍走此通道）
            self._llm_client = LLMClient(
                profile=self._active_profile,
                temperature=self._config.temperature,
            )
            await self._llm_client.initialize()

            # 4. 初始化会话管理器
            self._session_manager = SessionManager(
                db_path=self._config.db_path,
                token_window=self._config.token_window,
                compress_ratio=self._config.compress_ratio,
                retain_count=self._config.retain_count,
                model_name=self._config.model_name,
            )

            # 5. 初始化消息构建器
            self._message_builder = MessageBuilder(
                instructions=self._config.instructions,
            )

            # 6. 记录已注册工具（执行时由 build_langchain_tools 按请求绑定）
            tool_names = list(TOOL_REGISTRY.keys())
            logger.info(f"已加载 {len(tool_names)} 个工具: {tool_names}")

            self._is_initialized = True
            logger.info(f"CustomerAgent 初始化成功: model={self._config.model_name}")
            return True

        except Exception as e:
            logger.error(
                f"CustomerAgent 初始化失败: error_type={type(e).__name__}"
            )
            return False

    async def async_reply(self, query: str, context: Context = None) -> Reply:
        """Reply serially per customer conversation."""
        session_id = self._session_id(context, query)
        lock = self._conversation_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            return await self._async_reply_unlocked(query, context, session_id)

    async def close(self) -> None:
        """Release per-account LLM, database, and in-memory resources."""
        async with self._initialize_lock:
            llm_client = self._llm_client
            session_manager = self._session_manager
            self._llm_client = None
            self._session_manager = None
            self._message_builder = None
            self._active_profile = None
            self._conversation_locks.clear()
            self._is_initialized = False

        if llm_client is not None:
            try:
                await llm_client.close()
            except Exception as exc:
                logger.warning(
                    f"LLM client cleanup failed: error_type={type(exc).__name__}"
                )
        if session_manager is not None:
            try:
                await asyncio.to_thread(session_manager.dispose)
            except Exception as exc:
                logger.warning(
                    f"session manager cleanup failed: error_type={type(exc).__name__}"
                )

    async def _async_reply_unlocked(
        self,
        query: str,
        context: Context = None,
        session_id: Optional[str] = None,
    ) -> Reply:
        """异步回复接口"""
        # 延迟初始化
        if not self._is_initialized:
            if not await self.initialize_async():
                return Reply(ReplyType.TEXT, "AI客服初始化失败，请检查配置。")

        try:
            # 构建 session_id 和 dependencies
            if context and context.channel_type and context_scope(context).get("user_id"):
                dependencies = self._message_builder.build_dependencies(context)
            else:
                dependencies = {}
            session_id = session_id or self._session_id(context, query)

            # 加载历史并检查压缩（DB 操作放工作线程，避免阻塞事件循环）
            history = await asyncio.to_thread(self._session_manager.get_history, session_id)
            if await asyncio.to_thread(self._session_manager.should_compress, session_id):
                logger.info(f"触发上下文压缩: session_id={session_id}")
                await self._compress_with_llm(session_id)
                # 压缩后重新加载，使本轮回复使用压缩后的历史
                history = await asyncio.to_thread(self._session_manager.get_history, session_id)

            # 当前这条买家消息附带的图片。只认 IMAGE 类型：买家在文本里粘
            # 一个链接属于文本内容，不该因此触发图片抓取。
            current_images = extract_context_images(context)

            # 预取商品列表（拼多多 HTTP，放工作线程）并注入 dependencies，
            # 避免 build_messages 内同步阻塞
            # Persist the user turn before invoking the model.  This keeps the
            # durable transcript complete even when the model or a tool fails.
            # 带图消息用信封持久化，历史里才能把图片本身还原出来。
            await asyncio.to_thread(
                self._session_manager.add_message,
                session_id=session_id,
                role="user",
                content=encode_history_content(query, current_images),
            )

            shop_id = dependencies.get("shop_id")
            user_id = dependencies.get("user_id")
            if shop_id and user_id:
                dependencies["product_list"] = await asyncio.to_thread(
                    self._message_builder.fetch_product_list_text, shop_id, user_id
                )
            else:
                dependencies["product_list"] = ""

            # 构建 messages
            messages = self._message_builder.build_messages(
                query,
                history,
                dependencies,
                images=current_images,
            )

            # 执行 Agent 循环
            final_content = await self._run_agent_loop(
                messages, dependencies, session_id=session_id
            )

            # 保存最终回复到历史（DB 写入放工作线程，避免阻塞事件循环）
            await asyncio.to_thread(
                self._session_manager.add_message,
                session_id=session_id,
                role="assistant",
                content=final_content,
            )

            return Reply(ReplyType.TEXT, final_content or "抱歉，我暂时无法回复。")

        except Exception as e:
            logger.error(
                f"CustomerAgent 回复失败: error_type={type(e).__name__}"
            )
            return Reply(ReplyType.TEXT, "抱歉，我现在无法回复，请稍后再试。")

    def _build_chat_model(self) -> LiteLLMChatModel:
        """LangChain chat model bound to this account's validated profile."""
        if self._active_profile is None:
            raise RuntimeError("CustomerAgent 尚未初始化")
        return LiteLLMChatModel(
            profile=self._active_profile,
            temperature=self._config.temperature,
        )

    def _agent_recursion_limit(self) -> int:
        """Map max_loops onto LangGraph's node-step budget.

        Each tool round is roughly a model node plus a tool node; add headroom
        for the initial and final model calls.
        """
        return max(4, int(self._config.max_loops) * 2 + 3)

    async def _run_agent_loop(
        self,
        messages: List[Dict[str, Any]],
        dependencies: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> str:
        """LangGraph create_agent 循环：模型 ⇄ 工具，直到产出最终文本。"""
        system_prompt, rest = split_system_prompt(messages)
        lc_input = openai_messages_to_langchain(rest)
        # create_agent prepends system_prompt as a SystemMessage when set.
        input_len = len(lc_input) + (1 if system_prompt else 0)

        try:
            model = self._build_chat_model()
            tools = build_langchain_tools(dependencies or {})
            agent = create_agent(
                model,
                tools or None,
                system_prompt=system_prompt or None,
            )
        except Exception as e:
            logger.error(f"Agent 构建失败: error_type={type(e).__name__}")
            return "抱歉，AI 服务暂时不可用，请稍后再试。"

        current: List[Any] = list(lc_input)
        received_state = False
        try:
            async for state in agent.astream(
                {"messages": lc_input},
                config={"recursion_limit": self._agent_recursion_limit()},
                stream_mode="values",
            ):
                msgs = state.get("messages") if isinstance(state, dict) else None
                if msgs:
                    current = list(msgs)
                    received_state = True
        except Exception as e:
            logger.error(f"Agent 循环失败: error_type={type(e).__name__}")
            text = final_ai_text(current)
            if text:
                await self._persist_current(current, input_len, session_id)
                return text
            return "抱歉，AI 服务暂时不可用，请稍后再试。"

        await self._persist_current(current, input_len, session_id)
        content = final_ai_text(current)
        if not content and not received_state:
            return "抱歉，AI 服务暂时不可用，请稍后再试。"
        return content or "抱歉，我暂时无法回复。"

    async def _persist_current(
        self,
        current: List[Any],
        input_len: int,
        session_id: Optional[str],
    ) -> None:
        """Persist intermediate tool rows; final text is saved by async_reply."""
        if not session_id or not self._session_manager:
            return
        rows: List[Dict[str, Any]] = []
        for message in current[input_len:]:
            if isinstance(message, ToolMessage):
                rows.append(
                    {
                        "role": "tool",
                        "content": message.text(),
                        "tool_call_id": message.tool_call_id,
                    }
                )
            elif isinstance(message, AIMessage) and message.tool_calls:
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
                        "tool_call_id": None,
                    }
                )
        for row in rows:
            try:
                await asyncio.to_thread(
                    self._session_manager.add_message,
                    session_id=session_id,
                    role=row["role"],
                    content=row["content"],
                    tool_call_id=row.get("tool_call_id"),
                )
            except Exception as exc:
                logger.warning(
                    f"persist agent transcript failed: error_type={type(exc).__name__}"
                )
                break

    async def _chat_with_vision_fallback(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """调用模型；图片内容被拒绝时退回纯文本，保证买家一定收到回复。

        这里刻意不预判、也不缓存「该模型不支持图片」这一结论：

        - 预判不可行：LiteLLM 的能力表对 volcengine / qwen / zhipu 的视觉
          模型同样返回不支持（实测 doubao-1-5-vision-pro 与 qwen-vl-max 均为
          False），拿它当开关会正好在支持视觉的模型上关掉图片。
        - 缓存不可取：「模型不接受图片」和「这一张图片抓不到」在错误码上是
          同一个参数类错误，无法区分。若因后者把前者永久记下来，一个本来
          支持视觉的模型会被静默降级到重启为止——这个失败比多一次重试严重。

        因此只做单次兜底：去掉图片块重试一次。真正不支持视觉的模型会因此
        每次图片消息多一次往返，并在日志里持续可见，便于发现配置问题。
        """
        if not has_image_blocks(messages):
            return await self._llm_client.chat(messages, **kwargs)

        try:
            return await self._llm_client.chat(messages, **kwargs)
        except Exception as exc:
            # 供应商拒绝图片内容时表现为参数类错误（BadRequest / 参数不支持）
            if getattr(exc, "category", None) is not LLMErrorCategory.PARAMETER:
                raise

            # 重试失败会原样抛出；此时不改变任何判定，交由上层按原有的
            # 「LLM 不可用」路径处理。
            stripped = strip_image_blocks(messages)
            response = await self._llm_client.chat(stripped, **kwargs)
            messages[:] = stripped

            profile = getattr(self, "_active_profile", None)
            logger.warning(
                "图片内容被模型拒绝，本轮已降级为纯文本；"
                f"provider={getattr(profile, 'provider', '')} "
                f"model={getattr(profile, 'model_name', '')}。"
                "若该模型支持视觉，请检查图片地址是否可被供应商访问。"
            )
            return response

    def _session_id(self, context: Optional[Context], query: str) -> str:
        if context is not None and context_scope(context).get("recipient_uid"):
            return make_conversation_key(context)
        return self._fallback_session_id

    def _conversation_session_id(self, context: Context) -> str:
        """会话键。客服侧消息的对话方是 to_uid，其余是 from_uid。

        两个方向必须落到同一个会话，否则 AI 看不到自己刚推过的内容。
        """
        customer_uid = ""
        if _context_value(context, "origin") == "merchant":
            customer_uid = _context_value(context, "to_uid")
        return make_conversation_key(context, customer_uid or None)

    @staticmethod
    def _context_role(context: Context) -> str:
        """会话历史里的角色：客服侧记 assistant，其余记 user。

        CONTEXT_ONLY 不只有客服侧消息：买家侧的 type=41「当前用户来自 商品详情页」
        也走这条路径。若一律记成 assistant，模型会以为那句话是自己说的，
        因此角色必须由 origin 推导，不能写死。
        """
        return "assistant" if _context_value(context, "origin") == "merchant" else "user"

    async def record_context(self, context: Context) -> bool:
        """把我方消息写入会话历史，但不生成回复。

        用于客服侧文本与商品卡（CONTEXT_ONLY）：让 AI 知道刚刚推送过什么，
        减少重复推荐。写入失败不影响主流程，调用方只需记录日志。
        """
        if not self._is_initialized:
            if not await self.initialize_async():
                return False

        content = context.content if context is not None else ""
        if not isinstance(content, str) or not content.strip():
            return False

        session_id = self._conversation_session_id(context)
        try:
            return await asyncio.to_thread(
                self._session_manager.add_message,
                session_id=session_id,
                role=self._context_role(context),
                content=content,
            )
        except Exception as exc:
            logger.warning(
                f"record_context failed: error_type={type(exc).__name__}"
            )
            return False

    async def _compress_with_llm(
        self,
        session_id: str,
    ) -> None:
        """使用 LLM 生成摘要并压缩历史。

        本方法运行在事件循环中（由 async_reply 调用），直接 await LLM 即可；
        切勿使用 asyncio.run——在已有事件循环里会抛 RuntimeError，导致
        压缩从未真正执行（历史只增不减）。
        """

        def summary_line(msg: Dict[str, Any]) -> str:
            """压缩输入的一行。

            带图消息在库里是信封 JSON，直接截取等于把 JSON 塞给摘要模型；
            图片本身不参与文字摘要，只保留文本并注明当时有图——压缩会删掉
            原始消息，这句注明是「买家发过图」唯一能留下的线索。
            """
            text, images = decode_history_content(msg.get("content", ""))
            suffix = "（含图片）" if images else ""
            return f"[{msg.get('role', 'unknown')}]: {text[:200]}{suffix}"

        async def summary_llm(messages: List[Dict[str, Any]]) -> str:
            """异步调用 LLM 生成摘要"""
            summary_prompt = (
                "请简洁地总结以下对话的要点，保留关键信息和用户意图。\n\n"
                f"对话内容（共 {len(messages)} 条消息）：\n"
                + "\n".join(summary_line(msg) for msg in messages if msg.get("content"))
            )

            try:
                response = await self._llm_client.chat(
                    messages=[
                        {"role": "system", "content": "你是一个对话摘要助手。请简洁地总结对话要点。"},
                        {"role": "user", "content": summary_prompt},
                    ],
                    tool_choice="none",
                    use_tools=False,
                )
                return response.content or "[摘要生成失败]"
            except Exception as e:
                logger.error(
                    f"生成摘要失败: error_type={type(e).__name__}"
                )
                return "[摘要生成失败]"

        await self._session_manager.compress_history(session_id, summary_llm)
