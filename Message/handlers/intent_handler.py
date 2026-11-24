"""意图识别与路由处理器

纯规则关键词分类（零 LLM 成本），五类粗粒度意图：

    投诉 complaint / 售后 aftersale / 咨询 consult / 闲聊 chitchat / 其他 other

挂在消息处理器链的关键词检测之后、AI 兜底之前，分类结果立即分流：

    complaint → 安抚话术 + 转人工（无在线客服时改发记录话术）
    chitchat  → 固定话术
    consult   → 知识库直答短路（客服知识命中才短路，未命中降级 AI）
    aftersale / other → 交给下游 AI 处理器兜底

设计原则（与 Channel/pinduoduo/message_rules 动作表一致）：

1. 每条文本消息强制落位一个意图，绝不静默丢弃
2. 分类未知一律降级为 other 走 AI 兜底，而不是丢掉
3. 分类是纯函数（classify_intent），可独立单测
4. 自动回复统一走 sender.send_text，回显抑制登记由 sender 负责，
   不会被人工回复沉淀误采
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

from bridge.context import Context, ContextType
from utils.logger_loguru import get_logger
from .base import BaseHandler

# 意图标签
INTENT_COMPLAINT = "complaint"
INTENT_AFTERSALE = "aftersale"
INTENT_CONSULT = "consult"
INTENT_CHITCHAT = "chitchat"
INTENT_OTHER = "other"

# 分类优先级：投诉 > 售后 > 咨询 > 闲聊（先重后轻）
_INTENT_PRIORITY = (
    INTENT_COMPLAINT,
    INTENT_AFTERSALE,
    INTENT_CONSULT,
    INTENT_CHITCHAT,
)

# 意图 -> IntentConfig 关键词字段名
_KEYWORD_FIELD = {
    INTENT_COMPLAINT: "complaint_keywords",
    INTENT_AFTERSALE: "aftersale_keywords",
    INTENT_CONSULT: "consult_keywords",
    INTENT_CHITCHAT: "chitchat_keywords",
}

# 打点用的路由动作描述
_ROUTE_ACTION = {
    INTENT_COMPLAINT: "transfer_with_comfort",
    INTENT_CHITCHAT: "canned_reply",
    INTENT_CONSULT: "knowledge_shortcut",
    INTENT_AFTERSALE: "ai_fallback",
    INTENT_OTHER: "ai_fallback",
}


def classify_intent(text: str, cfg) -> Tuple[str, str]:
    """纯规则关键词分类。

    Args:
        text: 买家消息文本
        cfg: IntentConfig（或提供同名关键词字段的任意对象）

    Returns:
        (意图标签, 命中关键词)；未命中返回 (INTENT_OTHER, "")
    """
    if not text or not text.strip():
        return INTENT_OTHER, ""
    lower = text.lower()
    stripped = text.strip()
    for intent in _INTENT_PRIORITY:
        # 闲聊只对短消息生效，避免长消息因寒暄词被误判
        if intent == INTENT_CHITCHAT and len(stripped) > cfg.chitchat_max_chars:
            continue
        keywords = getattr(cfg, _KEYWORD_FIELD[intent], ()) or ()
        for keyword in keywords:
            if str(keyword).lower() in lower:
                return intent, str(keyword)
    return INTENT_OTHER, ""


def _shares_token(query: str, *texts: str) -> bool:
    """轻量相关性护栏：查询与命中文本需共享至少一个词（jieba 分词，长度>=2）。

    防止知识库弱匹配被直答给客户；jieba 不可用时不拦截（返回 True）。
    """
    try:
        import jieba
    except Exception:
        return True
    query_tokens = {t.strip() for t in jieba.cut(query) if len(t.strip()) >= 2}
    if not query_tokens:
        return False
    for text in texts:
        text_tokens = {t.strip() for t in jieba.cut(text or "") if len(t.strip()) >= 2}
        if query_tokens & text_tokens:
            return True
    return False


class IntentHandler(BaseHandler):
    """意图识别与路由处理器。

    依赖均可注入（测试用），默认懒加载真实实现；初始化不做任何 I/O。
    """

    def __init__(self, config_override=None, knowledge_service=None, sender_getter=None):
        super().__init__("IntentHandler")
        self.config_override = config_override
        self.knowledge_service = knowledge_service
        self.sender_getter = sender_getter

    # ------------------------------------------------------------------
    # 依赖获取（懒加载，失败降级）
    # ------------------------------------------------------------------

    def _get_cfg(self):
        if self.config_override is not None:
            return self.config_override
        try:
            from config import config as app_config
            return app_config.get_model().intent
        except Exception as e:
            self.logger.warning(f"intent config load failed: error_type={type(e).__name__}")
            return None

    def _get_sender(self, channel_type=None):
        if self.sender_getter is not None:
            return self.sender_getter(channel_type)
        try:
            from bridge.sender import get_sender
            return get_sender(channel_type)
        except Exception as e:
            self.logger.warning(f"sender resolve failed: error_type={type(e).__name__}")
            return None

    def _get_knowledge_service(self):
        if self.knowledge_service is not None:
            return self.knowledge_service
        try:
            from core.di_container import container
            from database.knowledge_service import KnowledgeService
            return container.get(KnowledgeService)
        except Exception as e:
            self.logger.warning(f"knowledge service resolve failed: error_type={type(e).__name__}")
            return None

    @staticmethod
    def _identity(context: Context) -> Optional[Tuple[str, str, str]]:
        """从 context.kwargs 提取 (shop_id, user_id, from_uid)。"""
        kwargs = context.kwargs

        def _pick(key):
            return getattr(kwargs, key, None) or (
                kwargs.get(key) if isinstance(kwargs, dict) else None
            )

        shop_id = _pick("shop_id")
        user_id = _pick("user_id")
        from_uid = _pick("from_uid")
        if not all([shop_id, user_id, from_uid]):
            return None
        return str(shop_id), str(user_id), str(from_uid)

    # ------------------------------------------------------------------
    # 处理器接口
    # ------------------------------------------------------------------

    def can_handle(self, context: Context) -> bool:
        if context.type != ContextType.TEXT:
            return False
        if not context.content or not isinstance(context.content, str):
            return False
        cfg = self._get_cfg()
        return bool(cfg is not None and cfg.enabled)

    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        try:
            cfg = self._get_cfg()
            if cfg is None or not cfg.enabled:
                return False
            text = context.content.strip()
            if not text:
                return False

            intent, matched = classify_intent(text, cfg)
            ids = self._identity(context)
            message_id = (metadata or {}).get("message_id", "unknown")
            # 打点：只记意图/命中词/长度，不记原文（隐私约定）
            self.logger.info(
                f"intent classified - intent={intent} matched={matched!r} "
                f"message_id={message_id} content_length={len(text)} "
                f"route={_ROUTE_ACTION.get(intent, 'ai_fallback')}"
            )
            if ids is None:
                # 缺会话身份就无法发送/转接，降级给 AI（它有自己的降级路径）
                self.logger.warning("intent route skipped: missing conversation identity")
                return False

            if intent == INTENT_COMPLAINT:
                return await self._route_complaint(context, ids, cfg)
            if intent == INTENT_CHITCHAT:
                return await self._send_text(context, ids, cfg.chitchat_reply)
            if intent == INTENT_CONSULT:
                return await self._route_consult(context, ids, text, cfg)
            # aftersale / other：交给下游 AI 处理器
            return False
        except Exception as e:
            self.logger.error(f"intent handling failed: error_type={type(e).__name__}")
            return False

    # ------------------------------------------------------------------
    # 路由动作
    # ------------------------------------------------------------------

    async def _route_complaint(
        self, context: Context, ids: Tuple[str, str, str], cfg
    ) -> bool:
        """投诉：有在线客服 → 安抚话术 + 转人工；无客服 → 记录话术。"""
        shop_id, user_id, from_uid = ids
        sender = self._get_sender(context.channel_type)
        if sender is None:
            return False
        try:
            cs_list = await asyncio.to_thread(sender.get_cs_list, shop_id, user_id)
        except Exception as e:
            self.logger.error(f"get_cs_list failed: error_type={type(e).__name__}")
            return False

        my_cs_uid = f"cs_{shop_id}_{user_id}"
        available = (
            [uid for uid in cs_list.keys() if uid != my_cs_uid]
            if isinstance(cs_list, dict)
            else []
        )
        if not available:
            self.logger.info("complaint routed to record-only: no cs online")
            return await self._send_text(context, ids, cfg.complaint_no_cs_reply)

        handled = await self._send_text(context, ids, cfg.complaint_reply)
        try:
            result = await asyncio.to_thread(
                sender.transfer_to_cs, shop_id, user_id, from_uid, available[0]
            )
            if not (result and result.get("success")):
                self.logger.warning("complaint transfer failed")
            else:
                self.logger.info(f"complaint transferred to cs={available[0]}")
                handled = True
        except Exception as e:
            self.logger.error(f"transfer_to_cs failed: error_type={type(e).__name__}")
        return handled

    async def _route_consult(
        self, context: Context, ids: Tuple[str, str, str], text: str, cfg
    ) -> bool:
        """咨询：客服知识命中且相关 → 直答短路；否则降级 AI。"""
        shop_id, _user_id, _from_uid = ids
        ks = self._get_knowledge_service()
        if ks is None:
            return False
        try:
            result = await asyncio.to_thread(
                ks.search_knowledge, shop_id, text, None, 3
            )
        except Exception as e:
            self.logger.warning(f"consult knowledge search failed: error_type={type(e).__name__}")
            return False

        hits = (result or {}).get("customer_service_knowledge") or []
        if not hits:
            self.logger.info("consult routed to ai: no knowledge hit")
            return False

        top = hits[0]
        title = getattr(top, "title", "") or ""
        content = getattr(top, "content", "") or ""
        if not _shares_token(text, title, content):
            self.logger.info("consult routed to ai: weak knowledge match")
            return False

        answer = (content.strip() or title.strip())
        if not answer:
            return False
        if len(answer) > cfg.knowledge_max_chars:
            answer = answer[: cfg.knowledge_max_chars] + "…"
        self.logger.info(f"consult answered from knowledge: hit_title={title[:50]!r}")
        return await self._send_text(context, ids, answer)

    async def _send_text(
        self, context: Context, ids: Tuple[str, str, str], text: str
    ) -> bool:
        """发送固定话术；发送失败降级返回 False（交给下游 AI 重试）。"""
        if not text:
            return False
        shop_id, user_id, from_uid = ids
        sender = self._get_sender(context.channel_type)
        if sender is None:
            return False
        try:
            await asyncio.to_thread(sender.send_text, shop_id, user_id, from_uid, text)
            return True
        except Exception as e:
            self.logger.error(f"intent reply send failed: error_type={type(e).__name__}")
            return False
