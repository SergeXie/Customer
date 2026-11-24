"""人工回复沉淀知识库。

买家提问 -> 人工（客服侧）文本回复 -> 自动配对沉淀为客服知识
（title = 买家问题，content = 人工回复，tags 默认「人工回复」），
进入既有 RAG 检索链路，供 AI 回答后续同类问题。

三个关键设计：

1. 回显抑制：我方（AI 回复 / 关键词回复 / 工具文本）经发送器发出的文本会登记
   到发送缓存；拼多多 WebSocket 把我方消息回流为客服侧文本时按内容比对跳过，
   避免把 AI 自己的回答沉淀成「人工知识」造成自激污染。
2. 问题配对：买家纯文本进问题缓冲（TTL 内有效）；人工回复时优先取「上一次我方
   发言之后」新提出的问题（已答过的不重复配对），没有则退回窗口内全部问题。
   首条回复建知识条目；客服连发的后续消息在合并窗口内追加进同一条（客服习惯
   分多条发送一个答案）。
3. 去重：同店铺同标题+同内容已存在则不重复沉淀（复用批量导入的去重口径）。

本模块只做「观测 -> 配对 -> 沉淀」，消息类型判定（哪些算买家问题 / 人工文本
回复）由渠道层（pdd_message_handler）完成，保持渠道无关。
失败绝不影响消息主流程，调用方无需处理异常。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.logger_loguru import get_logger

logger = get_logger("HumanKnowledge")

# 连接键内单条列表的容量上限（防内存膨胀）
_LIST_CAP = 50
# 全量清扫间隔（秒）：清掉过期会话的残留状态
_SWEEP_INTERVAL = 60.0
# 问题最小长度（过滤「嗯」「好」等非问题碎片）
_MIN_QUESTION_CHARS = 2
# 标题上限（title 列 255，留余量）
_TITLE_MAX = 200


class HumanKnowledgeCapture:
    """人工回复沉淀状态机（线程安全）。

    消息层在事件循环线程调用 remember_question / mark_sent（纯内存，微秒级），
    capture_reply 含数据库写入，应由调用方放进工作线程（asyncio.to_thread）。
    """

    def __init__(self, knowledge_service: Any = None):
        self._lock = threading.RLock()
        # conv_key -> [(ts, text)] 买家待配对问题
        self._questions: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}
        # conv_key -> [(ts, text)] 我方已发送文本（回显比对）
        self._sent: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}
        # conv_key -> {"cs_id", "content", "ts"} 最近一次沉淀（连发消息合并用）
        self._last_entry: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._last_sweep = 0.0
        # 测试注入点
        self._knowledge = knowledge_service
        self.config_override: Optional[Any] = None
        self._clock: Callable[[], float] = time.time

    # ---------- 基础设施 ----------

    @staticmethod
    def _key(shop_id: Any, customer_uid: Any) -> Tuple[str, str]:
        return (str(shop_id or ""), str(customer_uid or ""))

    def _get_cfg(self):
        if self.config_override is not None:
            return self.config_override
        from config import config as app_config
        return app_config.get_model().knowledge_capture

    def _get_knowledge(self):
        if self._knowledge is None:
            from core.di_container import container
            from database.knowledge_service import KnowledgeService
            self._knowledge = container.get(KnowledgeService)
        return self._knowledge

    @staticmethod
    def _norm(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        return " ".join(text.split()).strip()

    def _sweep(self, now: float) -> None:
        """周期性清理过期状态（调用方须持锁）。"""
        if now - self._last_sweep < _SWEEP_INTERVAL:
            return
        self._last_sweep = now
        # 问题缓冲按窗口清扫；发送缓存按 echo TTL 清扫
        try:
            cfg = self._get_cfg()
            q_ttl = float(cfg.question_window_seconds)
            e_ttl = float(cfg.echo_ttl_seconds)
            m_ttl = float(cfg.merge_window_seconds)
        except Exception:
            q_ttl, e_ttl, m_ttl = 600.0, 600.0, 60.0
        for store, ttl in ((self._questions, q_ttl), (self._sent, e_ttl)):
            for k in list(store.keys()):
                items = [(ts, t) for ts, t in store[k] if now - ts <= ttl]
                if items:
                    store[k] = items
                else:
                    del store[k]
        for k in list(self._last_entry.keys()):
            if now - self._last_entry[k]["ts"] > m_ttl:
                del self._last_entry[k]

    # ---------- 观测入口 ----------

    def remember_question(self, shop_id: Any, customer_uid: Any, text: Any) -> None:
        """记录买家纯文本问题（供后续人工回复配对）。"""
        q = self._norm(text)
        if len(q) < _MIN_QUESTION_CHARS:
            return
        key = self._key(shop_id, customer_uid)
        now = self._clock()
        with self._lock:
            self._sweep(now)
            items = self._questions.setdefault(key, [])
            # 连续重复提问去重（保留最新时间戳）
            if items and items[-1][1] == q:
                items[-1] = (now, q)
            else:
                items.append((now, q))
                if len(items) > _LIST_CAP:
                    del items[:-_LIST_CAP]

    def mark_sent(self, shop_id: Any, customer_uid: Any, text: Any) -> None:
        """登记我方发出的文本（回显抑制 + 记录「答到哪了」的时间点）。

        注意：不清空问题缓冲——转人工场景里「我要退货」问完才转接、我方转接
        通知（[玫瑰]）先于人工回复发出，清缓冲会把最需要沉淀的问题丢掉。
        配对时优先取本时间点之后的新问题来规避「AI 已答过的问题」。
        """
        t = self._norm(text)
        if not t:
            return
        key = self._key(shop_id, customer_uid)
        now = self._clock()
        with self._lock:
            self._sweep(now)
            items = self._sent.setdefault(key, [])
            items.append((now, t))
            if len(items) > _LIST_CAP:
                del items[:-_LIST_CAP]

    def _is_echo(self, key: Tuple[str, str], text: str, now: float, ttl: float) -> bool:
        """我方发送缓存里有同文本 -> 判定为自家回显（调用方须持锁）。"""
        for ts, t in self._sent.get(key, []):
            if now - ts <= ttl and t == text:
                return True
        return False

    def _take_questions(
        self, key: Tuple[str, str], now: float, window: float, last_sent_ts: float
    ) -> List[str]:
        """取出待配对问题并消费（调用方须持锁）。

        优先取「上一次我方发言之后」的新问题（AI 已答过的不再配对）；
        没有新问题时退回窗口内全部问题（转人工场景：问题在转接通知之前）。
        """
        items = [(ts, t) for ts, t in self._questions.pop(key, []) if now - ts <= window]
        if not items:
            return []
        after = [t for ts, t in items if ts > last_sent_ts]
        return after or [t for _, t in items]

    def _last_sent_ts(self, key: Tuple[str, str], now: float, ttl: float) -> float:
        ts_list = [ts for ts, _ in self._sent.get(key, []) if now - ts <= ttl]
        return max(ts_list) if ts_list else 0.0

    # ---------- 沉淀 ----------

    def capture_reply(self, shop_id: Any, customer_uid: Any, text: Any) -> Optional[int]:
        """人工（客服侧）文本回复 -> 沉淀知识。返回知识条目 id，未沉淀返回 None。

        含数据库写入，调用方应放进工作线程执行。
        """
        try:
            return self._capture_reply_inner(shop_id, customer_uid, text)
        except Exception as exc:
            logger.warning(f"人工回复沉淀失败（不影响消息主流程）: {type(exc).__name__}: {exc}")
            return None

    def _capture_reply_inner(self, shop_id: Any, customer_uid: Any, text: Any) -> Optional[int]:
        reply = self._norm(text)
        if not reply:
            return None
        cfg = self._get_cfg()
        if not cfg.enabled:
            return None
        if len(reply) < int(cfg.min_reply_chars):
            logger.debug(f"人工回复过短，不沉淀: {reply[:20]}")
            return None

        key = self._key(shop_id, customer_uid)
        now = self._clock()
        with self._lock:
            self._sweep(now)
            echo_ttl = float(cfg.echo_ttl_seconds)
            if self._is_echo(key, reply, now, echo_ttl):
                logger.debug(f"自家发送回显，跳过沉淀: {reply[:20]}")
                return None
            questions = self._take_questions(
                key, now, float(cfg.question_window_seconds), self._last_sent_ts(key, now, echo_ttl)
            )
            last = self._last_entry.get(key)
            merge_ok = (
                last is not None and now - float(last["ts"]) <= float(cfg.merge_window_seconds)
            )

        # 连发消息合并：没有新问题、且刚沉淀过 -> 追加进上一条知识
        if not questions and merge_ok:
            return self._merge_reply(cfg, key, last, reply, now)

        if not questions:
            logger.debug(f"人工回复无配对问题（窗口外或已消费），不沉淀: {reply[:20]}")
            return None

        title = self._build_title(questions)
        ks = self._get_knowledge()
        db_shop_id = ks.resolve_shop_id(shop_id)
        if db_shop_id is None:
            logger.warning(f"沉淀跳过：店铺未登记 shop_id={shop_id}")
            return None

        # 包含式判重：连发合并会拉长原条目内容，精确去重会漏；回答已被已有
        # 条目覆盖（同标题下包含该文本）则不再沉淀
        if ks.has_customer_service_answer(db_shop_id, title, reply):
            logger.debug(f"人工回复已被已有知识覆盖，跳过沉淀: title={title}")
            return None

        success, skipped = ks.batch_import_customer_service(
            db_shop_id,
            [{
                "title": title,
                "content": reply,
                "tags": cfg.tags,
                "enabled": bool(cfg.default_enabled),
            }],
        )
        if success <= 0:
            logger.debug(f"人工回复沉淀被去重或写入失败: title={title}")
            return None

        cs_id = ks.find_customer_service_id(db_shop_id, title, reply)
        with self._lock:
            self._last_entry[key] = {"cs_id": cs_id, "content": reply, "ts": now}
        logger.info(
            f"人工回复已沉淀为知识: id={cs_id}, title={title}, "
            f"enabled={cfg.default_enabled}, tags={cfg.tags}"
        )
        return cs_id

    def _merge_reply(
        self,
        cfg: Any,
        key: Tuple[str, str],
        last: Dict[str, Any],
        reply: str,
        now: float,
    ) -> Optional[int]:
        """客服连发的后续消息并入上一条知识（客服习惯分多条发送一个答案）。"""
        cs_id = last.get("cs_id")
        if not cs_id:
            return None
        merged = f"{last.get('content', '')}\n{reply}"
        ks = self._get_knowledge()
        updated = ks.update_customer_service(cs_id, content=merged)
        if updated is None:
            # 上一条已被删除/停用，放弃合并
            with self._lock:
                if self._last_entry.get(key) is last:
                    del self._last_entry[key]
            return None
        with self._lock:
            last["content"] = merged
            last["ts"] = now
        logger.info(f"人工回复已并入知识条目: id={cs_id}")
        return cs_id

    @staticmethod
    def _build_title(questions: List[str]) -> str:
        title = "；".join(questions)
        if len(title) > _TITLE_MAX:
            title = title[:_TITLE_MAX - 1] + "…"
        return title


# 模块级单例：消息链路各挂点统一走它
human_knowledge = HumanKnowledgeCapture()

__all__ = ["HumanKnowledgeCapture", "human_knowledge"]
