"""意图识别与路由处理器测试。

验证：
1. classify_intent 纯函数分类（优先级、闲聊短消息守卫、大小写、未命中）
2. _shares_token 知识直答相关性护栏
3. IntentHandler 分流（投诉转人工/无客服话术、闲聊话术、知识直答/截断、弱匹配降级）
4. 开关关闭 / 身份缺失 / 发送失败 / 非文本 都安全降级给下游
5. handler_chain 挂载顺序（关键词之后、AI 之前）
"""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from config import IntentConfig
from Message.handlers.intent_handler import (
    INTENT_AFTERSALE,
    INTENT_CHITCHAT,
    INTENT_COMPLAINT,
    INTENT_CONSULT,
    INTENT_OTHER,
    IntentHandler,
    _shares_token,
    classify_intent,
)
from bridge.context import ChannelType, Context, ContextType


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class FakeSender:
    """内存版 ReplySender（记录发送/转接调用）。"""

    def __init__(self, cs_list=None, transfer_result=None, send_error=None):
        self.sent = []
        self.transfers = []
        self.cs_list = cs_list if cs_list is not None else {}
        self.transfer_result = (
            transfer_result if transfer_result is not None else {"success": True}
        )
        self.send_error = send_error

    def send_text(self, shop_id, user_id, from_uid, text):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append((shop_id, user_id, from_uid, text))

    def get_cs_list(self, shop_id, user_id):
        return self.cs_list

    def transfer_to_cs(self, shop_id, user_id, from_uid, cs_uid):
        self.transfers.append((shop_id, user_id, from_uid, cs_uid))
        return self.transfer_result


class FakeKS:
    """内存版 KnowledgeService（只实现 search_knowledge）。"""

    def __init__(self, hits=None, error=None):
        self.hits = hits if hits is not None else []
        self.error = error
        self.queries = []

    def search_knowledge(self, shop_id, query, goods_id, limit):
        self.queries.append((shop_id, query, goods_id, limit))
        if self.error is not None:
            raise self.error
        return {"customer_service_knowledge": self.hits}


def make_context(text, with_identity=True):
    return Context.create_pinduoduo_context(
        content=text,
        msg_id="m1",
        shop_id="shop1" if with_identity else None,
        user_id="user1" if with_identity else None,
        from_uid="buyer1" if with_identity else None,
        channel_type=ChannelType.PINDUODUO,
    )


METADATA = {"message_id": "m1"}


# ---------------------------------------------------------------------------
# classify_intent 纯函数
# ---------------------------------------------------------------------------

class ClassifyIntentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = IntentConfig()

    def test_complaint_beats_aftersale_by_priority(self):
        intent, matched = classify_intent("我要投诉，必须给我退款", self.cfg)
        self.assertEqual(intent, INTENT_COMPLAINT)
        self.assertEqual(matched, "投诉")

    def test_aftersale_keyword(self):
        intent, matched = classify_intent("这个怎么退款", self.cfg)
        self.assertEqual(intent, INTENT_AFTERSALE)
        self.assertEqual(matched, "退款")

    def test_consult_keyword(self):
        intent, matched = classify_intent("这个多少钱", self.cfg)
        self.assertEqual(intent, INTENT_CONSULT)
        self.assertEqual(matched, "多少钱")

    def test_consult_beats_chitchat_by_priority(self):
        # "好的"是闲聊词，但咨询优先级更高
        intent, _ = classify_intent("好的什么时候发货", self.cfg)
        self.assertEqual(intent, INTENT_CONSULT)

    def test_chitchat_short_message(self):
        intent, matched = classify_intent("你好", self.cfg)
        self.assertEqual(intent, INTENT_CHITCHAT)
        self.assertEqual(matched, "你好")

    def test_chitchat_guard_skips_long_message(self):
        text = "你好啊今天天气真不错我们晚上一起出去玩吧记得带上伞"
        self.assertGreater(len(text.strip()), self.cfg.chitchat_max_chars)
        intent, _ = classify_intent(text, self.cfg)
        self.assertEqual(intent, INTENT_OTHER)

    def test_chitchat_case_insensitive(self):
        intent, _ = classify_intent("HELLO", self.cfg)
        self.assertEqual(intent, INTENT_CHITCHAT)

    def test_unmatched_returns_other(self):
        intent, matched = classify_intent("今天天气真好一起出去玩", self.cfg)
        self.assertEqual(intent, INTENT_OTHER)
        self.assertEqual(matched, "")

    def test_empty_text_returns_other(self):
        self.assertEqual(classify_intent("", self.cfg)[0], INTENT_OTHER)
        self.assertEqual(classify_intent("   ", self.cfg)[0], INTENT_OTHER)


# ---------------------------------------------------------------------------
# _shares_token 相关性护栏
# ---------------------------------------------------------------------------

class SharesTokenTests(unittest.TestCase):
    def test_shared_token_returns_true(self):
        self.assertTrue(_shares_token("什么时候发货", "发货时间说明", "无关内容"))

    def test_disjoint_tokens_returns_false(self):
        self.assertFalse(_shares_token("苹果手机壳", "连衣裙女夏"))

    def test_too_short_tokens_returns_false(self):
        self.assertFalse(_shares_token("嗯", "嗯嗯好的"))

    def test_empty_text_returns_false(self):
        self.assertFalse(_shares_token("退款流程", ""))


# ---------------------------------------------------------------------------
# IntentHandler 路由
# ---------------------------------------------------------------------------

class IntentHandlerRoutingTests(unittest.IsolatedAsyncioTestCase):
    def _handler(self, cfg=None, sender=None, ks=None):
        return IntentHandler(
            config_override=cfg or IntentConfig(),
            knowledge_service=ks,
            sender_getter=lambda _ct: sender,
        )

    def test_can_handle_requires_text_and_enabled(self):
        handler = self._handler(sender=FakeSender())
        self.assertTrue(handler.can_handle(make_context("你好")))
        self.assertFalse(
            handler.can_handle(
                Context(type=ContextType.IMAGE, content="img", channel_type=ChannelType.PINDUODUO)
            )
        )
        disabled = self._handler(cfg=IntentConfig(enabled=False), sender=FakeSender())
        self.assertFalse(disabled.can_handle(make_context("你好")))

    async def test_complaint_transfers_to_cs_with_comfort_reply(self):
        sender = FakeSender(cs_list={"cs_shop1_user1": {}, "cs_999": {}})
        handler = self._handler(sender=sender)
        handled = await handler.handle(make_context("我要投诉你们卖假货"), METADATA)
        self.assertTrue(handled)
        self.assertEqual(len(sender.sent), 1)
        self.assertIn("抱歉", sender.sent[0][3])
        self.assertEqual(sender.transfers, [("shop1", "user1", "buyer1", "cs_999")])

    async def test_complaint_without_cs_sends_record_reply(self):
        sender = FakeSender(cs_list={"cs_shop1_user1": {}})  # 只有自己的客服号
        handler = self._handler(sender=sender)
        handled = await handler.handle(make_context("我要投诉你们卖假货"), METADATA)
        self.assertTrue(handled)
        self.assertEqual(len(sender.sent), 1)
        self.assertIn("记录", sender.sent[0][3])
        self.assertEqual(sender.transfers, [])

    async def test_complaint_transfer_failure_still_returns_handled(self):
        sender = FakeSender(
            cs_list={"cs_999": {}}, transfer_result={"success": False}
        )
        handler = self._handler(sender=sender)
        handled = await handler.handle(make_context("我要投诉"), METADATA)
        self.assertTrue(handled)  # 安抚话术已发出
        self.assertEqual(len(sender.sent), 1)

    async def test_chitchat_sends_canned_reply(self):
        sender = FakeSender()
        handler = self._handler(sender=sender)
        handled = await handler.handle(make_context("在吗"), METADATA)
        self.assertTrue(handled)
        self.assertEqual(sender.sent[0][3], IntentConfig().chitchat_reply)
        self.assertEqual(sender.transfers, [])

    async def test_consult_answers_from_knowledge(self):
        ks = FakeKS(hits=[SimpleNamespace(title="发货时间", content="付款后48小时内发货")])
        sender = FakeSender()
        handler = self._handler(sender=sender, ks=ks)
        handled = await handler.handle(make_context("什么时候发货"), METADATA)
        self.assertTrue(handled)
        self.assertEqual(sender.sent[0][3], "付款后48小时内发货")
        self.assertEqual(ks.queries[0][0], "shop1")

    async def test_consult_knowledge_answer_truncated(self):
        long_content = "发货说明" + "详" * 300
        ks = FakeKS(hits=[SimpleNamespace(title="发货", content=long_content)])
        sender = FakeSender()
        cfg = IntentConfig(knowledge_max_chars=50)
        handler = self._handler(cfg=cfg, sender=sender, ks=ks)
        handled = await handler.handle(make_context("什么时候发货"), METADATA)
        self.assertTrue(handled)
        self.assertEqual(sender.sent[0][3], long_content[:50] + "…")

    async def test_consult_weak_match_degrades_to_ai(self):
        ks = FakeKS(hits=[SimpleNamespace(title="连衣裙尺码表", content="S/M/L")])
        sender = FakeSender()
        handler = self._handler(sender=sender, ks=ks)
        handled = await handler.handle(make_context("什么时候发货"), METADATA)
        self.assertFalse(handled)  # 降级给 AI
        self.assertEqual(sender.sent, [])

    async def test_consult_no_hit_degrades_to_ai(self):
        ks = FakeKS(hits=[])
        sender = FakeSender()
        handler = self._handler(sender=sender, ks=ks)
        handled = await handler.handle(make_context("什么时候发货"), METADATA)
        self.assertFalse(handled)
        self.assertEqual(sender.sent, [])

    async def test_consult_ks_error_degrades_to_ai(self):
        ks = FakeKS(error=RuntimeError("db down"))
        sender = FakeSender()
        handler = self._handler(sender=sender, ks=ks)
        handled = await handler.handle(make_context("什么时候发货"), METADATA)
        self.assertFalse(handled)

    async def test_aftersale_and_other_fall_through_to_ai(self):
        sender = FakeSender()
        handler = self._handler(sender=sender)
        self.assertFalse(await handler.handle(make_context("我要退款"), METADATA))
        self.assertFalse(await handler.handle(make_context("今天天气真好一起出去玩"), METADATA))
        self.assertEqual(sender.sent, [])

    async def test_disabled_config_returns_false(self):
        sender = FakeSender()
        handler = self._handler(cfg=IntentConfig(enabled=False), sender=sender)
        self.assertFalse(await handler.handle(make_context("我要投诉"), METADATA))
        self.assertEqual(sender.sent, [])

    async def test_missing_identity_degrades_to_ai(self):
        sender = FakeSender()
        handler = self._handler(sender=sender)
        handled = await handler.handle(make_context("在吗", with_identity=False), METADATA)
        self.assertFalse(handled)
        self.assertEqual(sender.sent, [])

    async def test_send_failure_degrades_to_ai(self):
        sender = FakeSender(send_error=RuntimeError("net down"))
        handler = self._handler(sender=sender)
        handled = await handler.handle(make_context("在吗"), METADATA)
        self.assertFalse(handled)

    async def test_empty_content_returns_false(self):
        handler = self._handler(sender=FakeSender())
        self.assertFalse(await handler.handle(make_context("   "), METADATA))


# ---------------------------------------------------------------------------
# handler_chain 挂载
# ---------------------------------------------------------------------------

class HandlerChainMountingTests(unittest.TestCase):
    def test_intent_handler_mounted_between_keyword_and_ai(self):
        import Message

        stub = SimpleNamespace(business_hours=None)
        intent_stub, ai_stub = object(), object()
        with mock.patch.object(Message, "_get_keyword_handler", return_value=stub), \
                mock.patch.object(Message, "_get_intent_handler", return_value=intent_stub), \
                mock.patch.object(Message, "create_ai_handler", return_value=ai_stub):
            handlers = Message.handler_chain(use_ai=True)
        self.assertIs(handlers[0], stub)          # 关键词优先级最高
        self.assertIs(handlers[1], intent_stub)    # 意图识别在关键词之后
        self.assertIs(handlers[2], ai_stub)        # AI 兜底在意图之后
        self.assertEqual(type(handlers[3]).__name__, "CatchAllHandler")

    def test_intent_absent_when_disabled(self):
        import Message

        stub = SimpleNamespace(business_hours=None)
        with mock.patch.object(Message, "_get_keyword_handler", return_value=stub), \
                mock.patch.object(Message, "_get_intent_handler", return_value=None):
            handlers = Message.handler_chain(use_ai=False)
        self.assertNotIn("IntentHandler", [type(h).__name__ for h in handlers])

    def test_get_intent_handler_respects_enabled_flag(self):
        import Message

        fake_off = SimpleNamespace(
            config=SimpleNamespace(
                get_model=lambda: SimpleNamespace(intent=SimpleNamespace(enabled=False))
            )
        )
        with mock.patch.dict(sys.modules, {"config": fake_off}):
            self.assertIsNone(Message._get_intent_handler())

        fake_on = SimpleNamespace(
            config=SimpleNamespace(
                get_model=lambda: SimpleNamespace(intent=SimpleNamespace(enabled=True))
            )
        )
        with mock.patch.dict(sys.modules, {"config": fake_on}):
            handler = Message._get_intent_handler()
        self.assertIsInstance(handler, IntentHandler)

    def test_create_intent_handler_factory(self):
        import Message

        handler = Message.create_intent_handler()
        self.assertIsInstance(handler, IntentHandler)


if __name__ == "__main__":
    unittest.main()
