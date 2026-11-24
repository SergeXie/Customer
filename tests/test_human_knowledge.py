"""人工回复沉淀知识库测试。

验证：
1. 买家问题与人工回复配对沉淀（title=问题，content=回复）
2. 自家发送回显抑制（AI 回复不会被沉淀成"人工知识"）
3. 转人工场景：转接通知不丢问题（窗口内问题兜底配对）
4. AI 已答过的问题不重复配对（优先取我方发言之后的新问题）
5. 客服连发多条消息合并进同一条知识
6. 问题窗口过期 / 回复过短 / 开关关闭 / 去重 都不沉淀
7. 消息链路观测钩子的类型过滤（只收纯文本问答）
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from config import KnowledgeCaptureConfig
from service.human_knowledge import HumanKnowledgeCapture


class FakeKS:
    """内存版 KnowledgeService（只实现沉淀用到的四个方法）。"""

    def __init__(self):
        self.rows = []
        self.updates = []
        self._next_id = 1

    def resolve_shop_id(self, shop_id):
        return 7

    def batch_import_customer_service(self, shop_id, rows):
        success = 0
        for row in rows:
            dup = any(
                r["shop_id"] == shop_id
                and r["title"] == row["title"]
                and r["content"] == row["content"]
                for r in self.rows
            )
            if dup:
                continue
            self.rows.append({
                "id": self._next_id, "shop_id": shop_id,
                "title": row["title"], "content": row["content"],
                "tags": row.get("tags"), "enabled": bool(row.get("enabled", True)),
            })
            self._next_id += 1
            success += 1
        return success, len(rows) - success

    def find_customer_service_id(self, shop_id, title, content):
        for r in self.rows:
            if r["shop_id"] == shop_id and r["title"] == title and r["content"] == content:
                return r["id"]
        return None

    def has_customer_service_answer(self, shop_id, title, content):
        return any(
            r["shop_id"] == shop_id and r["title"] == title and content in r["content"]
            for r in self.rows
        )

    def update_customer_service(self, cs_id, content=None, **kwargs):
        for r in self.rows:
            if r["id"] == cs_id:
                if content is not None:
                    r["content"] = content
                self.updates.append((cs_id, content))
                return r
        return None


def make_capture(**cfg_kwargs):
    cfg = KnowledgeCaptureConfig(**cfg_kwargs)
    cap = HumanKnowledgeCapture(knowledge_service=FakeKS())
    cap.config_override = cfg
    cap.now = 1000.0
    cap._clock = lambda: cap.now
    return cap


class TestPairing(unittest.TestCase):
    def test_pairs_question_with_reply(self):
        cap = make_capture()
        cap.remember_question("shop", "u1", "退货后多久退款")
        cs_id = cap.capture_reply("shop", "u1", "亲，签收后48小时内原路退回哦")
        self.assertEqual(cs_id, 1)
        row = cap._knowledge.rows[0]
        self.assertEqual(row["title"], "退货后多久退款")
        self.assertEqual(row["content"], "亲，签收后48小时内原路退回哦")
        self.assertEqual(row["tags"], "人工回复")
        self.assertTrue(row["enabled"])

    def test_multi_questions_joined_in_title(self):
        cap = make_capture()
        cap.remember_question("shop", "u1", "怎么退货")
        cap.remember_question("shop", "u1", "运费谁承担")
        cap.capture_reply("shop", "u1", "七天无理由，运费我们承担")
        self.assertEqual(cap._knowledge.rows[0]["title"], "怎么退货；运费谁承担")

    def test_no_question_skips(self):
        cap = make_capture()
        self.assertIsNone(cap.capture_reply("shop", "u1", "已经帮您备注啦"))
        self.assertEqual(cap._knowledge.rows, [])


class TestEchoSuppression(unittest.TestCase):
    def test_own_sent_text_not_captured(self):
        cap = make_capture()
        cap.remember_question("shop", "u1", "什么时候发货")
        cap.mark_sent("shop", "u1", "48小时内发货哦")
        # WebSocket 回流的自家回显
        self.assertIsNone(cap.capture_reply("shop", "u1", "48小时内发货哦"))
        self.assertEqual(cap._knowledge.rows, [])

    def test_mark_sent_does_not_drop_questions(self):
        # 转人工场景：问题 -> 转接通知([玫瑰]) -> 人工回复，问题不能丢
        cap = make_capture()
        cap.remember_question("shop", "u1", "我要退货")
        cap.mark_sent("shop", "u1", "[玫瑰]")
        cs_id = cap.capture_reply("shop", "u1", "亲，请提供订单号帮您处理")
        self.assertEqual(cs_id, 1)
        self.assertEqual(cap._knowledge.rows[0]["title"], "我要退货")

    def test_ai_answered_question_not_repicked(self):
        # AI 答过 Q1 后买家追问 Q2，人工回复只配对 Q2
        cap = make_capture()
        cap.remember_question("shop", "u1", "有现货吗")
        cap.now += 10
        cap.mark_sent("shop", "u1", "现货充足，今天可发")
        cap.now += 10
        cap.remember_question("shop", "u1", "能便宜点吗")
        cap.now += 10
        cap.capture_reply("shop", "u1", "下单自动改价，拍下联系客服改价")
        self.assertEqual(cap._knowledge.rows[0]["title"], "能便宜点吗")


class TestMerge(unittest.TestCase):
    def test_consecutive_replies_merge(self):
        cap = make_capture()
        cap.remember_question("shop", "u1", "怎么退货")
        cs_id = cap.capture_reply("shop", "u1", "亲，七天无理由退货")
        self.assertEqual(cs_id, 1)
        cap.now += 20  # 合并窗口内（60s）
        cs_id2 = cap.capture_reply("shop", "u1", "退回地址私信您了哦")
        self.assertEqual(cs_id2, 1)
        self.assertEqual(len(cap._knowledge.rows), 1)
        self.assertEqual(cap._knowledge.rows[0]["content"], "亲，七天无理由退货\n退回地址私信您了哦")

    def test_merge_window_expiry_starts_new_pairing(self):
        cap = make_capture()
        cap.remember_question("shop", "u1", "怎么退货")
        cap.capture_reply("shop", "u1", "七天无理由退货")
        cap.now += 300  # 超出合并窗口
        self.assertIsNone(cap.capture_reply("shop", "u1", "还有什么可以帮您"))
        self.assertEqual(len(cap._knowledge.rows), 1)


class TestFiltersAndToggle(unittest.TestCase):
    def test_question_window_expiry(self):
        cap = make_capture(question_window_seconds=600)
        cap.remember_question("shop", "u1", "怎么退货")
        cap.now += 601
        self.assertIsNone(cap.capture_reply("shop", "u1", "七天无理由退货哦"))
        self.assertEqual(cap._knowledge.rows, [])

    def test_short_reply_filtered(self):
        cap = make_capture(min_reply_chars=4)
        cap.remember_question("shop", "u1", "在吗")
        self.assertIsNone(cap.capture_reply("shop", "u1", "在的"))
        self.assertEqual(cap._knowledge.rows, [])

    def test_disabled_capture(self):
        cap = make_capture(enabled=False)
        cap.remember_question("shop", "u1", "怎么退货")
        self.assertIsNone(cap.capture_reply("shop", "u1", "七天无理由退货哦"))
        self.assertEqual(cap._knowledge.rows, [])

    def test_default_enabled_false_stores_draft(self):
        cap = make_capture(default_enabled=False)
        cap.remember_question("shop", "u1", "怎么退货")
        cap.capture_reply("shop", "u1", "七天无理由退货哦")
        self.assertFalse(cap._knowledge.rows[0]["enabled"])

    def test_duplicate_not_captured_twice(self):
        cap = make_capture()
        cap.remember_question("shop", "u1", "怎么退货")
        cap.capture_reply("shop", "u1", "七天无理由退货哦")
        # 另一会话问出同样的问题、人工给出同样回答 -> 去重
        cap.remember_question("shop", "u2", "怎么退货")
        self.assertIsNone(cap.capture_reply("shop", "u2", "七天无理由退货哦"))
        self.assertEqual(len(cap._knowledge.rows), 1)

    def test_merged_content_still_deduped(self):
        # 原条目被连发合并拉长内容后，同一问答再来也要被去重（包含式判重）
        cap = make_capture()
        cap.remember_question("shop", "u1", "怎么退货")
        cap.capture_reply("shop", "u1", "七天无理由退货哦")
        cap.now += 10
        cap.capture_reply("shop", "u1", "退回运费我们承担")  # 并入同一条
        cap.now += 10
        cap.remember_question("shop", "u2", "怎么退货")
        self.assertIsNone(cap.capture_reply("shop", "u2", "七天无理由退货哦"))
        self.assertEqual(len(cap._knowledge.rows), 1)

    def test_conversation_isolation(self):
        cap = make_capture()
        cap.remember_question("shop", "u1", "怎么退货")
        cap.capture_reply("shop", "u2", "七天无理由退货哦")  # u2 没有提问
        self.assertEqual(cap._knowledge.rows, [])


class TestWiringFilters(unittest.TestCase):
    """消息链路钩子的类型过滤（pdd_message_handler 观测入口）。"""

    def _handler(self):
        from Channel.pinduoduo.core.pdd_message_handler import MessageHandlerMixin
        h = MessageHandlerMixin.__new__(MessageHandlerMixin)
        h.logger = mock.MagicMock()
        return h

    def _pm(self, origin, pdd_type, from_uid="u1", to_uid=""):
        return SimpleNamespace(origin=origin, pdd_type=pdd_type,
                               from_uid=from_uid, to_uid=to_uid)

    def test_question_hook_accepts_customer_text_only(self):
        from Channel.pinduoduo.message_rules import Origin, PDDMsgType
        from bridge.context import ContextType
        h = self._handler()
        fake = mock.MagicMock()
        import service.human_knowledge as hk_mod
        with mock.patch.object(hk_mod, "human_knowledge", fake):
            # 买家纯文本 -> 记录
            ctx = SimpleNamespace(content="怎么退货", type=ContextType.TEXT)
            h._observe_question(ctx, "shop", self._pm(Origin.CUSTOMER, PDDMsgType.TEXT))
            # 客服侧文本 -> 不记录
            h._observe_question(ctx, "shop", self._pm(Origin.MERCHANT, PDDMsgType.TEXT))
            # 图片 -> 不记录
            h._observe_question(ctx, "shop", self._pm(Origin.CUSTOMER, PDDMsgType.IMAGE))
            # 结构化内容 -> 不记录
            ctx2 = SimpleNamespace(content={"goods_id": 1}, type=ContextType.GOODS_INQUIRY)
            h._observe_question(ctx2, "shop", self._pm(Origin.CUSTOMER, PDDMsgType.TEXT))
        fake.remember_question.assert_called_once_with("shop", "u1", "怎么退货")

    def test_reply_hook_accepts_merchant_mallcs_text_only(self):
        from Channel.pinduoduo.message_rules import Origin, PDDMsgType
        from bridge.context import ContextType
        h = self._handler()
        fake = mock.MagicMock()
        fake.capture_reply = mock.MagicMock(return_value=None)
        import service.human_knowledge as hk_mod
        with mock.patch.object(hk_mod, "human_knowledge", fake):
            # 客服侧纯文本 -> 捕获（to_uid 是买家）
            ctx = SimpleNamespace(content="48小时发货", type=ContextType.MALL_CS)
            asyncio.run(h._observe_human_reply(
                ctx, "shop", self._pm(Origin.MERCHANT, PDDMsgType.TEXT, to_uid="u1")))
            # 买家文本 -> 不捕获
            asyncio.run(h._observe_human_reply(
                ctx, "shop", self._pm(Origin.CUSTOMER, PDDMsgType.TEXT)))
            # 商品卡 -> 不捕获
            ctx2 = SimpleNamespace(content="{}", type=ContextType.GOODS_CARD)
            asyncio.run(h._observe_human_reply(
                ctx2, "shop", self._pm(Origin.MERCHANT, PDDMsgType.TEXT, to_uid="u1")))
            # 客服侧图片 -> 不捕获
            asyncio.run(h._observe_human_reply(
                ctx, "shop", self._pm(Origin.MERCHANT, PDDMsgType.IMAGE, to_uid="u1")))
        fake.capture_reply.assert_called_once_with("shop", "u1", "48小时发货")


if __name__ == "__main__":
    unittest.main()
