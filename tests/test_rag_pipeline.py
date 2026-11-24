"""RAG 核心链路测试：语义切分 / BM25 / RRF / Milvus 存储 / 混合召回 / 引用组装。

约束验证：
1. 切分不切断表格代码块，带面包屑与重叠
2. RRF 融合给双路共识加分
3. Milvus Lite 在 ASCII 路径下完整走通 upsert/search/delete
4. 引用编号 [1][2] 与来源一一对应，token 预算生效
"""

import os
import shutil
import tempfile
import unittest

from rag.bm25 import BM25Index
from rag.chunker import chunk_text
from rag.context_builder import build_context
from rag.embeddings import HashEmbedder
from rag.hybrid import rrf_fuse
from rag.indexer import KnowledgeIndexer
from rag.models import DOC_CS, DOC_PRODUCT, Chunk, ScoredChunk
from rag.vector_store import MilvusStore


class TestChunker(unittest.TestCase):
    def test_heading_sections_and_breadcrumb(self):
        text = (
            "退款政策\n"
            "退货签收后 48 小时内退款到账。\n"
            "物流政策\n"
            "默认发顺丰，48 小时内发货。\n"
        )
        chunks = chunk_text(text, doc_type=DOC_CS, ref_id=1, shop_id=1, doc_title="售后手册")
        self.assertGreaterEqual(len(chunks), 1)
        joined = " ".join(c.breadcrumb for c in chunks)
        self.assertIn("售后手册", joined)
        # 标题应进入面包屑
        self.assertTrue("退款政策" in joined or "物流政策" in joined)
        # 内容不丢失
        all_text = "".join(c.text for c in chunks)
        self.assertIn("48 小时内退款", all_text)
        self.assertIn("顺丰", all_text)

    def test_table_not_split(self):
        lines = ["规格表"] + [f"| 行{i} | 值{i} |" for i in range(10)]
        chunks = chunk_text("\n".join(lines), doc_type=DOC_PRODUCT, ref_id=2, shop_id=1, doc_title="商品A")
        table_content = "".join(c.text for c in chunks)
        for i in range(10):
            self.assertIn(f"行{i}", table_content)

    def test_chunk_size_bounds_and_ids(self):
        text = "这是一句测试。" * 200
        chunks = chunk_text(text, doc_type=DOC_CS, ref_id=9, shop_id=3, doc_title="长文")
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertTrue(c.chunk_id.startswith("cs:9:"))
            self.assertEqual(c.shop_id, 3)
        # 分块大小被限制在合理范围（允许 1.5 倍容差）
        for c in chunks[:-1]:
            self.assertLessEqual(c.token_count, 384 * 1.6)

    def test_empty_text(self):
        self.assertEqual(chunk_text("", doc_type=DOC_CS, ref_id=1, shop_id=1), [])


class TestBM25(unittest.TestCase):
    def _chunk(self, cid, text, ref_id=1):
        return Chunk(
            chunk_id=cid, doc_type=DOC_CS, ref_id=ref_id, shop_id=1,
            chunk_idx=0, breadcrumb="b", text=text,
        )

    def test_search_ranking(self):
        idx = BM25Index()
        idx.add_chunk(self._chunk("cs:1:0", "退款政策 退货签收后 48 小时内退款"))
        idx.add_chunk(self._chunk("cs:2:0", "物流政策 默认发顺丰 48 小时内发货", ref_id=2))
        hits = idx.search("退款多久到账", top_k=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.chunk_id, "cs:1:0")

    def test_remove_ref(self):
        idx = BM25Index()
        idx.add_chunk(self._chunk("cs:1:0", "退款政策"))
        idx.add_chunk(self._chunk("cs:1:1", "退款流程", ref_id=1))
        idx.remove_ref(DOC_CS, 1)
        self.assertEqual(len(idx), 0)
        self.assertEqual(idx.search("退款"), [])

    def test_shop_filter(self):
        idx = BM25Index()
        c = self._chunk("cs:1:0", "仅限店铺 5 的知识")
        c.shop_id = 5
        idx.add_chunk(c)
        self.assertEqual(len(idx.search("知识", shop_id=5)), 1)
        self.assertEqual(idx.search("知识", shop_id=99), [])


class TestRRF(unittest.TestCase):
    def test_dual_hit_wins(self):
        def mk(cid, score, source):
            return ScoredChunk(
                chunk=Chunk(chunk_id=cid, doc_type=DOC_CS, ref_id=1, shop_id=1,
                            chunk_idx=0, breadcrumb="b", text=cid),
                score=score, source=source,
            )
        # "both" 双路命中（各自排名靠前），"only" 单路第一
        vec = [mk("both", 0.9, "vector"), mk("only", 0.99, "vector")]
        bm25 = [mk("both", 5.0, "bm25"), mk("only", 0.1, "bm25")]
        fused = rrf_fuse([vec, bm25], rrf_k=60)
        self.assertEqual(fused[0].chunk.chunk_id, "both")
        self.assertIn("vector", fused[0].source)
        self.assertIn("bm25", fused[0].source)


class TestHashEmbedder(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        e = HashEmbedder(dim=64)
        v1 = e.embed(["退款政策"])[0]
        v2 = e.embed(["退款政策"])[0]
        self.assertEqual(v1, v2)
        self.assertEqual(len(v1), 64)
        norm = sum(x * x for x in v1) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_similar_text_scores_higher(self):
        e = HashEmbedder(dim=256)
        a, b, c = e.embed(["退款政策 退货签收后退款", "退款 多久到账", "物流 发顺丰"])[0:3]
        dot = lambda x, y: sum(i * j for i, j in zip(x, y))
        self.assertGreater(dot(a, b), dot(a, c))


class TestMilvusStore(unittest.TestCase):
    """Milvus Lite 端到端（ASCII 临时路径）。"""

    def setUp(self):
        self.path = os.path.join(tempfile.gettempdir(), "rag_test_milvus.db")
        shutil.rmtree(self.path, ignore_errors=True)
        self.store = MilvusStore(path=self.path, dim=32)

    def tearDown(self):
        self.store.drop_all()

    def _chunk(self, cid, ref_id, text, doc_type=DOC_CS):
        emb = HashEmbedder(dim=32).embed([text])[0]
        return Chunk(chunk_id=cid, doc_type=doc_type, ref_id=ref_id, shop_id=1,
                     chunk_idx=0, breadcrumb="b", text=text, embedding=emb)

    def test_upsert_search_delete(self):
        c1 = self._chunk("cs:1:0", 1, "退款政策 48 小时到账")
        c2 = self._chunk("cs:2:0", 2, "物流政策 顺丰发货")
        self.assertEqual(self.store.upsert_chunks([c1, c2]), 2)

        vec = HashEmbedder(dim=32).embed(["退款到账时间"])[0]
        hits = self.store.search(shop_id=1, vector=vec, top_k=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.chunk_id, "cs:1:0")

        # doc_type 过滤
        hits_p = self.store.search(shop_id=1, vector=vec, top_k=5, doc_type=DOC_PRODUCT)
        self.assertEqual(hits_p, [])

        # 删除后不可见
        self.store.delete_by_ref(DOC_CS, 1)
        hits2 = self.store.search(shop_id=1, vector=vec, top_k=5)
        self.assertTrue(all(h.chunk.ref_id != 1 for h in hits2))

    def test_non_ascii_path_falls_back(self):
        s = MilvusStore(path=r"D:\临时目录\milvus", dim=32)
        self.assertNotIn("临时", s.path)


class TestIndexerAndHybrid(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.gettempdir(), "rag_test_indexer.db")
        shutil.rmtree(self.path, ignore_errors=True)
        self.store = MilvusStore(path=self.path, dim=32)
        self.bm25 = BM25Index()
        self.embedder = HashEmbedder(dim=32)
        self.indexer = KnowledgeIndexer(self.store, self.bm25, self.embedder)

    def tearDown(self):
        self.store.drop_all()

    def test_index_and_hybrid_search(self):
        self.indexer.index_customer_service(1, 1, "退款政策", "退货签收后 48 小时内退款到账，原路返回。")
        self.indexer.index_customer_service(2, 1, "物流政策", "默认发顺丰，48 小时内发货。")
        self.indexer.index_product(3, 1, "维生素C", "每片含维生素C 100mg，每日一片。")

        # 向量入库必须真实成功（防止 upsert 静默失败靠 BM25 兜底混过断言）
        self.assertEqual(self.store.count(), 3)

        hits = self.indexer.search("退款多久到账", shop_id=1, top_k=3)
        self.assertGreaterEqual(len(hits), 1)
        top = hits[0]
        self.assertEqual(top.chunk.doc_type, DOC_CS)
        self.assertEqual(top.chunk.ref_id, 1)
        self.assertIn("退款", top.chunk.breadcrumb)

        # 产品检索走 doc_type 过滤
        p_hits = self.indexer.search("维生素C 含量", shop_id=1, doc_type=DOC_PRODUCT, top_k=3)
        self.assertGreaterEqual(len(p_hits), 1)
        self.assertEqual(p_hits[0].chunk.doc_type, DOC_PRODUCT)

        # 重建后仍可检索（BM25 从 Milvus 恢复）
        chunks = self.store.list_chunks()
        self.bm25.rebuild(chunks)
        hits2 = self.indexer.search("退款多久到账", shop_id=1, top_k=3)
        self.assertEqual(hits2[0].chunk.ref_id, 1)

    def test_remove(self):
        self.indexer.index_customer_service(1, 1, "退款政策", "48 小时退款")
        self.indexer.remove(DOC_CS, 1)
        self.assertEqual(self.indexer.search("退款", shop_id=1), [])


class TestContextBuilder(unittest.TestCase):
    def _hit(self, cid, ref_id, text, score):
        return ScoredChunk(
            chunk=Chunk(chunk_id=cid, doc_type=DOC_CS, ref_id=ref_id, shop_id=1,
                        chunk_idx=0, breadcrumb=f"手册 > 第{ref_id}节", text=text),
            score=score,
        )

    def test_citations_and_dedupe(self):
        hits = [
            self._hit("cs:1:0", 1, "退货签收后 48 小时内退款到账", 0.9),
            self._hit("cs:1:1", 1, "退货签收后 48 小时内退款到账", 0.8),  # 完全重复
            self._hit("cs:2:0", 2, "默认发顺丰，48 小时内发货", 0.7),
        ]
        text, citations = build_context(hits)
        self.assertEqual(len(citations), 2)
        self.assertIn("[1]", text)
        self.assertIn("[2]", text)
        self.assertEqual(citations[0].index, 1)
        self.assertEqual(citations[0].ref_id, 1)
        self.assertIn("第1节", citations[0].breadcrumb)

    def test_token_budget(self):
        hits = [
            self._hit(f"cs:{i}:0", i, "很详细的内容。" * 50, score=1.0 - i * 0.01)
            for i in range(1, 10)
        ]
        text, citations = build_context(hits, budget=200)
        self.assertLess(len(citations), 9)
        self.assertGreaterEqual(len(citations), 1)

    def test_empty(self):
        text, citations = build_context([])
        self.assertEqual(text, "")
        self.assertEqual(citations, [])


if __name__ == "__main__":
    unittest.main()
