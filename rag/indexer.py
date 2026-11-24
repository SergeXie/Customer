"""
rag.indexer —— 知识入库与索引维护

把知识记录（产品知识 / 客服知识）切分、向量化，双路写入
Milvus（向量）+ BM25（倒排），并在检索时委托 HybridRetriever。

关键性质：
- 幂等：同一条知识重复索引 = 覆盖（先 delete_by_ref 再写入）；
- 降级安全：向量化失败只影响向量路（BM25 照常），单路可用；
- 可重建：SQLite 是事实源，rebuild_all 随时全量重建两个索引。
"""
from typing import Iterable, List, Optional

from utils.logger_loguru import get_logger

from .bm25 import BM25Index
from .chunker import chunk_text
from .embeddings import EmbeddingProvider
from .hybrid import HybridRetriever, NoopReranker, NoopRewriter
from .models import DOC_CS, DOC_PRODUCT, Chunk, ScoredChunk
from .vector_store import MilvusStore

logger = get_logger("RagIndexer")


class KnowledgeIndexer:
    """知识索引维护 + 检索入口。"""

    def __init__(
        self,
        store: MilvusStore,
        bm25: BM25Index,
        embedder: EmbeddingProvider,
        *,
        top_k_vector: int = 30,
        top_k_bm25: int = 30,
        rrf_k: int = 60,
    ):
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.retriever = HybridRetriever(
            store,
            bm25,
            embedder,
            top_k_vector=top_k_vector,
            top_k_bm25=top_k_bm25,
            rrf_k=rrf_k,
            rewriter=NoopRewriter(),   # 第 5 步 Query 改写：接口已留，默认不改写
            reranker=NoopReranker(),   # 第 7 步 Rerank：接口已留，默认不精排
        )
        self._bm25_loaded = False

    # ---------- 索引维护 ----------

    def index_chunks(self, chunks: List[Chunk]):
        """切分结果双路入库。向量路失败仅记日志（BM25 保证基础召回）。"""
        if not chunks:
            return
        doc_type, ref_id = chunks[0].doc_type, chunks[0].ref_id
        self.store.delete_by_ref(doc_type, ref_id)
        self.bm25.remove_ref(doc_type, ref_id)

        try:
            vectors = self.embedder.embed([c.text for c in chunks])
            for c, v in zip(chunks, vectors):
                c.embedding = v
            self.store.upsert_chunks(chunks)
        except Exception as e:
            logger.warning(f"向量化入库失败（BM25 不受影响）: {e}")
        for c in chunks:
            self.bm25.add_chunk(c)

    def index_customer_service(self, cs_id: int, shop_id: int, title: str, content: str):
        text = f"{title}\n{content}" if title and title not in (content or "") else (content or title or "")
        chunks = chunk_text(
            text,
            doc_type=DOC_CS,
            ref_id=int(cs_id),
            shop_id=int(shop_id),
            doc_title=title or "客服知识",
        )
        self.index_chunks(chunks)

    def index_product(
        self,
        product_id: int,
        shop_id: int,
        goods_name: str,
        extracted_content: str = "",
        specifications: str = "",
    ):
        body = "\n".join(p for p in (extracted_content or "", specifications or "") if p and p.strip())
        text = f"{goods_name}\n{body}" if body else (goods_name or "")
        chunks = chunk_text(
            text,
            doc_type=DOC_PRODUCT,
            ref_id=int(product_id),
            shop_id=int(shop_id),
            doc_title=goods_name or "产品知识",
        )
        self.index_chunks(chunks)

    def remove(self, doc_type: str, ref_id: int):
        self.store.delete_by_ref(doc_type, int(ref_id))
        self.bm25.remove_ref(doc_type, int(ref_id))

    def rebuild_all(self, products: Iterable, cs_items: Iterable) -> int:
        """
        全量重建。products / cs_items 为可迭代记录（需含索引所需字段），
        由调用方（KnowledgeService）从 SQLite 读出传入。
        """
        self.store.clear()  # 行级清空；不能 drop_collection（与后台建索引竞争会崩）
        self.bm25.rebuild([])
        self._bm25_loaded = True
        count = 0
        for p in products:
            self.index_product(
                product_id=p["id"],
                shop_id=p["shop_id"],
                goods_name=p.get("goods_name") or "",
                extracted_content=p.get("extracted_content") or "",
                specifications=p.get("specifications") or "",
            )
            count += 1
        for cs in cs_items:
            self.index_customer_service(
                cs_id=cs["id"],
                shop_id=cs["shop_id"],
                title=cs.get("title") or "",
                content=cs.get("content") or "",
            )
            count += 1
        logger.info(f"知识索引全量重建完成: {count} 条")
        return count

    # ---------- 检索 ----------

    def ensure_bm25_loaded(self, chunks_loader=None):
        """
        懒加载：BM25 是内存索引，进程重启后从 Milvus 恢复
        （Milvus 持久化，SQLite 为事实源，indexer 保证两者同步）。
        """
        if self._bm25_loaded:
            return
        try:
            chunks = self.store.list_chunks()
            self.bm25.rebuild(chunks)
        except Exception as e:
            logger.warning(f"BM25 从 Milvus 恢复失败（将按需重建）: {e}")
        self._bm25_loaded = True

    def search(
        self,
        query: str,
        shop_id: int,
        doc_type: Optional[str] = None,
        top_k: int = 8,
        history: Optional[List[str]] = None,
    ) -> List[ScoredChunk]:
        self.ensure_bm25_loaded()
        return self.retriever.retrieve(
            query, shop_id=int(shop_id), doc_type=doc_type, top_k=top_k, history=history
        )
