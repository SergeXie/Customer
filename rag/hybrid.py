"""
rag.hybrid —— 混合召回与融合

对应第 6 步「混合召回」+ 第 7 步前置：
- 向量（Milvus）top-N 与 BM25 top-N 各取一路；
- RRF 融合（score = Σ 1/(k + rank)，默认 k=60），过滤（shop_id/doc_type）前置；
- Query 改写（第 5 步）与 Rerank 精排（第 7 步）留接口：
  QueryRewriter / Reranker 协议 + Noop 默认实现，后续可替换为
  多轮指代消解 / cross-encoder 而不动本模块。
"""
from typing import List, Optional, Protocol, runtime_checkable

from utils.logger_loguru import get_logger

from .bm25 import BM25Index
from .embeddings import EmbeddingProvider
from .models import ScoredChunk
from .vector_store import MilvusStore

logger = get_logger("RagHybrid")


@runtime_checkable
class QueryRewriter(Protocol):
    def rewrite(self, query: str, history: Optional[List[str]] = None) -> str:
        """把用户问题改写成独立、完整的检索查询。"""
        ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: List[ScoredChunk], top_k: int) -> List[ScoredChunk]:
        """对候选精排，返回 top_k。"""
        ...


class NoopRewriter:
    """默认：不改写，原句即查询。"""

    def rewrite(self, query: str, history: Optional[List[str]] = None) -> str:
        return query


class NoopReranker:
    """默认：不精排，按融合得分取 top_k。"""

    def rerank(self, query: str, candidates: List[ScoredChunk], top_k: int) -> List[ScoredChunk]:
        return candidates[:top_k]


def rrf_fuse(rankings: List[List[ScoredChunk]], rrf_k: int = 60) -> List[ScoredChunk]:
    """
    Reciprocal Rank Fusion：
    score(d) = Σ_lists 1 / (rrf_k + rank_list(d))
    同一 chunk 命中多路时得分叠加（自然奖励双路共识）。
    """
    best: dict = {}
    for ranked in rankings:
        for rank, sc in enumerate(ranked, start=1):
            cid = sc.chunk.chunk_id
            fused = 1.0 / (rrf_k + rank)
            entry = best.get(cid)
            if entry is None:
                best[cid] = ScoredChunk(chunk=sc.chunk, score=fused, source=sc.source)
            else:
                entry.score += fused
                if sc.source not in entry.source.split("+"):
                    entry.source = f"{entry.source}+{sc.source}" if entry.source else sc.source
    return sorted(best.values(), key=lambda s: s.score, reverse=True)


class HybridRetriever:
    """向量 + BM25 双路召回，RRF 融合，接口预留改写与精排。"""

    def __init__(
        self,
        store: MilvusStore,
        bm25: BM25Index,
        embedder: EmbeddingProvider,
        *,
        top_k_vector: int = 30,
        top_k_bm25: int = 30,
        rrf_k: int = 60,
        rewriter: Optional[QueryRewriter] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.top_k_vector = max(1, int(top_k_vector))
        self.top_k_bm25 = max(1, int(top_k_bm25))
        self.rrf_k = max(1, int(rrf_k))
        self.rewriter = rewriter or NoopRewriter()
        self.reranker = reranker or NoopReranker()

    def retrieve(
        self,
        query: str,
        shop_id: int,
        doc_type: Optional[str] = None,
        top_k: int = 8,
        history: Optional[List[str]] = None,
    ) -> List[ScoredChunk]:
        query = (self.rewriter.rewrite(query, history) or query).strip()
        if not query:
            return []

        rankings: List[List[ScoredChunk]] = []

        # 向量路（失败降级为 BM25 单路，绝不阻断）
        try:
            vec = self.embedder.embed([query])[0]
            v_hits = self.store.search(shop_id, vec, top_k=self.top_k_vector, doc_type=doc_type)
            if v_hits:
                rankings.append(v_hits)
        except Exception as e:
            logger.warning(f"向量召回失败，仅用 BM25: {e}")

        # BM25 路
        try:
            b_hits = self.bm25.search(query, top_k=self.top_k_bm25, shop_id=shop_id, doc_type=doc_type)
            if b_hits:
                rankings.append(b_hits)
        except Exception as e:
            logger.warning(f"BM25 召回失败: {e}")

        if not rankings:
            return []

        fused = rrf_fuse(rankings, rrf_k=self.rrf_k)
        return self.reranker.rerank(query, fused, top_k)
