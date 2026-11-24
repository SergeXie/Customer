"""
rag.bm25 —— BM25 倒排索引（内存态）

对应「BM25 倒排」召回路（图中的 ES 角色，本地化为纯 Python 实现）：
- jieba 分词建倒排，标准 BM25（k1=1.5, b=0.75）；
- 与 Milvus 中的 chunk 粒度一致，同一份 Chunk 双路入库；
- SQLite 是唯一事实源，本索引可随时从 Milvus/SQLite 全量重建，无需持久化。
"""
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .models import Chunk, ScoredChunk


def _tokenize(text: str) -> List[str]:
    try:
        import jieba
        return [w.strip() for w in jieba.lcut(text or "") if w.strip()]
    except Exception:
        return [ch for ch in (text or "") if not ch.isspace()]


class BM25Index:
    """分块级 BM25 索引，支持按 ref 增删（知识增删改时增量维护）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: Dict[str, Chunk] = {}            # chunk_id -> Chunk
        self._doc_len: Dict[str, int] = {}           # chunk_id -> token 数
        self._postings: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {chunk_id: tf}
        self._ref_to_ids: Dict[Tuple[str, int], set] = defaultdict(set)

    # ---------- 增删 ----------

    def add_chunk(self, chunk: Chunk):
        self.remove_chunk(chunk.chunk_id)
        tokens = _tokenize(chunk.text)
        self._docs[chunk.chunk_id] = chunk
        self._doc_len[chunk.chunk_id] = max(1, len(tokens))
        for term, tf in Counter(tokens).items():
            self._postings[term][chunk.chunk_id] = tf
        self._ref_to_ids[(chunk.doc_type, chunk.ref_id)].add(chunk.chunk_id)

    def remove_chunk(self, chunk_id: str):
        chunk = self._docs.pop(chunk_id, None)
        if chunk is None:
            return
        self._doc_len.pop(chunk_id, None)
        self._ref_to_ids[(chunk.doc_type, chunk.ref_id)].discard(chunk_id)
        # 从倒排中清除（惰性清理 term 空项）
        empty_terms = []
        for term, posting in self._postings.items():
            posting.pop(chunk_id, None)
            if not posting:
                empty_terms.append(term)
        for term in empty_terms:
            self._postings.pop(term, None)

    def remove_ref(self, doc_type: str, ref_id: int):
        for cid in list(self._ref_to_ids.get((doc_type, ref_id), ())):
            self.remove_chunk(cid)
        self._ref_to_ids.pop((doc_type, ref_id), None)

    def rebuild(self, chunks: List[Chunk]):
        self._docs.clear()
        self._doc_len.clear()
        self._postings.clear()
        self._ref_to_ids.clear()
        for c in chunks:
            self.add_chunk(c)

    def __len__(self):
        return len(self._docs)

    # ---------- 检索 ----------

    def search(
        self,
        query: str,
        top_k: int = 30,
        shop_id: Optional[int] = None,
        doc_type: Optional[str] = None,
    ) -> List[ScoredChunk]:
        q_terms = _tokenize(query)
        if not q_terms or not self._docs:
            return []
        n = len(self._docs)
        avgdl = sum(self._doc_len.values()) / max(1, n)
        scores: Dict[str, float] = defaultdict(float)
        for term in q_terms:
            posting = self._postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for cid, tf in posting.items():
                chunk = self._docs[cid]
                if shop_id is not None and chunk.shop_id != shop_id:
                    continue
                if doc_type and chunk.doc_type != doc_type:
                    continue
                dl = self._doc_len[cid]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[cid] += idf * tf * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max(1, top_k)]
        return [ScoredChunk(chunk=self._docs[cid], score=float(s), source="bm25") for cid, s in ranked]
