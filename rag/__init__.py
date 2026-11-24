"""
rag 包：知识库 RAG 检索核心链路
================================

对应「检索链路九步」中的核心环节（本地化落地）：

    知识入库 → 语义切分(chunker) → 向量化(embeddings) → Milvus 索引(vector_store)
                                                          + BM25 倒排(bm25)
    查询 → 混合召回 + RRF 融合(hybrid) → 上下文组装 + 引用编号(context_builder)

Query 理解改写 / Rerank 精排 / 生成引用校验 留有接口（QueryRewriter / Reranker），
默认 Noop 实现，后续可无侵入替换。

设计约束：
- 向量库只用 Milvus（Milvus Lite 本地文件模式），代码不直接依赖 faiss；
- Milvus 数据目录必须是纯 ASCII 路径（faiss 索引持久化在非 ASCII 路径下会失败），
  默认 %LOCALAPPDATA%\\AgentCustomer\\milvus；
- 任何 RAG 组件失败都不得阻断业务：调用方（KnowledgeService）始终可以
  回退到旧的关键词检索。
"""
from .models import Chunk, ScoredChunk, Citation, DOC_PRODUCT, DOC_CS
from .tokens import count_tokens
from .chunker import chunk_text
from .embeddings import EmbeddingProvider, HashEmbedder, LiteLLMEmbedder, build_embedder
from .vector_store import MilvusStore, default_store_path
from .bm25 import BM25Index
from .hybrid import (
    HybridRetriever,
    QueryRewriter,
    NoopRewriter,
    Reranker,
    NoopReranker,
)
from .context_builder import build_context
from .indexer import KnowledgeIndexer

__all__ = [
    "Chunk",
    "ScoredChunk",
    "Citation",
    "DOC_PRODUCT",
    "DOC_CS",
    "count_tokens",
    "chunk_text",
    "EmbeddingProvider",
    "HashEmbedder",
    "LiteLLMEmbedder",
    "build_embedder",
    "MilvusStore",
    "default_store_path",
    "BM25Index",
    "HybridRetriever",
    "QueryRewriter",
    "NoopRewriter",
    "Reranker",
    "NoopReranker",
    "build_context",
    "KnowledgeIndexer",
]
