"""
rag.models —— RAG 链路的核心数据结构

与具体存储/框架解耦：indexer 把 ORM 记录适配成 Chunk，
vector_store / bm25 / hybrid 围绕 Chunk / ScoredChunk 协作。
"""
from dataclasses import dataclass, field
from typing import Optional

# 文档类型常量（同时用作 Milvus 过滤字段值与 chunk id 前缀）
DOC_PRODUCT = "product"  # 产品知识（product_knowledge 表）
DOC_CS = "cs"            # 客服知识（customer_service_knowledge 表）


@dataclass
class Chunk:
    """一个语义分块。"""
    chunk_id: str                 # 全局唯一，格式 "{doc_type}:{ref_id}:{chunk_idx}"
    doc_type: str                 # DOC_PRODUCT / DOC_CS
    ref_id: int                   # 对应知识记录的主键（ProductKnowledge.id / CustomerServiceKnowledge.id）
    shop_id: int                  # 店铺内部 id（shops.id），用于检索过滤
    chunk_idx: int                # 在原文档内的分块序号
    breadcrumb: str               # 面包屑：文档 > 章节
    text: str                     # 分块正文
    token_count: int = 0          # 分块 token 数（入库时填）
    embedding: Optional[list] = field(default=None, repr=False)  # 向量（embedder 填充）


@dataclass
class ScoredChunk:
    """带检索得分的分块。"""
    chunk: Chunk
    score: float                  # 融合得分（RRF 分数或向量相似度，越大越相关）
    source: str = ""              # 命中来源：vector / bm25 / hybrid（调试用）


@dataclass
class Citation:
    """上下文引用编号：正文里的 [n] 指向这里。"""
    index: int
    doc_type: str
    ref_id: int
    breadcrumb: str
    snippet: str
