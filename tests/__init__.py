"""Regression tests for account isolation and message processing."""

# 测试隔离：禁止 KnowledgeService 的 RAG 索引钩子写全局 Milvus 库，
# 避免测试数据泄漏到用户数据目录（RAG 组件本身由 tests/test_rag_pipeline.py 直接覆盖）。
import os

os.environ.setdefault("AGENT_CUSTOMER_RAG_DISABLED", "1")
