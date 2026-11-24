"""
rag.vector_store —— Milvus 向量存储

向量库只用 Milvus（Milvus Lite 本地文件模式，pymilvus MilvusClient），
本模块不依赖任何其他向量库实现。

⚠ 路径约束（Windows）：Milvus 数据目录必须是纯 ASCII 路径。
  milvus-lite 内部的索引持久化在含中文等非 ASCII 字符的路径下会写文件失败
  （其底层 C++ IO 按 ANSI 代码页打开文件）。因此默认把数据放到
  %LOCALAPPDATA%\\AgentCustomer\\milvus；若配置了非 ASCII 路径会被拒绝并回退默认值。

集合结构（knowledge_chunks）：
    id          VARCHAR 主键   "{doc_type}:{ref_id}:{chunk_idx}"
    shop_id     INT64         店铺内部 id（检索过滤）
    doc_type    VARCHAR       product / cs（检索过滤）
    ref_id      INT64         知识记录主键
    chunk_idx   INT32         分块序号
    breadcrumb  VARCHAR(512)  面包屑
    text        VARCHAR(8192) 分块正文
    token_count INT32
    embedding   FLOAT_VECTOR  （L2 归一化，IP 度量 ≈ 余弦）
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional

from utils.logger_loguru import get_logger

from .models import Chunk, ScoredChunk

logger = get_logger("RagVectorStore")

COLLECTION = "knowledge_chunks"
_TEXT_MAX = 8192
_BREADCRUMB_MAX = 512

# 向量索引类型：FLAT（精确检索）。AUTOINDEX 在 milvus-lite 3.2.2 映射到 HNSW，
# 其 build/load 在 Windows 端到端实测触发原生段错误；桌面级数据量 FLAT 足够快。
# 该值会写入哨兵文件，变更时自动重建索引（迁移保护）。
INDEX_TYPE = "FLAT"


def default_store_path() -> str:
    """默认 Milvus 数据目录：纯 ASCII，且以 .db 结尾（MilvusClient 本地库要求）。"""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return str(Path(local) / "AgentCustomer" / "milvus.db")
    return str(Path.home() / ".local" / "share" / "agent-customer" / "milvus.db")


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


class MilvusStore:
    """MilvusClient 的薄封装：建表 / upsert / 删除 / 向量检索。"""

    def __init__(self, path: str = "", dim: int = 1024):
        self.dim = max(16, int(dim))
        chosen = (path or "").strip() or default_store_path()
        if not _is_ascii(chosen):
            fallback = default_store_path()
            logger.warning(
                f"Milvus 数据目录含非 ASCII 字符，索引持久化会失败，已回退默认目录: "
                f"{chosen} -> {fallback}"
            )
            chosen = fallback
        # MilvusClient 本地库路径必须以 .db 结尾（URI 除外）
        if not chosen.startswith(("http://", "https://", "unix://", "tcp://")):
            if not chosen.lower().endswith(".db"):
                chosen = chosen + ".db"
        self.path = chosen
        self._client = None

    # ---------- 生命周期 ----------

    def _sentinel_path(self) -> Path:
        p = Path(self.path)
        return p.parent / (p.name + ".clean")

    def _prepare_store_dir(self):
        """
        打开前的完整性自检（哨兵机制）。

        正常关闭时在 <store>.clean 哨兵里写入建库时的索引类型；进程崩溃/被杀时
        哨兵缺失，而 milvus-lite 崩溃残留的索引文件会在 load 时引发原生段错误
        （实测），因此脏状态直接删除数据目录重建；索引类型变更（如 AUTOINDEX
        残留的 HNSW 索引）同样删除重建——SQLite 是唯一事实源，空索引会由
        KnowledgeService 的首次检索自动全量重建。
        """
        store_path = Path(self.path)
        sentinel = self._sentinel_path()
        if store_path.exists():
            wipe_reason = None
            if not sentinel.exists():
                wipe_reason = "上次未正常关闭（可能损坏）"
            else:
                try:
                    marker = sentinel.read_text(encoding="ascii").strip()
                except Exception:
                    marker = ""
                if marker != INDEX_TYPE:
                    wipe_reason = f"索引配置变更（{marker or '未知'} -> {INDEX_TYPE}）"
            if wipe_reason:
                logger.warning(f"Milvus 数据目录{wipe_reason}，自动删除并重建索引: {self.path}")
                import time
                for attempt in range(5):
                    if not store_path.exists():
                        break
                    try:
                        shutil.rmtree(store_path)
                        break
                    except Exception:
                        time.sleep(0.4 * (attempt + 1))
                if store_path.exists():
                    # 清不掉就拒绝打开（半删状态打开会在 upsert/load 阶段炸出内部错误），
                    # 上层 KnowledgeService 捕获后会降级为关键词检索
                    raise RuntimeError(f"Milvus 数据目录损坏且无法清理: {self.path}")
        store_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._prepare_store_dir()
            Path(self.path).mkdir(parents=True, exist_ok=True)
            self._client = MilvusClient(self.path)
            self._ensure_collection()
        return self._client

    def _ensure_collection(self):
        from pymilvus import DataType

        if self.client.has_collection(COLLECTION):
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="shop_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=16)
        schema.add_field(field_name="ref_id", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_idx", datatype=DataType.INT32)
        schema.add_field(field_name="breadcrumb", datatype=DataType.VARCHAR, max_length=_BREADCRUMB_MAX)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=_TEXT_MAX)
        schema.add_field(field_name="token_count", datatype=DataType.INT32)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        index_params = self.client.prepare_index_params()
        # INDEX_TYPE=FLAT（精确检索）而非 AUTOINDEX：AUTOINDEX 在 milvus-lite 3.2.2
        # 会映射到 HNSW，其 build/load 在 Windows 端到端场景实测触发原生段错误
        # （faiss.swigfaiss.add -> faiss_hnsw.build -> collection.load 崩溃）。
        # 桌面级知识库数据量（万级 chunk）FLAT 全扫足够快，且无索引构建风险。
        index_params.add_index(field_name="embedding", index_type=INDEX_TYPE, metric_type="IP")
        self.client.create_collection(COLLECTION, schema=schema, index_params=index_params)
        logger.info(f"Milvus 集合已创建: {COLLECTION} (dim={self.dim}, path={self.path})")

    def _ensure_loaded(self):
        """
        查询/检索前确保集合已加载。

        milvus-lite 空闲或后台建索引后集合会进入 released 状态，
        之后的 search/query 会报 "call load() before search"，必须显式 load。
        """
        try:
            self.client.load_collection(COLLECTION)
        except Exception as e:
            logger.debug(f"load_collection: {e}")

    def _release_embedded_server(self):
        """
        停止 milvus-lite 嵌入式服务，释放数据目录文件锁。

        MilvusClient.close() 只关闭 gRPC 通道；milvus-lite 的嵌入式服务按路径
        缓存在 server_manager 里，直到进程退出才释放，期间数据目录的文件一直
        被锁住（Windows 上 rmtree 必失败，残留半删目录会引发 WAL 内部错误）。
        必须显式 release_server。释放后再次访问 self.client 会自动重启服务
        （pymilvus 每次建连都重新调 start_and_get_uri，返回新端口），功能无影响。
        """
        try:
            from milvus_lite.server_manager import server_manager_instance

            server_manager_instance.release_server(self.path)
        except Exception as e:
            logger.debug(f"release milvus-lite server: {e}")

    def close(self):
        if self._client is not None:
            try:
                self.client.flush(COLLECTION)
            except Exception:
                pass
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            # 关键：停掉嵌入式服务释放文件锁，否则同进程内删不掉数据目录
            self._release_embedded_server()
            # 正常关闭 → 写哨兵（记录索引类型，下次校验配置一致性后安全打开）
            try:
                self._sentinel_path().write_text(INDEX_TYPE, encoding="ascii")
            except Exception:
                pass

    def clear(self):
        """
        清空全部行数据（保留集合）。

        注意：不能用 drop_collection —— milvus-lite 会在后台异步建索引，
        与 DropCollection 并发时会访问已释放内存导致段错误（实测崩溃）。
        行级删除与后台建索引可以安全并发。
        """
        self._ensure_loaded()
        self.client.delete(COLLECTION, filter='doc_type == "product" or doc_type == "cs"')

    def drop_all(self):
        """删除整个数据目录（测试清理 / 全量重建用）。"""
        try:
            if self._client is not None:
                self.clear()
        except Exception:
            pass
        self.close()
        # 后台索引/压缩线程可能短暂占用文件，带退避重试清理
        removed = False
        for attempt in range(5):
            if not Path(self.path).exists():
                removed = True
                break
            try:
                shutil.rmtree(self.path)
                removed = True
                break
            except Exception:
                import time
                time.sleep(0.4 * (attempt + 1))
        if removed:
            # 数据目录已清空（不存在=干净状态），写哨兵避免下次误判为脏状态
            try:
                self._sentinel_path().write_text(INDEX_TYPE, encoding="ascii")
            except Exception:
                pass
        else:
            # 删除失败 → 删掉哨兵，让下次打开按脏状态自动重建，
            # 绝不能把半删目录标记为干净（WAL 残缺会引发 upsert/load 内部错误）
            logger.warning(f"Milvus 数据目录清理失败（下次打开将自动重建）: {self.path}")
            try:
                self._sentinel_path().unlink(missing_ok=True)
            except Exception:
                pass

    # ---------- 写入 ----------

    @staticmethod
    def _row(chunk: Chunk) -> dict:
        return {
            "id": chunk.chunk_id[:128],
            "shop_id": int(chunk.shop_id),
            "doc_type": chunk.doc_type[:16],
            "ref_id": int(chunk.ref_id),
            "chunk_idx": int(chunk.chunk_idx),
            "breadcrumb": (chunk.breadcrumb or "")[:_BREADCRUMB_MAX],
            "text": (chunk.text or "")[:_TEXT_MAX],
            "token_count": int(chunk.token_count or 0),
            "embedding": list(chunk.embedding or []),
        }

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """写入/覆盖分块（同 id 覆盖）。返回写入数量。"""
        rows = [self._row(c) for c in chunks if c.embedding and len(c.embedding) == self.dim]
        if not rows:
            return 0
        self.client.upsert(COLLECTION, rows)
        return len(rows)

    def delete_by_ref(self, doc_type: str, ref_id: int):
        """删除某条知识记录的全部分块。"""
        self._ensure_loaded()
        self.client.delete(COLLECTION, filter=f"doc_type == {self._q(doc_type)} and ref_id == {int(ref_id)}")

    @staticmethod
    def _q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    # ---------- 检索 ----------

    def search(
        self,
        shop_id: int,
        vector: List[float],
        top_k: int = 30,
        doc_type: Optional[str] = None,
    ) -> List[ScoredChunk]:
        """向量检索（内积 ≈ 余弦，向量已归一化）。前置 shop_id / doc_type 过滤。"""
        self._ensure_loaded()
        flt = f"shop_id == {int(shop_id)}"
        if doc_type:
            flt += f" and doc_type == {self._q(doc_type)}"
        res = self.client.search(
            COLLECTION,
            data=[list(vector)],
            filter=flt,
            limit=max(1, int(top_k)),
            output_fields=["doc_type", "ref_id", "chunk_idx", "breadcrumb", "text", "token_count", "shop_id"],
        )
        out: List[ScoredChunk] = []
        for hit in (res[0] if res else []):
            ent = hit.get("entity", {})
            chunk = Chunk(
                chunk_id=str(hit.get("id", "")),
                doc_type=str(ent.get("doc_type", "")),
                ref_id=int(ent.get("ref_id", 0)),
                shop_id=int(ent.get("shop_id", 0)),
                chunk_idx=int(ent.get("chunk_idx", 0)),
                breadcrumb=str(ent.get("breadcrumb", "")),
                text=str(ent.get("text", "")),
                token_count=int(ent.get("token_count", 0)),
            )
            out.append(ScoredChunk(chunk=chunk, score=float(hit.get("score", 0.0)), source="vector"))
        return out

    def list_chunks(self, shop_id: Optional[int] = None, limit: int = 20000) -> List[Chunk]:
        """读出分块（BM25 重建用）。桌面级数据量一次读完即可。"""
        self._ensure_loaded()
        flt = f"shop_id == {int(shop_id)}" if shop_id is not None else ""
        rows = self.client.query(
            COLLECTION,
            filter=flt,
            output_fields=["id", "shop_id", "doc_type", "ref_id", "chunk_idx", "breadcrumb", "text", "token_count"],
            limit=limit,
        )
        return [
            Chunk(
                chunk_id=str(r.get("id", "")),
                doc_type=str(r.get("doc_type", "")),
                ref_id=int(r.get("ref_id", 0)),
                shop_id=int(r.get("shop_id", 0)),
                chunk_idx=int(r.get("chunk_idx", 0)),
                breadcrumb=str(r.get("breadcrumb", "")),
                text=str(r.get("text", "")),
                token_count=int(r.get("token_count", 0)),
            )
            for r in rows or []
        ]

    def count(self) -> int:
        try:
            self._ensure_loaded()
            rows = self.client.query(COLLECTION, filter="", output_fields=["count(*)"], limit=1)
            return int(rows[0]["count(*)"])
        except Exception:
            try:
                return len(self.list_chunks())
            except Exception:
                return 0
