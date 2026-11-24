"""
rag.embeddings —— 向量化抽象

- LiteLLMEmbedder：走 litellm 的 embedding 接口（OpenAI 兼容），
  可接 bge-m3 / text-embedding-3 等真实语义模型（需在配置里指定模型）。
- HashEmbedder：零依赖离线兜底（jieba 词 + 字 bigram 特征哈希，签名哈希 + L2 归一），
  保证链路在无网络/无 API Key 时也可运行；语义能力弱于真实模型，但可随时切换。
- build_embedder：按配置工厂；"auto" = 配了模型走 LiteLLM，否则哈希兜底。

向量一律 L2 归一化，Milvus 用内积（IP）等价于余弦相似度。
"""
import hashlib
from typing import List, Protocol, runtime_checkable

from utils.logger_loguru import get_logger

logger = get_logger("RagEmbeddings")


@runtime_checkable
class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化。失败抛异常，由调用方决定降级策略。"""
        ...


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


class HashEmbedder:
    """离线确定性向量化：jieba 词级 + 字 bigram 特征哈希（signed hashing）。"""

    def __init__(self, dim: int = 1024):
        self.dim = max(16, int(dim))

    def _tokens(self, text: str):
        try:
            import jieba
            words = [w.strip() for w in jieba.lcut(text) if w.strip()]
        except Exception:
            words = []
        chars = [ch for ch in text if not ch.isspace()]
        bigrams = ["".join(chars[i:i + 2]) for i in range(len(chars) - 1)]
        return words + bigrams

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 == 0 else -1.0
            vec[bucket] += sign
        return _l2_normalize(vec)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t or "") for t in texts]


class LiteLLMEmbedder:
    """经 litellm 调用远端 embedding 模型（OpenAI Chat 兼容供应商均可）。"""

    def __init__(self, model: str, dim: int = 1024, api_key: str = "", api_base: str = ""):
        if not model:
            raise ValueError("LiteLLMEmbedder 需要配置 embedding 模型名")
        self.model = model
        self.dim = max(16, int(dim))
        self.api_key = api_key or ""
        self.api_base = api_base or ""
        self._batch_size = 32

    def embed(self, texts: List[str]) -> List[List[float]]:
        import litellm

        out: List[List[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [t or "" for t in texts[start:start + self._batch_size]]
            kwargs = {"model": self.model, "input": batch}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base
            resp = litellm.embedding(**kwargs)
            items = sorted(resp.get("data", []), key=lambda d: d.get("index", 0))
            for item in items:
                vec = [float(x) for x in item["embedding"]]
                if len(vec) != self.dim:
                    # 模型实际维度与配置不符：以实际为准并告警（建库维度以首个向量为准）
                    logger.warning(f"embedding 维度与配置不符: got={len(vec)} configured={self.dim}")
                    self.dim = len(vec)
                out.append(_l2_normalize(vec))
        return out


def build_embedder(rag_config) -> EmbeddingProvider:
    """
    按 RagConfig.embedding 构造向量化器。
    provider: "auto" | "litellm" | "hash"
    """
    emb_cfg = getattr(rag_config, "embedding", None)
    provider = (getattr(emb_cfg, "provider", "auto") or "auto").lower()
    model = (getattr(emb_cfg, "model", "") or "").strip()
    dim = int(getattr(emb_cfg, "dim", 1024) or 1024)

    if provider == "hash":
        return HashEmbedder(dim=dim)
    if provider == "litellm" or (provider == "auto" and model):
        try:
            return LiteLLMEmbedder(
                model=model,
                dim=dim,
                api_key=getattr(emb_cfg, "api_key", "") or "",
                api_base=getattr(emb_cfg, "api_base", "") or "",
            )
        except Exception as e:
            logger.warning(f"LiteLLM embedding 初始化失败，退化为本地哈希向量: {e}")
    elif provider not in ("auto", "hash", "litellm"):
        logger.warning(f"未知 embedding provider: {provider}，使用本地哈希向量")
    return HashEmbedder(dim=dim)
