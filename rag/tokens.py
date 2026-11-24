"""
rag.tokens —— token 计数

优先 tiktoken（cl100k_base）；若编码文件不可用（离线首次运行），
退化为启发式估算（CJK 按 1 字 ≈ 1 token，ASCII 按 4 字符 ≈ 1 token）。
只用于预算控制，不追求精确。
"""
from functools import lru_cache

_ENC = None
_ENC_TRIED = False


@lru_cache(maxsize=1)
def _encoding():
    global _ENC, _ENC_TRIED
    if not _ENC_TRIED:
        _ENC_TRIED = True
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENC = None
    return _ENC


def _estimate(text: str) -> int:
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿" or "　" <= ch <= "〿" or "＀" <= ch <= "￯")
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    other = len(text) - cjk - ascii_chars
    return cjk + (ascii_chars + 3) // 4 + other


def count_tokens(text: str) -> int:
    """统计文本 token 数（近似即可）。"""
    if not text:
        return 0
    enc = _encoding()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return _estimate(text)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """按 token 预算截断文本（贪心逐段，避免碎切）。"""
    if max_tokens <= 0 or not text:
        return ""
    enc = _encoding()
    if enc is not None:
        try:
            ids = enc.encode(text)
            if len(ids) <= max_tokens:
                return text
            return enc.decode(ids[:max_tokens])
        except Exception:
            pass
    # 估算模式：按字符比例粗截（CJK≈1 token/字）
    if _estimate(text) <= max_tokens:
        return text
    return text[:max_tokens]
