"""
rag.context_builder —— 上下文组装与引用编号

对应第 8 步「上下文组装」：
- 去重合并：同一记录的多分块按得分归并，互相包含的文本只留一份；
- token 预算：默认 4000，超预算按得分从低到高丢弃；
- 引用编号：输出文本里出现 [1] [2] …，与 Citation 列表一一对应，
  面包屑保证「引用与来源不错位」。
"""
from typing import Dict, List, Tuple

from .models import Citation, ScoredChunk
from .tokens import count_tokens, truncate_to_tokens

DEFAULT_BUDGET = 4000
_SNIPPET_MAX = 500


def _normalize(text: str) -> str:
    return "".join((text or "").split())


def build_context(
    hits: List[ScoredChunk],
    budget: int = DEFAULT_BUDGET,
) -> Tuple[str, List[Citation]]:
    """
    把混合召回结果组装为带引用编号的上下文。

    Returns:
        (context_text, citations)
        context_text 形如：
            [1] 客服政策 > 退款：退货签收后 48 小时内退款…
            [2] 商品A > 成分：…
    """
    if not hits:
        return "", []

    # 1) 去重：文本互相包含的只留高分；同 chunk_id 去重
    seen_ids = set()
    kept: List[ScoredChunk] = []
    for sc in sorted(hits, key=lambda s: s.score, reverse=True):
        cid = sc.chunk.chunk_id
        if cid in seen_ids:
            continue
        norm = _normalize(sc.chunk.text)
        if not norm:
            continue
        dominated = False
        for prev in kept:
            pn = _normalize(prev.chunk.text)
            if norm in pn or pn in norm:
                dominated = True
                break
        if dominated:
            continue
        seen_ids.add(cid)
        kept.append(sc)

    # 2) 引用编号 + token 预算（从高分到低分装填）
    citations: List[Citation] = []
    lines: List[str] = []
    used = 0
    for sc in kept:
        n = len(citations) + 1
        snippet = truncate_to_tokens(sc.chunk.text.strip(), _SNIPPET_MAX)
        line = f"[{n}] {sc.chunk.breadcrumb}：{snippet}"
        cost = count_tokens(line)
        if citations and used + cost > budget:
            break
        citations.append(Citation(
            index=n,
            doc_type=sc.chunk.doc_type,
            ref_id=sc.chunk.ref_id,
            breadcrumb=sc.chunk.breadcrumb,
            snippet=snippet,
        ))
        lines.append(line)
        used += cost

    return "\n".join(lines), citations


def format_citation_map(citations: List[Citation]) -> str:
    """把引用表渲染给 LLM（prompt 约束：答案句挂引用，无依据拒答）。"""
    if not citations:
        return ""
    rows = [f"[{c.index}] {c.breadcrumb}" for c in citations]
    return "引用来源：\n" + "\n".join(rows)
