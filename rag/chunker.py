"""
rag.chunker —— 语义切分

对应九步链路的第 3 步「语义切分 chunking」：
- 按标题层级切分（# 标题 / 【小节】/ 一、二、/ 1. 等），不切断表格、代码块；
- 段内按句子打包，目标 256~512 token（默认 384），带 10~15% 重叠（默认 12%）；
- 每个分块带面包屑 breadcrumb（文档 > 章节），供引用编号溯源。
"""
import re
from typing import List, Optional

from .models import Chunk
from .tokens import count_tokens

# 标题行模式：Markdown 标题 / 【小节】/ 中文序号 / 阿拉伯序号 / 井号
_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,6}\s+.+"                    # # 标题
    r"|【[^】]{1,40}】.*"              # 【小节】
    r"|[一二三四五六七八九十]+、.{1,40}" # 一、小节
    r"|(?:\d+)[.、]\s*\S{1,40}"        # 1. 小节 / 1、小节
    r"|(?:[-*_])\s*\S{1,40}"           # - 小节
    r")$"
)

# 表格行 / 代码块行保护：整行随所在分块走，不作为切分边界
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# 句子结束标点
_SENT_END_RE = re.compile(r"(?<=[。！？；!?;])\s*")

# 句读/表格符号：出现即不是裸标题
_SENT_PUNCT = set("。！？；，、：,!?;:|｜（）()【】[]「」《》<>")


def _is_heading(line: str) -> bool:
    """结构感知的标题判定：显式标题模式，或「裸标题行」（短行且无句读/表格符号）。"""
    s = line.strip()
    if not s or _TABLE_ROW_RE.match(s):
        return False
    if _HEADING_RE.match(s):
        return True
    # 裸标题行：如「退款政策」「物流政策」这类无标记小节名
    return 2 <= len(s) <= 24 and not any(ch in s for ch in _SENT_PUNCT)

DEFAULT_TARGET_TOKENS = 384      # 目标分块大小（256~512 区间内）
DEFAULT_MIN_TOKENS = 64          # 太小的尾巴并回上一块
DEFAULT_OVERLAP_RATIO = 0.12     # 重叠 12%（10~15% 区间内）


def _split_sentences(text: str) -> List[str]:
    """按句号类标点切句，保留换行（表格/代码块整行保留）。"""
    sentences: List[str] = []
    for line in text.splitlines():
        if _TABLE_ROW_RE.match(line) or _CODE_FENCE_RE.match(line):
            if line.strip():
                sentences.append(line.strip())
            continue
        for part in _SENT_END_RE.split(line):
            if part and part.strip():
                sentences.append(part.strip())
    return sentences


def _split_sections(text: str, doc_title: str) -> List[tuple]:
    """
    结构感知切分：按标题行切成 (breadcrumb, section_text) 列表。
    表格与代码块内的伪标题行不会触发切分。
    """
    sections: List[tuple] = []
    cur_title = doc_title
    cur_lines: List[str] = []
    in_code = False

    for line in text.splitlines():
        if _CODE_FENCE_RE.match(line):
            in_code = not in_code
            cur_lines.append(line)
            continue
        if not in_code and _is_heading(line):
            if cur_lines and "".join(cur_lines).strip():
                sections.append((cur_title, "\n".join(cur_lines)))
            cur_title = f"{doc_title} > {line.strip().lstrip('#').strip()}"
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_lines and "".join(cur_lines).strip():
        sections.append((cur_title, "\n".join(cur_lines)))
    if not sections and text.strip():
        sections.append((doc_title, text))
    return sections


def chunk_text(
    text: str,
    *,
    doc_type: str,
    ref_id: int,
    shop_id: int,
    doc_title: str = "",
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> List[Chunk]:
    """
    把一篇知识文本切成 Chunk 列表。

    Args:
        text: 原文（产品知识正文 / 客服知识标题+正文）
        doc_type/ref_id/shop_id: 归属信息（写入 chunk 元数据）
        doc_title: 文档标题（面包屑根节点）
        target_tokens: 目标分块大小
        min_tokens: 低于该大小的尾块并回上一块
        overlap_ratio: 相邻分块的重叠比例（0~0.3）
    """
    text = (text or "").strip()
    if not text:
        return []
    doc_title = (doc_title or "").strip() or "知识库"
    overlap_ratio = max(0.0, min(0.3, overlap_ratio))

    chunks: List[Chunk] = []
    idx = 0
    for breadcrumb, section in _split_sections(text, doc_title):
        sentences = _split_sentences(section)
        if not sentences:
            continue

        buf: List[str] = []
        buf_tokens = 0
        for sent in sentences:
            sent_tokens = count_tokens(sent)
            # 单句超长：硬切（尽量在句内标点处），单独成块
            if sent_tokens > target_tokens * 1.5:
                if buf:
                    chunks.append(_make_chunk(idx, breadcrumb, buf, doc_type, ref_id, shop_id))
                    idx += 1
                    buf, buf_tokens = [], 0
                for piece in _hard_split(sent, target_tokens):
                    chunks.append(_make_chunk(idx, breadcrumb, [piece], doc_type, ref_id, shop_id))
                    idx += 1
                continue
            if buf_tokens + sent_tokens > target_tokens and buf:
                chunks.append(_make_chunk(idx, breadcrumb, buf, doc_type, ref_id, shop_id))
                idx += 1
                # 重叠：从上一块尾部回捞句子，约 overlap_ratio 比例
                carry: List[str] = []
                carry_tokens = 0
                for prev in reversed(buf):
                    prev_tokens = count_tokens(prev)
                    if carry_tokens + prev_tokens > target_tokens * overlap_ratio:
                        break
                    carry.insert(0, prev)
                    carry_tokens += prev_tokens
                buf, buf_tokens = carry, carry_tokens
            buf.append(sent)
            buf_tokens += sent_tokens

        if buf:
            if buf_tokens < min_tokens and chunks and chunks[-1].breadcrumb == breadcrumb:
                # 尾巴太小并回同节上一块
                last = chunks[-1]
                merged_text = last.text + "\n" + "\n".join(buf)
                chunks[-1] = _make_chunk(last.chunk_idx, breadcrumb, [merged_text], doc_type, ref_id, shop_id)
            else:
                chunks.append(_make_chunk(idx, breadcrumb, buf, doc_type, ref_id, shop_id))
                idx += 1

    return chunks


def _make_chunk(chunk_idx: int, breadcrumb: str, parts: List[str], doc_type: str, ref_id: int, shop_id: int) -> Chunk:
    body = "\n".join(p for p in parts if p).strip()
    return Chunk(
        chunk_id=f"{doc_type}:{ref_id}:{chunk_idx}",
        doc_type=doc_type,
        ref_id=ref_id,
        shop_id=shop_id,
        chunk_idx=chunk_idx,
        breadcrumb=breadcrumb,
        text=body,
        token_count=count_tokens(body),
    )


def _hard_split(sent: str, target_tokens: int) -> List[str]:
    """超长单句按标点/长度硬切。"""
    pieces = [p for p in re.split(r"(?<=[，,、。：:])\s*", sent) if p]
    out: List[str] = []
    buf = ""
    for piece in pieces:
        if buf and count_tokens(buf + piece) > target_tokens:
            out.append(buf)
            buf = piece
        else:
            buf += piece
        # 单个碎片仍超长：按字符截断
        while count_tokens(buf) > target_tokens * 1.5:
            cut = max(1, int(len(buf) * target_tokens * 1.2 / count_tokens(buf)))
            out.append(buf[:cut])
            buf = buf[cut:]
    if buf:
        out.append(buf)
    return out
