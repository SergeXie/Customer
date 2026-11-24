# Agent-Customer

电商AI客服桌面应用程序，基于 PyQt6 构建，支持多平台渠道集成，集成 AI 大模型实现智能自动回复。


## 功能特性

- **渠道支持**：目前支持拼多多平台 WebSocket 实时消息接收
- **AI 智能回复**：基于 LangChain / LangGraph Agent（`create_agent`），多轮工具调用 + 会话上下文管理
- **AI 主动推荐**：客服代理可主动获取商品列表、发送商品卡片给用户
- **双知识库体系**：产品知识库 + 客服知识库，分别检索商品信息与售后/物流/退款等政策
- **关键词转人工**：自动识别用户意图，支持关键词触发转人工服务
- **意图识别与路由**：纯规则关键词五类意图分类（投诉/售后/咨询/闲聊/其他），立即分流：投诉转人工、闲聊固定话术、咨询知识库直答、其余 AI 兜底
- **人工回复沉淀知识库**：客服人工回复买家后，自动把「买家问题 + 人工回复」沉淀为客服知识，进入 RAG 检索供 AI 复用
- **知识库 Excel 管理**：客服知识支持批量导入、导出、模板下载，「导出改完再导回」格式完全兼容
- **消息队列处理**：异步消息队列 + 处理器链，支持高并发场景
- **自动重连机制**：WebSocket 连接支持断线自动重连和心跳检测

### AI Agent 可用工具

| 工具名称 | 功能描述 |
|----------|----------|
| `get_shop_products` | 获取店铺商品列表（支持价格区间、销量、库存、标签等） |
| `send_goods_link` | 向用户发送商品卡片链接 |
| `get_product_knowledge` | 查询指定商品的详细知识（成分、规格、用法、价格等） |
| `search_customer_service_knowledge` | 搜索客服知识库（售后、物流、退款等政策问答） |
| `transfer_conversation` | 转接会话给人工客服 |

## 环境要求

- Python >= 3.11
- Windows 操作系统（打包为 exe 后可在 Windows 上独立运行）

## 安装

```bash
# 安装依赖
uv sync
```

## 启动

```bash
python app.py
```

## 配置

首次运行后会生成 `config.json`。推荐在“系统设置 → LLM 模型配置”中填写 API Key；示例、日志和 CI 都不需要环境变量，也不包含真实密钥。

应用通过 LiteLLM 直接调用 Chat Completions（Agent 循环由 LangGraph `create_agent` 驱动），不需要单独启动 LiteLLM Proxy。设置页提供以下六个供应商：

| 设置页供应商 | LiteLLM 路由前缀 | 默认 Base URL | Base URL 规则 |
| --- | --- | --- | --- |
| DeepSeek | `deepseek/` | `https://api.deepseek.com` | 可留空使用默认值，或填写 HTTPS 覆盖 |
| 火山引擎 | `volcengine/` | `https://ark.cn-beijing.volces.com/api/v3` | 可留空使用默认值，或填写 HTTPS 覆盖 |
| OpenAI-compatible | `openai/` | 无 | 必须填写符合 Chat Completions 约定的 HTTPS Base URL；模型名可任意填写 |
| Kimi/Moonshot | `moonshot/` | `https://api.moonshot.cn/v1` | 可留空使用默认值，或填写 HTTPS 覆盖 |
| 智谱/Z.AI | `zai/` | `https://open.bigmodel.cn/api/paas/v4` | 可留空使用默认值，或填写 HTTPS 覆盖 |
| Qwen/DashScope | `dashscope/` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 可留空使用默认值，或填写 HTTPS 覆盖 |

## RAG 检索链路

知识库检索采用 RAG 混合召回：语义切分（结构感知 chunking）→ 向量化 → Milvus Lite 向量检索 + 本地 BM25 倒排检索 → RRF 融合 → 上下文组装（token 预算）→ 引用编号 `[1][2]`。知识新增/更新/删除会自动维护索引；RAG 不可用或无命中时自动回退 jieba 关键词检索。回答要求句末挂引用编号，无引用依据的问题不得编造。

向量库只使用 Milvus（Milvus Lite 本地文件模式），索引类型为 `FLAT`。数据目录默认位于 `%LOCALAPPDATA%\AgentCustomer\milvus.db`（必须是纯 ASCII 路径，非 ASCII 路径会自动回退默认目录）。SQLite 是唯一事实源，索引可随时全量重建；进程异常退出后下次打开会自动清理重建。

`config.json` 中 `rag` 配置块（均有默认值，可不填）：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `rag.enabled` | `true` | 总开关；关闭后走关键词检索 |
| `rag.embedding.provider` | `auto` | `auto`/`litellm`/`hash`；`auto` 在配置了模型时走 LiteLLM，否则用本地哈希向量 |
| `rag.embedding.model` | 空 | 远端 embedding 模型标识 |
| `rag.embedding.api_key` / `api_base` | 空 | 留空时复用 LLM 配置 |
| `rag.embedding.dim` | `1024` | 向量维度 |
| `rag.store_path` | 空 | Milvus 数据目录，留空用默认目录 |
| `rag.top_k_vector` / `top_k_bm25` | `30` / `30` | 两路召回各自取的候选数 |
| `rag.rrf_k` | `60` | RRF 融合参数 |
| `rag.final_top_k` | `8` | 融合后进入上下文的条数 |
| `rag.token_budget` | `4000` | 上下文组装 token 预算 |

## 人工回复沉淀知识库

客服人员在拼多多人工回复买家后，系统自动把这次「买家问题 + 人工回复」沉淀为一条客服知识（默认标签 `人工回复`），并进入 RAG 检索链路，供后续 AI 回答同类问题时复用。

工作方式：

- 买家的纯文本提问进入问题缓冲；客服侧的人工文本回复触发沉淀，自动配对生成知识条目（标题为买家问题，内容为人工回复）
- **回显抑制**：AI 自动回复、关键词回复、工具发送的文本在发送前登记，回流时不沉淀，避免 AI 把自己的回答当成人工知识造成自激污染
- **连发合并**：客服连续多条消息在合并窗口内并入同一条知识；同一问答再次出现时自动去重（含合并后内容的包含式判重）
- **窗口过期**：问题缓冲超过有效期后不再配对；转人工通知等非买家提问不会误入缓冲
- **草稿模式**：`default_enabled=false` 时沉淀的知识默认停用，需人工在知识库中审核启用后才会被检索

`config.json` 中 `knowledge_capture` 配置块（均有默认值，可不填）：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `knowledge_capture.enabled` | `true` | 沉淀功能总开关 |
| `knowledge_capture.default_enabled` | `true` | 新沉淀知识是否直接启用；`false` 为草稿模式 |
| `knowledge_capture.question_window_seconds` | `600` | 买家问题与人工回复配对的有效窗口 |
| `knowledge_capture.merge_window_seconds` | `60` | 客服连发消息合并为同一条知识的时间窗口 |
| `knowledge_capture.echo_ttl_seconds` | `600` | 自家回显抑制缓存的有效期 |
| `knowledge_capture.min_reply_chars` | `4` | 触发沉淀的最短人工回复字数 |
| `knowledge_capture.tags` | `人工回复` | 沉淀知识的默认标签 |

## 意图识别与路由

买家文本消息在进入 AI 之前先做一次纯规则关键词意图分类（零 LLM 成本），按粗粒度五类意图立即分流：

| 意图 | 分类依据（命中关键词） | 分流动作 |
| --- | --- | --- |
| 投诉 `complaint` | 投诉/骗/假货/举报/维权/315 等 | 安抚话术 + 转人工；无在线客服时改发记录话术 |
| 售后 `aftersale` | 退款/退货/换货/破损/漏发/质量问题 等 | 交给 AI 兜底（走知识库工具） |
| 咨询 `consult` | 多少钱/发货/快递/尺码/用法 等 | 客服知识命中且相关 → 直答短路；弱匹配/未命中降级 AI |
| 闲聊 `chitchat` | 你好/在吗/谢谢/ok 等（仅短消息） | 固定话术直接回复 |
| 其他 `other` | 未命中 | 交给 AI 兜底 |

设计约定：

- 分类优先级：投诉 > 售后 > 咨询 > 闲聊（先重后轻），每条消息强制落位一个意图，绝不静默丢弃
- **闲聊短消息守卫**：消息超过 `chitchat_max_chars` 不判闲聊，避免长消息因寒暄词被误判
- **知识直答护栏**：知识命中后要求查询与知识内容共享至少一个分词（jieba），弱匹配降级 AI，防止答非所问
- 打点只记意图/命中词/消息长度（不记原文）；分类失败或发送失败一律降级给下游 AI 处理器

`config.json` 中 `intent` 配置块（均有默认值，可不填）：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `intent.enabled` | `true` | 意图识别总开关，关闭后全部走 AI 兜底 |
| `intent.chitchat_reply` | `在的哦，请问有什么可以帮您？` | 闲聊意图固定话术 |
| `intent.complaint_reply` | `非常抱歉给您带来不好的体验…` | 投诉安抚话术（转人工前发送） |
| `intent.complaint_no_cs_reply` | `非常抱歉，当前暂无在线人工客服…` | 投诉且无在线客服时的话术 |
| `intent.chitchat_max_chars` | `20` | 闲聊判定的消息长度上限 |
| `intent.knowledge_max_chars` | `200` | 咨询知识直答的回复长度上限 |
| `intent.complaint_keywords` | 投诉/骗/假货/举报等 | 投诉关键词列表 |
| `intent.aftersale_keywords` | 退款/退货/换货等 | 售后关键词列表 |
| `intent.consult_keywords` | 多少钱/发货/快递等 | 咨询关键词列表 |
| `intent.chitchat_keywords` | 你好/在吗/谢谢等 | 闲聊关键词列表 |

## 开发规范

### 新增/修改接口前

1. **先用 curl 或 Python 脚本测试接口**，确认真实请求参数、请求头、响应结构
2. **根据实际响应结构修改解析代码**，不要凭猜测写字段名
3. **修改后用 mock 数据或真实调用验证解析逻辑**

> 例如：修改 `product_manager.py` 的商品列表接口时，先用 curl 测试接口，确认数据在 `result.onSaleGoods` 字段而非 `result.goodsList`，字段名为驼峰 `goodsId` 而非下划线 `goods_id`，价格单位是"分"需除以 100 转换为"元"

## 测试

项目带完整单测（292 项，覆盖 RAG 检索链路、人工回复沉淀、意图识别路由、LLM 传输、消息解析路由、打包配置等），运行：

```bash
python -m unittest discover -s tests -t .
```

测试默认通过 `AGENT_CUSTOMER_RAG_DISABLED=1` 隔离全局 Milvus 索引，不会污染 `%LOCALAPPDATA%\AgentCustomer\milvus.db`；RAG 链路专项测试（`tests/test_rag_pipeline.py`）内部自建临时向量库验证完整流程。


## 项目结构

```
text
Agent-Customer/
├── Agent/                  # AI 代理模块（LangChain / LangGraph）
│   └── CustomerAgent/
│       ├── custom/         # LangChain ChatModel、工具适配、消息转换、会话管理
│       └── tools/          # Agent 工具集（商品/知识/转人工）
├── Channel/                # 渠道集成
│   └── pinduoduo/
│       ├── core/           # 连接、生命周期、状态、消息处理拆分模块
│       └── utils/API/      # 拼多多 API 封装
├── Message/                # 消息处理（队列 + 处理器链）
│   ├── core/               # consumer / handlers / queue
│   └── handlers/           # 预处理器、关键词、意图识别、AI 处理器
├── bridge/                 # 桥接模块（Context/Reply/SendService）
├── core/                   # 核心服务（DI 容器、连接状态）
├── database/               # 数据库（SQLAlchemy + 知识服务 + 商品同步）
├── rag/                    # RAG 检索链路（切分/向量化/Milvus/BM25/混合召回/引用）
├── service/                # 服务层（账号/关键词服务、人工回复沉淀、知识 Excel 导入导出）
├── ui/                     # PyQt6 用户界面
├── utils/                  # 工具模块（日志、路径等）
├── scripts/                # 构建与抓包脚本
└── app.py                  # 应用入口
```

## 技术栈

| 类别 | 技术 |
|------|------|
| UI 框架 | PyQt6 + pyqt6-fluent-widgets |
| AI 框架 | LangChain 1.x + LangGraph `create_agent` + LiteLLM 直接异步调用 |
| 数据库 | SQLAlchemy + SQLite |
| 中文分词 | jieba（BM25/关键词检索） |
| 向量检索 | pymilvus + Milvus Lite（FLAT 索引）+ BM25 混合召回（RRF） |
| Token 统计 | tiktoken |
| 异步通信 | asyncio + websockets |
| 文档解析 | pypdf + python-docx + openpyxl + xlrd |
| 日志 | Loguru |
| 配置 | Pydantic |

## License

MIT
