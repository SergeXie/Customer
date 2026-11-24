# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Agent-Customer
生成命令: pyinstaller scripts/agent_customer.spec
"""

import sys
import os
from pathlib import Path
import tomllib

block_cipher = None

# 项目根目录（spec 文件位于 scripts/ 目录，上两级为项目根目录）
if '__file__' in globals():
    _spec_dir = Path(os.path.abspath(__file__)).parent
    PROJECT_ROOT = _spec_dir.parent
elif len(sys.argv) > 0:
    # fallback: 从命令行参数推导
    PROJECT_ROOT = Path(sys.argv[0]).resolve().parent.parent
else:
    # last fallback: 从 cwd 推导
    PROJECT_ROOT = Path.cwd()

try:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as _version_file:
        PROJECT_VERSION = tomllib.load(_version_file)["project"]["version"]
except (OSError, KeyError, TypeError):
    PROJECT_VERSION = "0.0.0-dev"

# ================================
# Playwright 驱动打包
# ================================
# collect_all 收集 playwright/driver/ 下的 node 与 JS，否则打包后启动浏览器会
# 因驱动缺失而失败。配合 pdd_login 的 channel="chrome"，用户无需安装 Playwright 浏览器。
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules
_pw_datas, _pw_binaries, _pw_hidden = collect_all("playwright")

# litellm 在 import 阶段就会读取自身的数据文件：
# get_model_cost_map() 会无条件读 litellm/model_prices_and_context_window_backup.json，
# 即使远端取数成功也要读本地那份（它被当作实参先求值，用于完整性校验）。
# 该文件没有任何兜底路径，缺失会在 import litellm 时抛 FileNotFoundError。
# 只加 hiddenimports 不会带上数据文件，因此这里必须显式收集。
# 排除 proxy/（24MB Web 控制台，本应用用不到）与 .tmp（tokenizer 缓存残留）。
_llm_datas = collect_data_files("litellm", excludes=["**/proxy/**", "**/*.tmp"])

_llm_hidden = [
    "litellm",
    "litellm.main",
    "litellm.llms",
]
# litellm.utils 通过 importlib.resources 访问 litellm.litellm_core_utils.tokenizers
# 下的词表文件，该子包必须整体收集，否则冻结后报 ModuleNotFoundError。
_llm_hidden.extend(collect_submodules("litellm.litellm_core_utils"))
for _provider_package in (
    "litellm.llms.openai",
    "litellm.llms.deepseek",
    "litellm.llms.volcengine",
    "litellm.llms.moonshot",
    "litellm.llms.zai",
    "litellm.llms.openai_like",
    "litellm.llms.dashscope",
    "litellm.llms.custom_httpx",
):
    _llm_hidden.extend(collect_submodules(_provider_package))

# node.exe 不会进入 binaries，但会出现在 datas（目标 playwright/driver），这里显式补进 binaries
# 以便在 EXE 元信息（VirusTotal/安全软件）层面可见，行为与 datas 等价，属防御性冗余。
_pw_node = [
    (src, dst) for src, dst in _pw_datas if src.lower().endswith("node.exe")
]
_pw_binaries = list(_pw_binaries) + _pw_node

# ================================
# RAG / Milvus Lite 打包收集
# ================================
# milvus_lite 按路径懒加载 adapter/engine/index 子模块，pymilvus 懒加载
# milvus_lite.server_manager，faiss 被 milvus_lite 的索引模块在运行时 import——
# 静态分析都会漏，必须整体收集（datas/binaries/hiddenimports 一起收）。
_milvus_datas, _milvus_binaries, _milvus_hidden = collect_all("milvus_lite")
_pm_datas, _pm_binaries, _pm_hidden = collect_all("pymilvus")
_fa_datas, _fa_binaries, _fa_hidden = collect_all("faiss")
# faiss 由 delvewheel 打包：libopenblas 等原生 DLL 位于站点包顶层 faiss_cpu.libs/
# （faiss/__init__.py 按 ../faiss_cpu.libs 解析并 add_dll_directory），该目录在
# faiss 包外，collect_all 收不到，需显式带入冻结目录下的同名同级位置。
import faiss as _faiss_mod
_faiss_libs_dir = Path(_faiss_mod.__file__).resolve().parent.parent / "faiss_cpu.libs"
_faiss_lib_bins = (
    [(str(p), "faiss_cpu.libs") for p in sorted(_faiss_libs_dir.glob("*")) if p.is_file()]
    if _faiss_libs_dir.is_dir()
    else []
)

# ================================
# 基础配置
# ================================
a = Analysis(
    [str(PROJECT_ROOT / "app.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=_pw_binaries + _milvus_binaries + _pm_binaries + _fa_binaries + _faiss_lib_bins,
    datas=[
        # 图标文件
        (str(PROJECT_ROOT / "icon" / "icon.ico"), "icon"),
        (str(PROJECT_ROOT / "icon" / "Customer-Agent-qr.png"), "icon"),
    ] + _pw_datas + _llm_datas + _milvus_datas + _pm_datas + _fa_datas,
    hiddenimports=[
        # === PyQt6 & Fluent Widgets ===
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.sip",
        "qfluentwidgets",
        "qfluentwidgets.common",
        "qfluentwidgets.components",
        "qfluentwidgets.window",
        # === AI / Agent ===
        # === 嵌入器 ===
        "openai",
        "openai._models",
        "openai._client",
        "litellm",
        "litellm.main",
        "litellm.litellm_core_utils",
        "litellm.llms.openai",
        "litellm.llms.deepseek",
        "litellm.llms.volcengine",
        "litellm.llms.moonshot",
        "litellm.llms.zai",
        "tiktoken",
        "tiktoken_ext",
        "tiktoken_ext.openai_public",
        # === 数据库 ===
        "sqlalchemy",
        "sqlalchemy.orm",
        "sqlalchemy.sql",
        "sqlalchemy.dialects.sqlite",
        # === 日志 ===
        "loguru",
        "loguru._logger",
        # === Web / 网络 ===
        "websockets",
        "websockets.client",
        "websockets.server",
        "aiohttp",
        "aiohttp.client",
        "aiohttp.web",
        "requests",
        "urllib3",
        "charset_normalizer",
        # === 浏览器自动化 ===
        "playwright",
        "playwright._impl",
        "playwright.async_api",
        # === 数据处理 ===
        "pandas",
        "pandas._libs",
        "numpy",
        "numpy.core",
        "numpy._core",
        "openpyxl",
        "openpyxl.cell",
        "openpyxl.workbook",
        "pypdf",
        "docx",
        "docx.document",
        "docx.oxml",
        # === 图像 ===
        "PIL",
        "PIL._imaging",
        # === 异步 ===
        "asyncio",
        "aiofiles",
        # === 工具类 ===
        "pydantic",
        "pydantic.fields",
        "pydantic.dataclasses",
        "pydantic.v1",
        "pydantic.v1.fields",
        "volcengine",
        "volcengine.base",
        "volcengine.viking_knowledgebase",
        "volcengine.viking_knowledgebase.Collection",
        "volcengine.viking_knowledgebase.Doc",
        # === 项目内部模块 ===
        "config",
        "core.di_container",
        "core.connection_status",
        "core.base_service",
        "database.db_manager",
        "database.models",
        "database.knowledge_service",
        # RAG 检索链路（在 knowledge_service 方法内懒加载，静态分析发现不了）
        "rag",
        "rag.models",
        "rag.tokens",
        "rag.chunker",
        "rag.embeddings",
        "rag.vector_store",
        "rag.bm25",
        "rag.hybrid",
        "rag.context_builder",
        "rag.indexer",
        "bridge.context",
        "bridge.reply",
        "bridge.sender",
        "utils.db_pragma",
        "service.account_service",
        "service.keyword_service",
        "service.human_knowledge",
        "Message.message",
        "Message.core.queue",
        "Message.core.consumer",
        "Message.core.handlers",
        "Message.handlers.base",
        "Message.handlers.ai_handler",
        "Message.handlers.keyword_handler",
        "Message.handlers.preprocessor",
        "Agent.bot",
        "Agent.CustomerAgent.custom.customer_agent",
        "Agent.CustomerAgent.custom.tool_decorator",
        "Agent.CustomerAgent.custom.session_manager",
        "Agent.CustomerAgent.tools",
        "Channel.channel",
        "Channel.pinduoduo.pdd_channel",
        "Channel.pinduoduo.pdd_message",
        "Channel.pinduoduo.pdd_login",
        "Channel.pinduoduo.utils.base_request",
        "Channel.pinduoduo.utils.API.get_token",
        "Channel.pinduoduo.utils.API.send_message",
        "Channel.pinduoduo.utils.API.get_shop_info",
        "Channel.pinduoduo.utils.API.Set_up_online",
        "Channel.pinduoduo.utils.API.product_manager",
        "Channel.pinduoduo.utils.API.get_user_info",
        "utils.logger_loguru",
        "utils.logger_config",
        "utils.resource_manager",
        "utils.path_utils",
        "utils.runtime_path",
        "utils.secret_store",
        # === 避免 pydantic v1/v2 冲突 ===
        "pydantic.errors",
        "pydantic.fields",
        "pydantic.main",
        "pydantic.schema",
        "pydantic.types",
        "pydantic.validators",
        "pydantic.class_validators",
        "pydantic.config",
        "pydantic.parse",
        "pydantic.tools",
        "pydantic.utils",
        "pydantic.validators",
        # === rich (for volcengine) ===
        "rich",
        # === 避免 importlib 静默失败 ===
        "importlib",
        "importlib.util",
        "importlib.abc",
        # === jsonschema (for openapi) ===
        # === certifi / charset ===
        "certifi",
        "charset_normalizer",
        # === httpx (for openai) ===
        "httpx",
        "httpcore",
    ] + _pw_hidden + _llm_hidden + _milvus_hidden + _pm_hidden + _fa_hidden,
    hookspath=[],
    hooksconfig={},
    keys=block_cipher,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # Windows 下关闭 UPX（兼容性更好）
    console=False,        # 不显示控制台窗口（GUI 程序）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ================================
# 生成 PYZ（Python 库）
# ================================
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ================================
# 生成 EXE
# ================================
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AgentCustomer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "icon" / "icon.ico"),
    # Windows 特定
    version="",
    description="电商AI客服助手",
    product_name="Agent-Customer",
    product_version=PROJECT_VERSION,
    company_name="",
    legal_copyright="",
    # 权限
    RequestedExecutionLevel="asInvoker",
)

# ================================
# 收集所有文件到 dist 目录
# ================================
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AgentCustomer",
)
