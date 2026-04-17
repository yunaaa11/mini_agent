# Agent with RAG

本项目是一个基于 LangGraph 和 FastAPI 构建的生产级智能体框架，集成了混合搜索、多工具调度及全链路监控系统。

---
# 🌟 核心特性
* **双路混合检索 (Hybrid RAG)**：结合 Chroma 向量检索与 BM25 关键词检索，并通过 FlashRank 进行重排序 (Re-ranking)，显著提升长尾知识的召回精度。
* **多智能体协同**：利用 LangGraph 维护复杂的决策图状态，实现“知识检索-工具调用-逻辑推理”的闭环。
* **生产级观测系统 (Observability)**：
    * **分级日志机制**：异步捕获 Agent 决策流、工具执行耗时及 RAG 检索链路。
    * **自动日志轮转**：内置自动切分与压缩逻辑，确保系统长期运行的稳定性。
* **可控性增强**：通过自定义 Prompt 模版与参数抽取器 (CityParser)，解决大模型在特定场景下的参数幻觉问题。

## 一、项目结构

```text
agent-demo/
├── agent.py            # Agent 核心逻辑（决策 + Tool 调用）
├── rag.py              # 混合检索与精排逻辑 (BM25 + Chroma + FlashRank)
├── tools.py            # 异步工具集（API 封装与结果缓存）
├── llm.py              # LLM API 封装
├── main.py             # 终端调试入口（CLI）
├── app.py              # 流式响应 (SSE) 后端服务
├── requirements.txt    # Python 依赖
├── Dockerfile          # Docker 构建文件
|── README.md           # 项目说明
├── logger.py           # 日志模块（标准库封装，支持文件/控制台双输出）
├── logs/               # [自动生成] 存放运行日志，便于调试和追踪 Agent 决策流
```

---

## 二、功能说明

### 1️⃣ 核心 Agent 架构：基于 LangGraph 的有向无环图 (DAG)
不同于传统的线性 Chain，本项目采用 LangGraph 维护复杂的决策逻辑：

* **多智能体协作路由**：系统会根据用户意图，在 General Agent（通用任务）与 Booking Expert（订票专家）之间动态路由，实现多场景精准覆盖。

* **状态持久化与断点恢复**：集成 AsyncSqliteSaver，支持对话状态的长效记忆与 Checkpoint 机制，确保会话上下文不丢失。

* **结构化输出控制**：强制模型通过 JSON 协议进行 Tool Calling，并配合 CityParser 自研参数提取器，有效降低了大模型在处理地理信息时的参数幻觉。

### 2️⃣ 高性能混合检索 (Advanced RAG)
为了解决单一向量检索在专有名词和长尾知识上召回率低的问题，本项目构建了双路检索链路：

* **双路并行召回**：同步调用 Chroma 语义向量检索与 BM25 关键词检索，确保既能理解语义，又能精准匹配关键词（如航班号、特定产品型号）。

* **FlashRank 精排层**：检索结果汇总后，引入 ms-marco-TinyBERT 模型进行交叉验证重排序（Re-rank），从初步召回的 Top-10 结果中筛选出最相关的 Top-5，极大提升了上下文注入的信噪比。

* **文档增量管理**：支持 PDF 与文本上传，内置 MD5 全文去重机制，避免重复加载相同内容导致的向量库膨胀与计算浪费。

### 3️⃣ 生产级工具链与缓存 (Tooling)
* **异步流式工具**：所有工具（天气查询、在线搜索、列车预订）均采用 async 异步封装，防止 I/O 阻塞。

* **智能缓存机制**：集成 TTLCache，对高频请求（如天气查询、热门搜索）进行结果缓存，显著降低 API 调用成本并提升响应速度。

* **实时搜索补位**：当本地知识库（RAG）无法满足需求时，系统会自动触发 Tavily Online Search 抓取即时信息，确保信息的时效性。

### 4️⃣ 全链路可观测性 (Observability)
* **分级日志系统**：自研 logger.py 模块，将控制台（INFO）用于实时监控，将文件（DEBUG）用于深度复盘。

* **决策流溯源**：日志完整记录了从“用户输入 -> RAG 检索片段 -> Agent 思考链 (Thought) -> 工具调用参数 -> 最终输出”的全链路数据快照，大幅缩短了模型调试与 Prompt 优化周期。

---

## 三、运行方式

1. 环境准备

```bash
# 拷贝环境变量
cp .env.example .env 
# 安装依赖
pip install -r requirements.txt
```
---

2. 运行服务
CLI 模式: python main.py (快速调试 Agent 逻辑)

API 模式: python app.py (启动后端服务，监听 5000 端口)

---

## 四、Docker 部署（企业常用）

### 1️⃣ 构建镜像

```bash
docker build -t agent-rag-demo .
```

### 2️⃣ 启动容器

```bash
docker run -p 5000:5000 agent-rag-demo
```

访问：

```text
http://localhost:5000/agent
```

