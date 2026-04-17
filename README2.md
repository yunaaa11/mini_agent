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
├── rag.py              # 混合检索与重排序逻辑
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

### 1️⃣ Agent 能力

* 根据用户自然语言任务决定是否调用工具
* 使用严格 JSON 协议控制 LLM 行为
* 动态抽取参数（如城市名）

### 2️⃣ Tool Calling

* 当前内置工具：`get_weather(city)`
* 工具返回结果会回传给 LLM 生成最终回答

### 3️⃣ RAG（检索增强生成）

* 使用 sentence-transformers 生成文本向量
* 根据用户任务检索 Top-K 相关知识
* 检索结果作为上下文注入 Prompt

---

## 三、运行方式

### ✅ 方式一：本地运行（推荐调试）

```bash
pip install -r requirements.txt
python main.py
```

终端输入示例：

```text
查询上海今天的天气并给我一句出门建议
```

---

### ✅ 方式二：HTTP API 服务

```bash
python app.py
```

请求示例：

```http
POST http://127.0.0.1:5000/agent
Content-Type: application/json

{
  "task": "查询广州今天的天气并给我一句出门建议"
}
```

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

---

## 五、技术选型说明（面试可用）

* **LLM API**：OpenAI / 可替换为其他厂商
* **Agent 架构**：ReAct（决策 + 工具执行）
* **RAG**：SentenceTransformer + 向量相似度
* **服务框架**：Flask
* **部署方式**：Docker 单容器

---

## 六、项目亮点（JD 对齐）

* 工程化使用 LLM（非聊天）
* 严格 JSON 协议，输出可控
* Agent / Tool / RAG 解耦设计
* 支持 CLI 与 API 两种使用方式
* 可直接容器化部署

---

## 七、后续可扩展方向

* 多 Tool（搜索 / 数据库 / 计算器）
* Planner + Executor 两阶段 Agent
* 向量数据库（FAISS / Milvus）
* Prompt 版本管理与 A/B 测试

---

> 本项目为教学与能力展示 Demo，可根据实际业务扩展。

