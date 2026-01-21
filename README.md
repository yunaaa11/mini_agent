# Agent with RAG

这是一个**可控的 LLM Agent 小项目**， Agent / RAG 的基本工程要求：

* 非聊天式 LLM 使用（JSON 协议）
* Tool Calling（天气查询）
* RAG（向量检索增强）
* CLI / HTTP 双入口
* 可 Docker 化部署

---

## 一、项目结构

```text
agent-demo/
├── agent.py            # Agent 核心逻辑（决策 + Tool 调用）
├── rag.py              # RAG 检索逻辑（Embedding + Top-K）
├── rag_data.py         # 本地知识库
├── tools.py            # 工具函数（天气查询）
├── llm.py              # LLM API 封装
├── main.py             # 终端调试入口（CLI）
├── app.py              # Flask API 服务入口
├── requirements.txt    # Python 依赖
├── Dockerfile          # Docker 构建文件
└── README.md           # 项目说明
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

