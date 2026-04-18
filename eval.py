import asyncio
import pandas as pd
from openai import OpenAI
from datasets import Dataset
# Ragas 相关导入
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
# from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
# 你的项目模块导入
from config import Config
from rag import retrieve
from agent import run_agent

# 1. 测试集定义
test_dataset = [
    {"question": "北京冬天冷不冷？穿什么衣服？", "ground_truth": "北京冬季（12-2月）平均气温-5℃至2℃，寒冷干燥，常有北风。建议穿羽绒服、保暖内衣等。"},
    {"question": "我想订一张去上海的火车票", "ground_truth": "系统应首先询问用户的姓名和身份证号。"},
    {"question": "兵马俑周一闭馆吗？", "ground_truth": "兵马俑周一正常开放，仅除夕闭馆。"},
    {"question": "那边有什么好吃的？", "ground_truth": "系统应主动询问具体城市。"}
]

async def get_agent_response(question):
    """封装 agent.py 的输出，提取纯文本回复"""
    full_text = ""
    async for chunk in run_agent(question, session_id="eval_test"):
        if chunk not in ["[THOUGHT_START]", "[THOUGHT_END]"]:
            full_text += chunk
    return full_text

async def main():
    results = [] # 存放 RAG 问答类数据
    task_results = [] # 存放业务订票类数据
    print("=== 正在收集 Agent 回答 ===")
    for item in test_dataset:
        print(f"正在测试问题：{item['question']}")
        
        answer = await get_agent_response(item["question"])
        
        # 2. 任务分类判定
        # 如果是订票任务，走手动逻辑评估
        if any(kw in item['question'] for kw in ["订", "票", "买"]):
            # 判定标准：是否包含关键信息的追问
            is_success = "姓名" in answer and "身份证" in answer
            task_results.append({
                "question": item["question"],
                "status": "Success" if is_success else "Fail",
                "answer": answer
            })
            continue # 跳过 RAGAS 评估

        # RAG 类任务处理
        # 只有非业务任务才会执行到这里
        contexts = retrieve(item["question"])
        results.append({
            "question": item["question"],
            "contexts": contexts,
            "answer": answer,
            "ground_truth": item["ground_truth"]
        })
    
    # 1. 创建原生 OpenAI Client
    openai_client = OpenAI(
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_BASE_URL
    )
    
    # 2. 创建 LLM
    evaluator_llm = llm_factory(model=Config.LLM_MODEL, client=openai_client)

    # 3. 核心修复：显式初始化每个指标对象
    # 注意：确保这里每一项都是带 () 的实例
    f = Faithfulness(llm=evaluator_llm)
    cp = ContextPrecision(llm=evaluator_llm)
    cr = ContextRecall(llm=evaluator_llm)
    # ar = AnswerRelevancy(llm=evaluator_llm)

    selected_metrics = [f, cp, cr]
    # 安全执行：只有 results 不为空才跑 Ragas
    if results:
        print("\n=== 正在计算 RAG 质量得分... ===")
        dataset = Dataset.from_list(results)
        score = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            run_config=RunConfig(max_workers=1, timeout=300)
        )
        print("\n=== RAG 质量得分摘要 ===")
        print(score)
        df = score.to_pandas()
        df.to_csv("rag_eval_results.csv", index=False)
    else:
        print("\n[提示] 没有需要跑 RAGAS 的数据。")

    # 打印业务任务执行报告（放在最最后）
    print("\n" + "="*30)
    print("=== 业务任务执行报告 (Task Success) ===")
    print("="*30)
    if task_results:
        for t in task_results:
            color = "✅" if t["status"] == "Success" else "❌"
            print(f"{color} 问题: {t['question']}")
            print(f"   判定结果: {t['status']}")
            print(f"   Agent回复: {t['answer'][:50]}...") # 只显前50字
    else:
        print("没有业务任务测试项。")

if __name__ == "__main__":
    # 解决部分环境下 asyncio.run 冲突的问题
    try:
        asyncio.run(main())
    except RuntimeError:
        # 如果在 Jupyter 或某些特殊 shell 环境下运行
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())