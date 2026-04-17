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
    {
        "question": "北京冬天冷不冷？穿什么衣服？",
        "ground_truth": "北京冬季（12-2月）平均气温-5℃至2℃，寒冷干燥，常有北风。建议穿羽绒服、保暖内衣等。"
    },
    {
        "question": "我想订一张去上海的火车票",
        "ground_truth": "系统应首先询问用户的姓名和身份证号，查询余票并展示车次。"
    },
    {
        "question": "兵马俑周一闭馆吗？",
        "ground_truth": "兵马俑周一正常开放，仅除夕闭馆。"
    },
    {
        "question": "那边有什么好吃的？",
        "ground_truth": "若指上海则推荐小笼包、生煎等；系统应主动询问具体城市。"
    }
]

async def get_agent_response(question):
    """封装 agent.py 的输出，提取纯文本回复"""
    full_text = ""
    async for chunk in run_agent(question, session_id="eval_test"):
        if chunk not in ["[THOUGHT_START]", "[THOUGHT_END]"]:
            full_text += chunk
    return full_text

async def main():
    results = []
    print("=== 正在收集 Agent 回答 ===")
    for item in test_dataset:
        print(f"正在测试问题：{item['question']}")
        
        # 获取检索到的上下文 (用于 ContextPrecision/Recall)
        contexts = retrieve(item["question"])
        # 获取 AI 回答 (用于 Faithfulness)
        answer = await get_agent_response(item["question"])
        
        results.append({
            "question": item["question"],
            "contexts": contexts,
            "answer": answer,
            "ground_truth": item["ground_truth"]
        })
    
    print("\n=== 所有回答已收集，开始调用 Ragas 进行定量评估 ===")
    
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
    
    selected_metrics = [f, cp, cr]

    # 4. 执行评估
    print("正在计算分数，请观察下方是否弹出 HTTP 请求日志...")
    
    dataset = Dataset.from_list(results)
    
    score = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
        run_config=RunConfig(max_workers=1, timeout=300)
    )

    # 5. 展示
    print("\n=== 评估得分摘要 ===")
    print(score)
    df = score.to_pandas()
    df.to_csv("eval_results.csv", index=False)
    print("\n详细结果已保存至 eval_results.csv")

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