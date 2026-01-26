from agent import run_agent
from rag import retrieve

def run_agent_with_rag(task:str,city:str):
    print("[RAG] agent_with_rag被调用")
    print("[RAG]city=",city)
    print("[RAG]检索query=",task+city)
     # 将 task 和 city 拼接后作为查询，
    # 用于从知识库中检索最相关的背景文档
    docs=retrieve(task+city)
    # 构造“增强后的任务提示词（Prompt）”    
    # 将城市信息 + 检索到的背景知识 + 用户原始任务  
    # 统一交给 Agent，提高回答的准确性与上下文感知
    print("[RAG]命中文档：",docs[0])
    enriched_task=f"""
    城市：{city}

    相关背景知识：
    {docs[0]}

    用户任务：
    {task}
    """
    # 将增强后的任务传给 Agent 执行
    return run_agent(enriched_task)
# 用户输入
#    ↓
# RAG 检索（retrieve）
#    ↓
# Prompt 注入背景知识
#    ↓
# Agent 推理 / 调用工具
#    ↓
# 最终输出
