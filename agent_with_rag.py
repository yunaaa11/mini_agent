from agent import run_agent
from rag import retrieve

def run_agent_with_rag(task:str,city:str):
    docs=retrieve(task+city)
    enriched_task=f"""
    城市：{city}

相关背景知识：
{docs[0]}

用户任务：
{task}
"""
    return run_agent(enriched_task)