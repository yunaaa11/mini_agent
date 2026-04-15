from langchain_core.messages import HumanMessage
from graph import get_agent_app
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from logger import default_logger as logger
import asyncio
async def run_agent(task:str, session_id: str = "default")->str:
    """
    执行 Agent,支持多轮对话记忆。
    session_id: 区分不同会话，相同 session_id 会记住历史。
    """
   
    app = await get_agent_app()
    logger.info(f"收到任务: {task}, session_id={session_id}")
    config={"configurable":{"thread_id":session_id}}
    final_state=await app.ainvoke(
        {"messages":[HumanMessage(content=task)]},
        config=config
    )
    last_message=final_state["messages"][-1]
    if hasattr(last_message,"content"):
        result=last_message.content
    else:
        result=str(last_message)
    logger.info(f"Agent 返回: {result}")
    return result

if __name__ == "__main__":
    # 测试多轮记忆
    print("=== 第一轮 ===")
    print(run_agent("北京天气怎么样", session_id="user1"))
    print("\n=== 第二轮（应该记住上一轮话题） ===")
    print(run_agent("那上海呢", session_id="user1"))
