
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage
from graph import get_agent_app
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from logger import default_logger as logger
import asyncio
async def run_agent(task:str, session_id: str = "default")-> AsyncGenerator[str, None]:
    """
    执行 Agent,支持多轮对话记忆。
    session_id: 区分不同会话，相同 session_id 会记住历史。
    """
    app = await get_agent_app()
    logger.info(f"收到任务: {task}", extra={"session_id": session_id})
    config={"configurable":{"thread_id":session_id}}
    input_data={"messages":[HumanMessage(content=task)]}
    last_type = None
    # 使用 v2 版本的 stream_events 监听模型生成的 token
    async for event in app.astream_events(input_data,config,version="v2"):
        kind=event["event"]
        # 只处理 chat_model 的流输出，不要处理任何 chain 或 tool 的事件
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            
            # 过滤掉所有思考过程，只取 content
            if hasattr(chunk, 'content') and chunk.content:
                # 过滤掉那些长得像日志的废话
                if "事实核查结论" in chunk.content or "ping -" in chunk.content:
                    continue
                yield chunk.content

    # 善后：如果流程结束了 last_type 还是 thought，强制关闭它
    if last_type == "thought":
        yield "[THOUGHT_END]"

if __name__ == "__main__":
    # 测试多轮记忆
    print("=== 第一轮 ===")
    print(run_agent("北京天气怎么样", session_id="user1"))
    print("\n=== 第二轮（应该记住上一轮话题） ===")
    print(run_agent("那上海呢", session_id="user1"))
