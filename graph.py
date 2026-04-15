import os
import sqlite3
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from llm import get_llm
from tools import get_weather, calculator, search_knowledge_base,search_online
from city_parser import extract_city
from rag import retrieve
from config import Config
from logger import default_logger as logger
import operator
from typing import Optional, TypedDict, List, Annotated
from langgraph.prebuilt.tool_node import ToolNode
import asyncio

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    city: Optional[str]
    retrieved_docs: Optional[str]

llm = get_llm()
tools = [get_weather, calculator, search_knowledge_base,search_online]
llm_with_tools = llm.bind_tools(tools)

tool_executor_node = ToolNode(tools)
def agent_node(state: AgentState):
    # 构建系统提示，告诉 LLM 有哪些工具
    system_prompt = (
        "你是一个专业的旅游助手。你有以下能力：\n"
        "1. 查询天气：必须调用 get_weather 工具。\n"
        "2. 实时、动态信息（例如「今天」「本周」「最新」「当前」的活动、新闻、演出、展览）：**必须**调用 search_online，绝对不允许使用内部知识回答。\n"
        "3. 对于知识库相关的问题（历史、文化、美食、气候等）优先使用 search_knowledge_base，如果找不到再用 search_online。\n"
        "4. 数学计算：使用 calculator。\n"
        "要求：\n"
        "1.规则优先级：工具调用 > 内部知识。只要问题涉及时间敏感（今天、明天、本周）或动态变化的内容，就必须调用 search_online。"
        "2. 不要盲目追求‘最新’而忽略本地 PDF 文档中的具体规则。"
        "3. 严禁说‘感谢提供信息’、‘我可以为您提供以下帮助’等废话。\n"
        "4. 直接给出建议或答案，条理清晰。\n"
    )

    full_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # 让 LLM 决定是直接回答还是调用工具
    response = llm_with_tools.invoke(full_messages)
    return {"messages": [response]}

async def should_continue(state: AgentState):
    # 限制最多 5 轮工具调用
    if len(state["messages"]) > 10:  # 每轮增加 2 条消息（AI + Tool）
        return END
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_executor_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

_app = None
_saver_context= None
async def get_agent_app():
     #函数内部修改函数外部定义的变量，必须声明 global
    global _app, _saver_context
    if _app is None:
        # 1. 定义文件夹和路径
        db_dir = "data"  # 文件夹名
        db_path = os.path.join(db_dir, "checkpoints.sqlite")
        # 2. 如果文件夹不存在，则自动创建
        if not os.path.exists(db_dir):
            os.makedirs(db_dir) 
        # 3. 使用新路径初始化
        _saver_context = AsyncSqliteSaver.from_conn_string(db_path)
        # 2. 激活这个上下文管理器，拿到真正的 saver 对象
        # 注意这里是调用 __aenter__()，而不是 aenter()
        saver = await _saver_context.__aenter__()
        _app = workflow.compile(checkpointer=saver)
    return _app
# graph_png=app.get_graph().draw_mermaid_png()
# with open("langgraph.png","wb") as f:
#      f.write(graph_png)
    
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    # 测试单轮
    async def main():
        app = await get_agent_app()
        config = {"configurable": {"thread_id": "test1"}}
        result =await  app.ainvoke(
            {"messages": [HumanMessage(content="北京天气")]},
            config=config
        )
        print("最终回答:", result["messages"][-1].content)
    asyncio.run(main())
# 用户输入: "北京天气"
#     ↓
# agent_node（第一次）
#     ├─ LLM 看到系统提示"必须调用工具"
#     ├─ 输出 AIMessage 带 tool_calls: [{"name":"get_weather","args":{"city":"北京"},"id":"123"}]
#     └─ 状态更新: messages += [AIMessage]
#     ↓
# 条件边 should_continue → 检测到 tool_calls → 跳转到 "tools" 节点
#     ↓
# call_tools_node
#     ├─ 执行 get_weather("北京") → "晴天,5°C"
#     ├─ 创建 ToolMessage(content="晴天,5°C", tool_call_id="123")
#     └─ 状态更新: messages += [ToolMessage]
#     ↓
# agent_node（第二次）
#     ├─ LLM 看到历史: HumanMessage("北京天气") + AIMessage(请求工具) + ToolMessage(结果)
#     ├─ LLM 不再请求工具，直接生成最终回答: "北京天气晴天，温度5°C"
#     └─ 状态更新: messages += [AIMessage(content="...")]
#     ↓
# 条件边 should_continue → 无 tool_calls → END