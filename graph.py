from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from llm import get_llm
from tools import get_weather, calculator, search_knowledge_base
from city_parser import extract_city
from rag import retrieve
from config import Config
from logger import default_logger as logger
import operator
from typing import Optional, TypedDict, List, Annotated
from langgraph.prebuilt import ToolNode
import asyncio

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    city: Optional[str]
    retrieved_docs: Optional[str]

llm = get_llm()
tools = [get_weather, calculator, search_knowledge_base]
llm_with_tools = llm.bind_tools(tools)

tool_executor_node = ToolNode(tools)
def agent_node(state: AgentState):
    # 构建系统提示，告诉 LLM 有哪些工具
    system_prompt = (
        "你是一个专业的旅游助手。你有以下能力：\n"
        "1. 查询天气：必须调用 get_weather 工具。\n"
        "2. 旅游建议/本地知识：必须调用 search_knowledge_base 工具查询知识库，不要凭空猜测。\n"
        "3. 数学计算：使用 calculator 工具。"
        "要求：\n"
        "1. 严禁说‘感谢提供信息’、‘我可以为您提供以下帮助’等废话。\n"
        "2. 直接给出建议或答案，条理清晰。\n"
    )

    full_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # 让 LLM 决定是直接回答还是调用工具
    response = llm_with_tools.invoke(full_messages)
    return {"messages": [response]}

# def call_tools_node(state: AgentState):
#     last_msg = state["messages"][-1]
#     # 提取 tool_calls 属性，如果没有则返回空字典（表示没有工具要执行）
#     tool_calls = getattr(last_msg, "tool_calls", [])
#     if not tool_calls:
#         return {}

#     tool_messages = []
#     for tc in tool_calls:
#         tool_name = tc["name"]
#         tool_args = tc["args"]
#         if  tool_name == "get_weather":
#             city = tool_args["city"]
#             result = get_weather(city)
#         elif tool_name == "calculator":
#             expression = tool_args["expression"]
#             result = calculator(expression)
#         elif tool_name == "search_knowledge_base":
#             # 调用你在 tools.py 或 rag.py 里定义的检索函数
#             result = search_knowledge_base.invoke(tool_args)
#         else:
#             result = f"未知工具: {tool_name}"
#         tool_messages.append(ToolMessage(content=result, # 工具返回的结果文本 
#                                             tool_call_id=tc["id"]))
#         logger.info(f"执行工具 {tool_name}，参数={tool_args}，结果={result}")
#     # 返回新消息列表（会被追加到 state["messages"] 中）
#     return {"messages": tool_messages}

async def should_continue(state: AgentState):
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

saver = MemorySaver()
app = workflow.compile(checkpointer=saver)
graph_png=app.get_graph().draw_mermaid_png()
with open("langgraph.png","wb") as f:
     f.write(graph_png)
    
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    # 测试单轮
    async def main():
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