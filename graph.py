import os
import sqlite3
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from llm import get_llm
from tools import get_weather, calculator, search_knowledge_base,search_online,book_train_ticket
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
# 通用 Agent 工具集
general_tools = [get_weather, calculator, search_knowledge_base, search_online]
llm_with_general_tools=llm.bind_tools(general_tools)
general_tools_node=ToolNode(general_tools)
# 订票专家工具集
booking_tools = [book_train_ticket]
llm_with_booking_tools=llm.bind_tools(booking_tools)
booking_tools_node=ToolNode(booking_tools)

#路由：根据用户输入特征选择路径
def router(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1].content.lower()
    # 路径 1: 闲聊/简单打招呼 -> direct_reply
    greetings = ["你好", "hi", "hello", "在吗", "早上好"]
    if any(g in last_message for g in greetings) and len(last_message)<10:
        return "direct_reply"
    
    # 路径 2: 订票意图识别与上下文粘滞
    is_in_booking_flow = False
    # 逆向遍历历史消息，查找最近的 AI 消息
    for msg in reversed(messages[:-1]):
        if isinstance(msg, AIMessage):
            # 如果 AI 的上一句包含关键引导词，判定为处于订票流程中
            if any(key in msg.content for key in ["姓名", "身份证", "目的地", "订票"]):
                is_in_booking_flow = True
            break # 只看最近的一条 AI 消息即可，避免跨度太长的“记忆干扰”

    booking_keywords = ["订票", "买票", "票", "下单", "预订", "购票"]
    if any(k in last_message for k in booking_keywords) or is_in_booking_flow:
        logger.info("--- 路由决策: 进入订票专家 ---")
        return "booking_expert"
    
    # 路径 3: 默认走通用 RAG Agent -> agent
    return "agent"

# 快速回复节点
def direct_reply_node(state: AgentState):
    content = "您好！我是您的智能助理。我可以帮您查天气、搜攻略，或者直接帮您订火车票。请问今天有什么可以帮您？"
    return {"messages": [AIMessage(content=content)]}
# 通用 Agent 节点 (负责 RAG 和 外部搜索)
async def agent_node(state: AgentState):
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
        "4.如果用户提到订票相关的后续补充信息，请指引用户继续完成流程，不要说你没有订票权限。"
        "5. 直接给出建议或答案，条理清晰。\n"
    )

    full_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # 让 LLM 决定是直接回答还是调用工具
    response = await llm_with_general_tools.ainvoke(full_messages)
    return {"messages": [response]}

# 订票专家节点 (负责引导用户补全身份信息)
async def booking_expert_node(state:AgentState):
    system_prompt = (
        "你是一个订票专家。预订火车票必须具备：姓名、身份证号、目的地。\n"
        "【核心规则】：\n"
        "1. 只要缺一项信息，就必须针对性地追问，严禁引导用户去 12306。\n"
        "2. 只要信息全了，**必须**调用 book_train_ticket 工具。\n"
        "3. 身份证号必须是 18 位，不符请提示用户重输。"
    )
    full_messages=[SystemMessage(content=system_prompt)]+state["messages"]
    response=await llm_with_booking_tools.ainvoke(full_messages)
    return {"messages":[response]}
async def should_continue_general(state: AgentState):
    # 限制最多 5 轮工具调用
    if len(state["messages"]) > 10:  # 每轮增加 2 条消息（AI + Tool）
        return END
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END
async def should_continue_booking(state: AgentState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "booking_tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("direct_reply",direct_reply_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", general_tools_node)
workflow.add_node("booking_expert", booking_expert_node)
workflow.add_node("booking_tools",booking_tools_node)
# 设置意图入口
workflow.set_conditional_entry_point(
    router,
    {
        "direct_reply": "direct_reply",
        "agent": "agent",
        "booking_expert": "booking_expert"
    }
)
#定义边
workflow.add_edge("direct_reply",END)
# 通用 RAG 链路
workflow.add_conditional_edges("agent", should_continue_general, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
# 业务订票链路
workflow.add_conditional_edges("booking_expert", should_continue_booking, {"booking_tools": "booking_tools", END: END})
workflow.add_edge("booking_tools", "booking_expert")

# --- 持久化与工厂函数 ---
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
    async def main():
        app = await get_agent_app()
        # 核心：这个 ID 决定了记忆的归属
        config = {"configurable": {"thread_id": "user_session_123"}}
        
        print("--- 旅游助理已上线（输入 'exit' 退出） ---")
        
        while True:
            user_input = input("\n用户: ")
            if user_input.lower() in ["exit", "quit", "退出"]:
                break
            
            # 使用 astream 持续监听状态变化
            inputs = {"messages": [HumanMessage(content=user_input)]}
            async for event in app.astream(inputs, config=config, stream_mode="values"):
                # 我们只看最后一项输出（即最新的消息）
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        print(f"助理: {last_msg.content}")
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