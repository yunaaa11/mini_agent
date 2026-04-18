import datetime
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
    rewrite_query: Optional[str]

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
async def router(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1].content.lower()
    # 路径 1: 闲聊/简单打招呼 -> direct_reply
    greetings = ["你好", "hi", "hello", "在吗", "早上好"]
    if any(g in last_message for g in greetings) and len(last_message)<10:
        return "direct_reply"
    
    # 2. 使用 LLM 进行意图识别 (语义路由)
    router_llm = get_llm(timeout=10) # 路由不需要太高温度
    prompt = f"""分类用户输入的意图。只需回答 "booking" 或 "general"。
    - 用户想买票、订票、补全身份信息进行预订：回答 "booking"
    - 用户问天气、景点、攻略、美食、计算或其他信息：回答 "general"
    
    用户输入: "{last_message}"
    意图分类:"""
    
    response = await router_llm.ainvoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()

    if "booking" in decision:
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
    last_message = state["messages"][-1]
    # 核心修正：如果最后一条消息是工具返回的结果(ToolMessage)或反思结果(AIMessage且包含【事实核查】)
    # 则【跳过】重写逻辑，直接让 LLM 总结回答
    is_tool_result = isinstance(last_message, ToolMessage) or (
        isinstance(last_message, AIMessage) and "【事实核查】" in last_message.content
    )

    if is_tool_result:
        search_query = state.get("rewrite_query", "总结检索结果")
        logger.info(f"--- 识别到工具结果，跳过重写，直接总结 ---")
    else:
        # 只有在处理用户新提问时才重写
        rewrite_llm = get_llm(temperature=0)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        rewrite_prompt = f"""当前日期是: {current_date}。
        根据对话历史，将用户的最新问题改写为一个完整、具体的搜索词。
        历史记录: {state["messages"][-3:]} 
        最新问题: {last_message.content}
        改写后的搜索词:"""
    
        rewrite_res = await rewrite_llm.ainvoke([HumanMessage(content=rewrite_prompt)])
        search_query = rewrite_res.content.strip()
        logger.info(f"--- 查询重写: {last_message.content} -> {search_query} ---")
    
    # 构建系统提示，告诉 LLM 有哪些工具
    system_prompt = (
    f"今天是 {datetime.datetime.now().strftime('%Y-%m-%d')}。\n"
    "你是一个极其严谨的专家级旅游助手。在回答之前，你必须执行以下逻辑与验证机制：\n\n"
    "1. 【强力 Grounding 与事实溯源】：\n"
    "   - **禁止输出任何本地文档中未提及的数字、年份或具体细节。** 即使你拥有外部知识，如果文档没写，你就必须视而不见。\n"
    "   - 严禁仅凭记忆回答。气候规律、地道美食和文化背景必须调用 search_knowledge_base。\n"
    "   - **规则优先级**：本地 PDF 文档中的建议 > 搜索引擎信息 > 你的内部预训练知识。\n\n"
    "2. 【天气与时效信息验证】：\n"
    "   - **实时天气**：必须调用 get_weather 获取温度和基础状况。\n"
    "   - **风险验证**：必须配合 search_online 验证今日是否有突发气象预警、自然灾害或景区闭园等动态信息，严禁仅靠本地文档回答“今天”的情况。\n\n"
    "3. 【约束过度服务与反问机制】：\n"
    "   - 如果用户问题指代不明（例如只问“天气”或“攻略”），且检索内容涉及多个城市，**必须反问用户是指哪个城市**，严禁盲目猜测或一次性列出所有城市。\n"
    "   - 只有当信息全备且指向明确时，才给出最终建议。\n\n"
    "4. 【降级补偿机制】：\n"
    "   - 若 get_weather 失败或 search_knowledge_base 无结果，必须立即启动 search_online 补全信息，绝对禁止回答“不知道”或“查不到”。\n"
    "   - 只有当 API 数据、本地文档和在线搜索结果相互验证后，才给出最终建议。\n\n"
    "5. 【交互规范】：\n"
    "   - **拒绝推诿**：你拥有订票、查询、全权处理建议的能力，严禁说“我没有权限”或“建议您去12306”。\n"
    "   - **拒绝废话**：严禁说“我可以为您提供帮助”、“感谢提供信息”。\n"
    "   - **输出要求**：直接给出结构化答案。条理清晰地区分：『实时天气』、『穿衣/气候建议』与『旅游专家提醒』。\n\n"
    f"当前处理意图：{search_query}"
)
    full_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    # 让 LLM 决定是直接回答还是调用工具
    response = await llm_with_general_tools.ainvoke(full_messages)
    return {"messages": [response], "rewrite_query": search_query}

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
    """
    自检逻辑：
    1. 如果 LLM 提出了 tool_calls -> 去执行工具。
    2. 如果工具刚刚执行完（最后一条是 ToolMessage），检查内容是否包含报错。
    3. 如果报错 -> 返回 booking_expert 让 LLM 反思并修复，而不是结束。
    """
    last_msg = state["messages"][-1]
    # 情况 A: LLM 决定调用工具
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "booking_tools"
    # 情况 B: 工具执行完毕后的自检
    if isinstance(last_msg,ToolMessage):
        content=last_msg.content
        error_keywords=["错误","失败","无效""Error", "不能为空"]
        if any(kw in content for kw in error_keywords):
            logger.warning(f"--- [自检系统] 发现执行异常，返回专家节点修正 ---")
            return "booking_expert" # 跳回专家节点，让 AI 看报错信息并回复用户
    return END

async def reflection_node(state:AgentState):
    last_msg=state["messages"][-1]
    #只有上一条工具返回内容才反思，否则跳过
    if not isinstance(last_msg,ToolMessage):
        return state
    # 优化：只有知识库检索到的内容才需要反思核对，天气等动态数据直接跳过
    if last_msg.name != "search_knowledge_base":
        return state
        
    logger.info("--- 针对知识库内容进行深度核对 ---")
    logger.info("--- 进入反思节点：核对检索事实 ---")
    checker_llm = get_llm(temperature=0)
    check_prompt =f"""
    ### 核心指令 (CRITICAL INSTRUCTION) ###
    1. 你现在是一个完全没有外部记忆、没有常识、没有个人偏好的纯粹事实核查机器人。
    2. 你的整个世界、所有的真相，仅限于下方提供的【原始检索内容】。
    3. 严禁利用你预训练学到的知识来修正文档。即使文档说“地球是方的”，你也必须认为地球是方的。
    
    【逻辑优先规则】：
    - 寻找关键词：'实际'、'但实际上'、'其实'。
    - **必须采信转折后的陈述**。例如文档说“传闻周一闭馆，但实际上全年无休”，你必须输出“全年无休”。

    【原始检索内容】: 
    {last_msg.content}

    请直接输出核心事实陈述，严禁废话："""
    response=await checker_llm.ainvoke([HumanMessage(content=check_prompt)])
    # 将校对后的事实作为一条 AI 的辅助记忆插入
    reflection_msg = AIMessage(content=f"【事实核查】：{response.content}")
    return {"messages": [reflection_msg]}

workflow = StateGraph(AgentState)
workflow.add_node("direct_reply",direct_reply_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", general_tools_node)
workflow.add_node("booking_expert", booking_expert_node)
workflow.add_node("booking_tools",booking_tools_node)
workflow.add_node("reflection", reflection_node)
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
# 通用链路：agent -> should_continue_general -> tools -> reflection -> agent
workflow.add_conditional_edges("agent", should_continue_general, {"tools": "tools", END: END})
workflow.add_edge("tools", "reflection")
workflow.add_edge("reflection", "agent")
# 业务订票链路
workflow.add_conditional_edges("booking_expert", should_continue_booking, {"booking_tools": "booking_tools", END: END})
workflow.add_conditional_edges("booking_tools", should_continue_booking,
    {"booking_expert": "booking_expert", END: END}
)

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
        # 1. 获取编译好的 app
        app = await get_agent_app()
        try:
            # 尝试生成 Mermaid 图片
            graph_png = app.get_graph().draw_mermaid_png()
            with open("langgraph_v2.png", "wb") as f:
                f.write(graph_png)
            print("--- 流程图已生成至: langgraph_v2.png ---")
        except Exception as e:
            print(f"--- 流程图生成失败（可能是缺少 pygraphviz）: {e} ---")

        config = {"configurable": {"thread_id": "user_session_123"}}
        
        print("--- 旅游助理已上线（输入 'exit' 退出） ---")
        
        while True:
            # 2. 获取用户输入
            task = input("\n用户: ")
            if task.lower() in ["exit", "quit", "退出"]:
                break
            
            # 3. 构造初始状态
            inputs = {"messages": [HumanMessage(content=task)]}
            
            print("助理: ", end="", flush=True)

            # 4. 使用 stream_mode="messages" 进行流式输出
            # msg 是消息片段，metadata 包含当前节点信息
            async for msg, metadata in app.astream(inputs, config=config, stream_mode="messages"):
                node_name = metadata.get("langgraph_node")
                
                # 调试用：看看现在运行到哪个节点了
                # print(f"\n[DEBUG] Node: {node_name}, Type: {type(msg)}") 

                if isinstance(msg, AIMessage):
                    # 如果有文本内容，直接打印
                    if msg.content:
                        print(msg.content, end="", flush=True)
                    
                    # 如果没有文本但在调用工具，给个反馈
                    elif msg.tool_calls:
                        print(f"\n助理: (正在尝试调用工具: {msg.tool_calls[0]['name']}...)", flush=True)
                
                # 如果是工具返回的消息，你也可以选择性打印
                elif isinstance(msg, ToolMessage):
                    print("\n助理: (已获取到相关信息，正在整理回答...)", flush=True)

    asyncio.run(main())
