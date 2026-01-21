import json
from llm import call_llm
from tools import get_weather
SYSTEM_PROMPT= """
你是一个严格的工具型 Agent,不是聊天助手。

你只能做两件事之一：

【情况一】需要调用工具  
你必须只返回 JSON,禁止返回任何解释性文字，格式如下：
{
  "tool": "get_weather",
  "args": {
    "city": "<从用户任务中识别的城市名>"
  }
}

【情况二】工具已经返回结果  
你必须基于工具结果，用中文给出最终答案。

⚠️ 重要规则：
- 如果用户问题涉及“天气”，你【必须】调用 get_weather 工具
- 第一次回复【只能】是 JSON 或最终答案
- 禁止说“请稍等 / 我将为您 / 好的”
- 禁止寒暄
"""
def run_agent(task:str):
    # 初始化对话消息列表
    # system：定义智能体的整体行为和规则
    # user：用户当前任务
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":task}
    ]
    # 第一次：让 LLM 决定是否调用工具
    reply=call_llm(messages)
    print("[LLM 第一次回复]")
    print(reply)
    try:
        # 尝试将 LLM 的回复解析为 JSON
        # 如果能解析，说明 LLM 可能在请求调用工具
        tool_call=json.loads(reply)
        # 判断是否是天气查询工具
        if tool_call["tool"]=="get_weather":
            # 从工具参数中取出城市信息
           city=tool_call["args"]["city"]
           # 调用真实工具（非 LLM）
           result=get_weather(city)
           # 把 LLM 的“工具调用请求”加入上下文
           messages.append({"role":"assistant","content":reply})
            # 把工具执行结果返回给 LLM
            # 让它基于真实数据生成最终回答
           messages.append({
               "role":"user",
               "content":f"工具返回结果：{result}，请给用户最终回答"
           })
           # 第二次调用 LLM
            # 目的：生成最终自然语言回复
           final_ansewer=call_llm(messages)
           return final_ansewer
    except Exception:
       # 如果：
        # - JSON 解析失败
        # - 或 LLM 没有请求工具
        # 则直接返回第一次 LLM 回复
        return reply
    
# #用户任务
#    ↓
# #LLM 第一次回复（决定是否用工具）
#    ↓
# #是否是 JSON 工具调用？
#    ├── 否 → 直接回答
#    └── 是
#         ↓
#      调用真实工具
#         ↓
#      工具结果 → 再喂给 LLM
#         ↓
#      LLM 生成最终回答
