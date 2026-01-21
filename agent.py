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
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":task}
    ]
    # 第一次：让 LLM 决定是否调用工具
    reply=call_llm(messages)
    print("[LLM 第一次回复]")
    print(reply)
    try:
        tool_call=json.loads(reply)
        if tool_call["tool"]=="get_weather":
           city=tool_call["args"]["city"]
           # 调用工具
           result=get_weather(city)
           # 把工具结果发回给 LLM
           messages.append({"role":"assistant","content":reply})
           messages.append({
               "role":"user",
               "content":f"工具返回结果：{result}，请给用户最终回答"
           })
           final_ansewer=call_llm(messages)
           return final_ansewer
    except Exception:
        # 没用工具，直接返回
        return reply