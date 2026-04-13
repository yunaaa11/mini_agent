from openai import OpenAI
from langchain_openai import ChatOpenAI
from config import Config
from logger import default_logger as logger
Config.validate()#启动时快速失败，提示你补充 .env 中的配置。

def get_llm(timeout=None):
    """返回 LangChain 的 ChatOpenAI 实例"""
    return ChatOpenAI(
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_BASE_URL,
        model=Config.LLM_MODEL,
        temperature=Config.temperature,
        timeout=timeout or Config.LLM_TIMEOUT,
        max_retries=2
    )

client=OpenAI(
   api_key=Config.OPENAI_API_KEY,    
   base_url=Config.OPENAI_BASE_URL
)

# def call_llm(messages):
#      # 向 LLM 发送对话请求
#     logger.debug(f"调用 LLM，消息数: {len(messages)}")
#     try:
#         response=client.chat.completions.create(
#         model=Config.LLM_MODEL,
#         messages=messages, # 对话上下文
#         temperature=0, # 温度为 0，输出更稳定、确定性更强（适合 Agent / Tool）
#         timeout=Config.LLM_TIMEOUT
#     )
#     # 返回模型生成的最终文本内容
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"[LLM_ERROR] {e}"