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
        max_retries=3
    )

client=OpenAI(
   api_key=Config.OPENAI_API_KEY,    
   base_url=Config.OPENAI_BASE_URL
)