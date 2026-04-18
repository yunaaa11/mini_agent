# config.py
import os
from dotenv import load_dotenv
load_dotenv()
# load_dotenv()  # 读取 .env 文件 一定要在你需要读取环境变量之前执行，否则 os.getenv 得不到值。

class Config:
    # ===== LLM 配置 =====
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen-turbo")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 60))
    temperature=float(os.getenv("temperature", 0.0))

    #向量库
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
    PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
    KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "./knowledge_base")
    chunk_size = int(os.getenv("chunk_size", 500))
    chunk_overlap = int(os.getenv("chunk_overlap", 50))
    separators=os.getenv("separators")
    TAVILY_API_KEY =os.getenv("TAVILY_API_KEY")
    MD5_RECORD_FILE = os.getenv('MD5_RECORD_FILE', './processed_md5.txt')
    AMAP_KEY=os.getenv("AMAP_KEY")

    # ===== 功能开关 =====
    ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"
    
     # LangSmith 配置
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "mini_agent")

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 未配置")
        if not cls.OPENAI_BASE_URL:
            raise RuntimeError("OPENAI_BASE_URL 未配置")
# 读取 .env
# 转换类型（bool / int）
# 提供统一入口