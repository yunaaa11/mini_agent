# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # 读取 .env 文件 一定要在你需要读取环境变量之前执行，否则 os.getenv 得不到值。

class Config:
    # ===== LLM 配置 =====
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen-turbo")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 15))

    # ===== 功能开关 =====
    ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 未配置")
        if not cls.OPENAI_BASE_URL:
            raise RuntimeError("OPENAI_BASE_URL 未配置")
# 读取 .env
# 转换类型（bool / int）
# 提供统一入口