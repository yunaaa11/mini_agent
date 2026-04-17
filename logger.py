import logging
from logging.handlers import RotatingFileHandler
import os

# 1. 动态注入 Session ID 的过滤器
class ContextFilter(logging.Filter):
    def filter(self, record):
        # 如果 record 里没有 session_id，给个默认值
        if not hasattr(record, "session_id"):
            record.session_id = "SYSTEM"
        return True

def setup_logger():
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加 Handler
    if not logger.handlers:
        # 2. 改进：日志轮转 (每个文件 5MB, 保留 5 个备份)
        log_file = "logs/agent.log"
        handler = RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
        )
        
        # 3. 改进：格式化加入 session_id
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(session_id)s] - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.addFilter(ContextFilter())
        
        # 同时输出到控制台方便调试
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        
    return logger

default_logger = setup_logger()