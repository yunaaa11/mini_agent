import logging
import sys
import os
from config import Config
def setup_logger(name:str="agent")->logging.Logger:
    """配置日志对象，同时输出到文件和控制台"""
    # 确保 logs 文件夹存在
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)
    #控制台
    console=logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    # 文件处理器 确定存哪
    file_handler = logging.FileHandler("logs/agent.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# 全局默认 logger
default_logger = setup_logger()