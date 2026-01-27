from openai import OpenAI
from config import Config
Config.validate()
# load_dotenv()  # 读取 .env
# 创建 LLM 客户端实例
# 使用的是阿里云 DashScope 提供的 OpenAI 兼容接口
client=OpenAI(
   api_key=Config.OPENAI_API_KEY,    
   base_url=Config.OPENAI_BASE_URL
)
# MODEL = os.getenv("LLM_MODEL", "qwen-turbo")
def call_llm(messages):
    #  """
    # 调用大语言模型（LLM）的统一封装函数

    # :param messages: 对话消息列表，格式符合 OpenAI Chat Completion 规范
    #                  例如：
    #                  [
    #                    {"role": "system", "content": "..."},
    #                    {"role": "user", "content": "..."}
    #                  ]
    # :return: LLM 生成的文本内容（字符串）
    # """
     # 向 LLM 发送对话请求
    try:
        response=client.chat.completions.create(
        model=Config.LLM_MODEL,
        # model="qwen-turbo", # 使用通义千问 qwen-turbo 模型
        messages=messages, # 对话上下文
        temperature=0, # 温度为 0，输出更稳定、确定性更强（适合 Agent / Tool）
        timeout=Config.LLM_TIMEOUT
        # timeout=15
    )
    # 返回模型生成的最终文本内容
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM_ERROR] {e}"