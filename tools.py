from logger import default_logger as logger
from langchain.tools import tool
from rag import retriever
import asyncio
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.tools.tavily_search import TavilySearchResults
@tool
async def get_weather(city:str) -> str:
    """获取指定城市的天气信息。城市名必须是中文，如'北京'、'上海'、'广州'。"""
    logger.info(f"调用工具 get_weather,城市={city}")
    await asyncio.sleep(1)
    fake_weather={
        "北京":"晴天,5°C",
        "上海":"多云,8°C",
        "广州": "小雨, 15°C"
    }
    return fake_weather.get(city, "暂无该城市天气数据")
@tool
async def calculator(expression:str)->str:
    """计算数学表达式，例如'2+3*4'"""
    logger.info(f"调用工具 calculator,表达式={expression}")
    try:
        result=eval(expression,{"__built__":{}},{})
        return f"计算结果:{result}"
    except Exception as e:
        return f"计算错误: {e}"
@tool
async def search_knowledge_base(query: str) -> str:
    """当用户询问关于气候、文化、美食或旅游建议等本地知识时，搜索知识库获取准确信息。"""
    logger.info(f"调用检索工具, 查询词={query}")
    try:
        #引入Reranker
        compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=3)
        # 3. 包装成压缩检索器
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=retriever
        )
        
        docs = await compression_retriever.ainvoke(query)
        if not docs:
                return "not_found"
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        logger.error(f"Rerank 检索失败: {e}")
        # 降级处理：如果 Rerank 出错，使用基础检索
        docs = await retriever.ainvoke(query)
        return "\n\n".join([d.page_content for d in docs])


#工具可以是数据库、接口、搜索、计算器