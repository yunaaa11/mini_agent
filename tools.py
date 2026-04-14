from config import Config
from logger import default_logger as logger
from langchain.tools import tool
from rag import retriever
from city_parser import extract_city
import asyncio
from flashrank import Ranker, RerankRequest
from langchain_community.document_compressors import FlashrankRerank
from langchain_tavily import TavilySearch
from asteval import Interpreter
from cachetools import TTLCache
import hashlib
aeval = Interpreter()
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
vector_retriever =retriever.vectorstore.as_retriever(search_kwargs={"k": 10})
cache = TTLCache(maxsize=100, ttl=300)
def get_cache_key(prefix: str, *args) -> str:
    """生成缓存键，例如 'weather:北京' -> MD5 或直接字符串"""
    raw = f"{prefix}:{':'.join(str(a) for a in args)}"
    # 如果键过长可 MD5 压缩，这里直接用原字符串（长度通常不长）
    return raw

@tool
async def get_weather(city:str) -> str:
    """获取指定城市的天气信息。城市名必须是中文，如'北京'、'上海'、'广州'。"""
    logger.info(f"调用工具 get_weather,城市={city}")
    # 1. 检查缓存
    cache_key = get_cache_key("weather", city)
    if cache_key in cache:
        logger.info(f"天气缓存命中: {city}")
        return cache[cache_key]
    
    await asyncio.sleep(1)
    fake_weather={
        "北京":"晴天,5°C",
        "上海":"多云,8°C",
        "广州": "小雨, 15°C"
    }
    result=fake_weather.get(city, "暂无该城市天气数据")
    # 3. 存入缓存
    cache[cache_key] = result
    return result

@tool
async def calculator(expression:str)->str:
    """计算数学表达式，例如'2+3*4'"""
    logger.info(f"调用工具 calculator,表达式={expression}")
    try:
        # 在受限环境中执行表达式
        result = aeval(expression)
        if aeval.error:
            # 如果表达式不合法（例如包含非法函数或语法错误）
            return f"计算错误: {aeval.error}"
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"
@tool
async def search_knowledge_base(query: str) -> str:
    """当用户询问关于气候、文化、美食或旅游建议等本地知识时，搜索知识库获取准确信息。"""
    logger.info(f"调用检索工具, 查询词={query}")
    target_city = extract_city(query)
    # 生成缓存键（区分有无城市）
    cache_key = get_cache_key("kb", query, target_city if target_city else "none")
    if cache_key in cache:
        logger.info(f"知识库缓存命中: {query[:30]}...")
        return cache[cache_key]
    try:
        base_docs = await vector_retriever.ainvoke(query)
        # 3. 【核心改动】手动过滤：只保留包含目标城市名称的文档
        if target_city:
            filtered_docs = [
                doc for doc in base_docs 
                if target_city in doc.page_content # 只有文本里带“上海”的才留下
            ]
            logger.info(f"过滤前 {len(base_docs)} 条，过滤后 {len(filtered_docs)} 条")
        else:
            filtered_docs = base_docs
        if not filtered_docs:
            return "未找到相关知识。"
        # Step 2: 准备 Rerank 数据格式
        # FlashRank 需要一个包含 id, text, meta 的字典列表
        passages = [
            {
                "id": i,
                "text": doc.page_content,
                "meta": doc.metadata
            }
            for i, doc in enumerate(filtered_docs)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = await asyncio.to_thread(ranker.rerank, rerank_request)
        logger.info(f"Rerank 后的最高分文档内容: {results[0]['text'][:50]}...")
        top_n = 3
        reranked_content = [res["text"] for res in results[:top_n]]
        final_result = "\n\n".join(reranked_content)
    except Exception as e:
        logger.error(f"Rerank 检索失败: {e}")
        # 降级处理：如果 Rerank 出错，使用基础检索
        docs = await retriever.ainvoke(query)
        final_result = "\n\n".join([d.page_content for d in docs])
     # 存入缓存
        cache[cache_key] = final_result
        return final_result
@tool
async def search_online(query:str):
     """当本地知识库无法回答，或者需要查询最新的天气、活动、突发情况时，调用此在线搜索工具。"""
     logger.info(f"调用在线搜索工具, 查询词={query}")
     try:
        search=TavilySearch(api_key=Config.TAVILY_API_KEY,k=3)
        results=await search.ainvoke(query)
        if isinstance(results, list):
                formatted = "\n".join([str(r) for r in results])
        else:
                formatted = str(results)
                return formatted
     except Exception as e:
        logger.error(f"在线搜索失败: {e}")
        return f"搜索失败：{e}"
#工具可以是数据库、接口、搜索、计算器