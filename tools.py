from logger import default_logger as logger
from langchain.tools import tool
from rag import retriever
from city_parser import extract_city
import asyncio
from flashrank import Ranker, RerankRequest
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.tools.tavily_search import TavilySearchResults
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
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
    target_city = extract_city(query)
    try:
        base_docs = await retriever.vectorstore.as_retriever(
            search_kwargs={"k": 10}
        ).ainvoke(query)
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
            for i, doc in enumerate(base_docs)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)
        logger.info(f"Rerank 后的最高分文档内容: {results[0]['text'][:50]}...")
        top_n = 3
        reranked_content = [res["text"] for res in results[:top_n]]
        return "\n\n".join(reranked_content)
    except Exception as e:
        logger.error(f"Rerank 检索失败: {e}")
        # 降级处理：如果 Rerank 出错，使用基础检索
        docs = await retriever.ainvoke(query)
        return "\n\n".join([d.page_content for d in docs])
@tool
async def search_online(query:str):
     """当本地知识库无法回答，或者需要查询最新的天气、活动、突发情况时，调用此在线搜索工具。"""
     search=TavilySearchResults(k=3)
     results=await search.ainvoke(query)
     return results
#工具可以是数据库、接口、搜索、计算器