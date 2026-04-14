from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
import os
from rag_data import DOCS


# 初始化 embeddings
embeddings = OpenAIEmbeddings(
    api_key=Config.OPENAI_API_KEY,
    base_url=Config.OPENAI_BASE_URL,
    model=Config.EMBEDDING_MODEL,
    check_embedding_ctx_length=False,
)

def load_documents_from_folder(folder_path:str):
    """从指定目录加载所有.txt文件,并切分成小块"""
    all_docs=[]
    if not os.path.exists(folder_path):
        print(f"创建目录 {folder_path}，请将知识库 .txt 文件放入")
        return []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):#只处理 .txt 文件。
            file_path=os.path.join(folder_path,file)
            loader=TextLoader(file_path,encoding="utf-8")
            #返回一个列表，里面通常只有一个 Document 对象（整个文件内容作为一个文档）。
            docs=loader.load()
            #可选：添加元数据，记录来源文件
            for doc in docs:
                doc.metadata["source"]=file
            all_docs.extend(docs)
    #将长文档切分成小段，提高检索精度
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=Config.chunk_size,
        chunk_overlap=Config.chunk_overlap,
        separators=Config.separators
    )
    #执行切分，返回新的 Document 列表，每个文档的 page_content 是一个小段落。
    split_docs=splitter.split_documents(all_docs)
    print(f"加载了 {len(all_docs)} 个原始文档，切分成 {len(split_docs)} 个文本块")
    return split_docs

def get_vectorstore():
    """获取或创建Chroma向量库"""
    if os.path.exists(Config.PERSIST_DIR) and os.listdir(Config.PERSIST_DIR):
        return Chroma(persist_directory=Config.PERSIST_DIR, embedding_function=embeddings)
    else:
        # 首次运行，从文件夹加载并创建向量库
        docs = load_documents_from_folder(Config.KNOWLEDGE_DIR)
        if not docs:
            print("警告：没有找到任何 .txt 文档，将创建空向量库")
            # 创建一个空的 Chroma 库，避免后续调用出错
            return Chroma(embedding_function=embeddings, persist_directory=Config.PERSIST_DIR)
        vectorstore=Chroma.from_documents(docs,embeddings,persist_directory=Config.PERSIST_DIR)
        vectorstore.persist()
        return vectorstore
#先有向量库再检索
vectorstore=get_vectorstore()
retriever=vectorstore.as_retriever(search_kwargs={"k":3})

def retrieve(query:str,top_k:int=1):
    """兼容旧接口，返回文档字符串列表"""
    docs=retriever.invoke(query)
    return [doc.page_content for doc in docs[:top_k]]


if __name__ == "__main__":
    # 测试检索
    query = "北京冬天冷不冷"
    docs = retrieve(query, top_k=1)
    print(f"查询: {query}")
    print(f"检索结果: {docs[0] if docs else '无'}")
# 文档 → 向量（离线）
# 用户问题 → 向量
# ↓
# 余弦相似度
# ↓
# 取最相似文档
