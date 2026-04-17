from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from config import Config
import os
from rag_data import DOCS
import hashlib
from logger import default_logger as logger

# 初始化 embeddings
embeddings = OpenAIEmbeddings(
    api_key=Config.OPENAI_API_KEY,
    base_url=Config.OPENAI_BASE_URL,
    model=Config.EMBEDDING_MODEL,
    check_embedding_ctx_length=False,
)
#md5去重模块

def get_text_md5(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()
#防止重复添加内容完全相同的文件
def is_md5_processed(md5_hash: str) -> bool:
    if not os.path.exists(Config.MD5_RECORD_FILE):
        return False
    with open(Config.MD5_RECORD_FILE, 'r', encoding='utf-8') as f:
        processed = set(line.strip() for line in f)
    return md5_hash in processed
#记录下来，下次遇到相同内容就会跳过。
def mark_md5_processed(md5_hash: str):
    with open(Config.MD5_RECORD_FILE, 'a', encoding='utf-8') as f:
        f.write(md5_hash + '\n')


def load_documents_from_folder(folder_path:str):
    """从指定目录加载所有.txt、.pdf文件,并切分成小块"""
    if not os.path.exists(folder_path):
        print(f"创建目录 {folder_path}，请将知识库文件放入")
        return []
    
    all_docs=[]
    for file in os.listdir(folder_path):
        if not (file.endswith('.txt') or file.endswith('.pdf')):
            #遇到其他文件（如 .docx、.md、.jpg）就跳过。
            continue
        #作用：拼接文件的完整路径（例如 ./knowledge_base/北京.txt）
        file_path = os.path.join(folder_path, file)
        try:
            if file.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
            else:  # PDF
                loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())
        except Exception as e:
            logger.error(f"加载文件 {file} 失败: {e}")
    #将长文档切分成小段，提高检索精度
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=Config.chunk_size,
        chunk_overlap=Config.chunk_overlap,
        separators=Config.separators
    )
    return splitter.split_documents(all_docs)
# 定义两个全局变量
retriever = None
vectorstore_obj = None # 新增一个全局变量来存 vectorstore

def get_hybrid_retriever():
    global vectorstore_obj # 声明使用全局变量
    
    # 1. 初始化向量库
    if os.path.exists(Config.PERSIST_DIR) and os.listdir(Config.PERSIST_DIR):
        vs = Chroma(persist_directory=Config.PERSIST_DIR, embedding_function=embeddings)
    else:
        docs = load_documents_from_folder(Config.KNOWLEDGE_DIR)
        vs = Chroma.from_documents(docs, embeddings, persist_directory=Config.PERSIST_DIR)
    
    vectorstore_obj = vs # 将 vs 赋值给全局变量
    
    vector_retriever = vs.as_retriever(search_kwargs={"k": 3})
    
    # 2. 初始化 BM25
    all_docs = load_documents_from_folder(Config.KNOWLEDGE_DIR)
    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 3
        
        ensemble_retriever = EnsembleRetriever(
             retrievers=[bm25_retriever, vector_retriever], # 注意：这里是你上次改对的 retrievers
             weights=[0.4, 0.6]
        )
        return ensemble_retriever
    else:
        return vector_retriever
#全局初始化检索器
retriever=get_hybrid_retriever()
vectorstore = vectorstore_obj

def retrieve(query:str,top_k:int=5):
    """使用混合检索获取文档内容"""
    try:
        docs=retriever.invoke(query)
        return [doc.page_content for doc in docs[:top_k]]
    except Exception as e:
        logger.error(f"检索失败:{e}")
        return []

def add_document_from_text(content:str,filename:str,metadata:dict=None):
    """
   增量添加文档到向量库（重启程序来刷新BM25索引）
    """
    #md5去重检查
    file_md5=get_text_md5(content)
    if is_md5_processed(file_md5):
        print(f"文件 {filename} 内容未变化，跳过添加")
        return False
    #构造Document(未切分)
    doc_metadata={"source":filename}
    if metadata:
        doc_metadata.update(metadata)
    doc=Document(page_content=content,metadata=doc_metadata)
    #切分文档（与全量加载相同的切分器）
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=Config.chunk_size,
        chunk_overlap=Config.chunk_overlap,
        separators=Config.separators
    )
    split_docs=splitter.split_documents([doc])
    #更新向量库
    vs = Chroma(persist_directory=Config.PERSIST_DIR, embedding_function=embeddings)
    vs.add_documents(split_docs)
    #记录md5
    mark_md5_processed(file_md5)
    print(f"成功添加{len(split_docs)}个文档块，来自{filename}")
    return True


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
