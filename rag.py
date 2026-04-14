from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
import os
from rag_data import DOCS
import hashlib

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

        if file.endswith(".txt"):#只处理 .txt 文件,计算 MD5 时需要全文内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
        else:  # PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            content = "\n".join([p.page_content for p in pages])
            #合并为一个 Document（方便 MD5 去重）
            docs = [Document(page_content=content, metadata={"source": file})]
        file_md5=get_text_md5(content)
        if is_md5_processed(file_md5):
            print(f"文件 {file} 内容未变化，跳过加载")
            continue
        #加载文档（可能返回多个Document,入pdf每页一个）
        for doc in docs:
            if "source" not in doc.metadata:
                doc.metadata["source"] = file
        all_docs.extend(docs)
        mark_md5_processed(file_md5)
    #如果加载后没有拿到任何文档（例如文件夹为空，或所有文件都被 MD5 去重跳过了），则直接返回空列表，避免后续切分报错
    if not all_docs:
        return []
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
        return vectorstore
#先有向量库再检索
vectorstore=get_vectorstore()
retriever=vectorstore.as_retriever(search_kwargs={"k":3})

def retrieve(query:str,top_k:int=5):
    """兼容旧接口，返回文档字符串列表"""
    docs=retriever.invoke(query)
    return [doc.page_content for doc in docs[:top_k]]

def add_document_from_text(content:str,filename:str,metadata:dict=None):
    """
    通过文本内容增量添加文档到向量库（支持 MD5 去重）
    :param content: 文档的完整文本内容
    :param filename: 文件名（用于日志和元数据）
    :param metadata: 额外的元数据（可选）
    :return: bool 是否成功添加
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
    #获取现有向量库(如果持久化目录不存在，会自动创建空库)
    vs=get_vectorstore() #复用已有向量库或新建
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
