from sentence_transformers import SentenceTransformer,util
from rag_data import DOCS
# 加载预训练的句向量模型
# all-MiniLM-L6-v2：轻量、速度快、语义效果好，常用于 RAG / 相似度搜索
model=SentenceTransformer("all-MiniLM-L6-v2")
# 对所有文档进行向量化
# convert_to_tensor=True 表示返回 PyTorch Tensor，方便后续计算相似度
doc_embeddings=model.encode(DOCS,convert_to_tensor=True)
def retrieve(query,top_k=1):
    """    根据用户查询，从 DOCS 中检索最相关的 top_k 条文档    
    :param query: 用户输入的查询文本（字符串）    
    :param top_k: 返回最相似的文档数量    
    :return: 相似度最高的文档列表    """
    # 将查询语句转换为向量
    q_emb=model.encode(query,convert_to_tensor=True)
    # 计算 查询向量 与 所有文档向量 的余弦相似度
    # 结果是一个 1 x N 的张量（N 为文档数量）
    scores=util.cos_sim(q_emb,doc_embeddings)[0]
    # 从相似度得分中取 top_k 个最大值对应的索引
    top_idx=scores.topk(top_k).indices
     # 根据索引返回原始文档内容
    return [DOCS[i] for i in top_idx]

# 文档 → 向量（离线）
# 用户问题 → 向量
# ↓
# 余弦相似度
# ↓
# 取最相似文档
