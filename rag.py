from sentence_transformers import SentenceTransformer,util
from rag_data import DOCS
model=SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings=model.encode(DOCS,convert_to_tensor=True)
def retrieve(query,top_k=1):
    q_emb=model.encode(query,convert_to_tensor=True)
    scores=util.cos_sim(q_emb,doc_embeddings)[0]
    top_idx=scores.topk(top_k).indices
    return [DOCS[i] for i in top_idx]