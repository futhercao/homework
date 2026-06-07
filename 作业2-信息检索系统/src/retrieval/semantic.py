"""
语义检索模块 — 基于sentence-transformers的稠密向量检索
使用多语言模型，支持中文查询语义匹配
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import RETRIEVAL_CONFIG, VECTOR_DIR

from src.nlp.pipeline import nlp
from src.storage.vector_store import VectorStore

# 延迟加载
_encoder = None
_doc_vectors = None   # {doc_id: np.ndarray}
_vector_store = None


def _load_encoder():
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            # 使用轻量多语言模型
            _encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("[语义] sentence-transformers 模型已加载")
        except Exception as e:
            print(f"[语义] 模型加载失败: {e}")
            _encoder = False
    return _encoder if _encoder is not False else None


def encode_documents(documents):
    """离线编码所有文档，构建语义索引"""
    global _doc_vectors, _vector_store
    encoder = _load_encoder()
    if not encoder:
        return

    texts = [f"{d.get('title', '')} {d.get('content', '')[:500]}" for d in documents]
    doc_ids = [d['id'] for d in documents]

    print(f"[语义] 编码 {len(texts)} 篇文档...")
    vectors = encoder.encode(texts, show_progress_bar=True, batch_size=32,
                             normalize_embeddings=True)

    _doc_vectors = {doc_id: vec for doc_id, vec in zip(doc_ids, vectors)}

    # 存入FAISS向量索引
    _vector_store = VectorStore(name='semantic_docs', dim=vectors.shape[1])
    _vector_store.add(vectors, doc_ids)
    _vector_store.save()
    print(f"[语义] 编码完成，维度: {vectors.shape[1]}")


def load_semantic_index():
    """加载已有的语义索引"""
    global _vector_store, _doc_vectors
    _vector_store = VectorStore(name='semantic_docs')
    if _vector_store.load():
        print(f"[语义] 已加载: {_vector_store.count} 向量")
        return True
    return False


class SemanticRetriever:
    """语义检索器 — 稠密向量匹配"""

    def search(self, query, top_k=20):
        global _vector_store
        encoder = _load_encoder()
        if not encoder or not _vector_store or _vector_store.count == 0:
            return []

        query_vec = encoder.encode([query], normalize_embeddings=True)[0]
        hits = _vector_store.search(query_vec, top_k * 2)

        results = []
        for doc_id, score, _ in hits:
            info = {}

            # 如果索引已加载，从index获取文档信息
            try:
                from src.retrieval.indexer import SearchIndex
                # 这里的信息由调用者填充
            except ImportError:
                pass

            results.append({
                'doc_id': doc_id,
                'score': round(float(score), 4),
                'algorithm': 'Semantic',
            })

        return results[:top_k]


def encode_query(query):
    """编码查询向量"""
    encoder = _load_encoder()
    if not encoder:
        return None
    return encoder.encode([query], normalize_embeddings=True)[0]
