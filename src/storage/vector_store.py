"""
向量存储层 — FAISS
- 图像特征向量索引
- 文本特征向量索引
- 快速近似最近邻搜索 (ANN)
"""
import os
import pickle
import numpy as np

from src.config import VECTOR_DIR

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorStore:
    """FAISS向量索引，支持图像+文本向量存储和检索"""

    def __init__(self, name='default', dim=512):
        self.name = name
        self.dim = dim
        self.index = None
        self.id_map = []          # 向量ID → 文档/图片ID
        self.metadata = {}        # ID → 附加元数据
        os.makedirs(VECTOR_DIR, exist_ok=True)

    def _ensure_index(self):
        if self.index is None and FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dim)  # Inner Product (cosine for normalized vectors)

    def add(self, vectors, ids, metadata_list=None):
        """批量添加向量: vectors (N, dim), ids (N,), metadata_list (N,)"""
        self._ensure_index()
        if not FAISS_AVAILABLE:
            return

        vectors = np.asarray(vectors, dtype=np.float32)
        # L2归一化 → Inner Product = Cosine Similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        for i, _id in enumerate(ids):
            self.id_map.append(_id)
            if metadata_list:
                self.metadata[_id] = metadata_list[i]

    def search(self, query_vector, top_k=10):
        """检索Top-K最相似向量，返回 [(id, score, metadata), ...]"""
        self._ensure_index()
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return []

        query = np.asarray([query_vector], dtype=np.float32)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.id_map):
                _id = self.id_map[idx]
                results.append((_id, float(score), self.metadata.get(_id, {})))
        return results

    @property
    def count(self):
        return self.index.ntotal if self.index else 0

    def save(self):
        """持久化向量索引到磁盘"""
        if not FAISS_AVAILABLE or self.index is None:
            return
        idx_path = os.path.join(VECTOR_DIR, f'{self.name}.faiss')
        map_path = os.path.join(VECTOR_DIR, f'{self.name}.pkl')
        faiss.write_index(self.index, idx_path)
        with open(map_path, 'wb') as f:
            pickle.dump({'id_map': self.id_map, 'metadata': self.metadata, 'dim': self.dim}, f)

    def load(self):
        """从磁盘加载向量索引"""
        if not FAISS_AVAILABLE:
            return False
        idx_path = os.path.join(VECTOR_DIR, f'{self.name}.faiss')
        map_path = os.path.join(VECTOR_DIR, f'{self.name}.pkl')
        if not os.path.exists(idx_path) or not os.path.exists(map_path):
            return False
        self.index = faiss.read_index(idx_path)
        with open(map_path, 'rb') as f:
            data = pickle.load(f)
        self.id_map = data['id_map']
        self.metadata = data['metadata']
        self.dim = data['dim']
        return True
