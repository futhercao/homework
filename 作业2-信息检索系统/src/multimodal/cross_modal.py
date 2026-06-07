"""
跨模态检索引擎
- 文搜图: 输入文字描述 → 返回最匹配的图片
- 图搜文: 上传图片 → 返回相关新闻文档
- 图文联合: 文本匹配分数 + 图像相似度加权融合
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import VECTOR_DIR, IMAGE_DIR
from src.storage.vector_store import VectorStore
from src.multimodal.visual_encoder import VisualEncoder


class CrossModalSearch:
    """跨模态检索引擎"""

    def __init__(self):
        self.encoder = VisualEncoder()
        self.image_index = VectorStore(name='image_features', dim=512)
        self.text_index = VectorStore(name='clip_text_features', dim=512)

    def index_images(self, image_paths, doc_ids=None):
        """为图片构建向量索引

        doc_ids 与 image_paths 一一对应 (可为 None)。内部按 path→doc_id 建映射,
        因此即使部分图片编码失败被跳过, doc_id 也不会错位。
        """
        if not self.encoder.ready:
            print('[跨模态] CLIP不可用，跳过图片索引')
            return

        path2doc = dict(zip(image_paths, doc_ids)) if doc_ids else {}

        print(f'[跨模态] 编码 {len(image_paths)} 张图片...')
        features = self.encoder.encode_batch(image_paths)

        if not features:
            return

        vectors = []
        ids = []
        metadatas = []
        for path, vec in features.items():
            vectors.append(vec)
            ids.append(path)
            metadatas.append({
                'image_path': path,
                'doc_id': path2doc.get(path, ''),
            })

        self.image_index.add(vectors, ids, metadatas)
        self.image_index.save()
        print(f'[跨模态] 图片索引完成: {len(vectors)} 张')

    def search_images_by_text(self, query, top_k=10):
        """文搜图: 文字描述 → 图片"""
        if not self.encoder.ready:
            return []
        if self.image_index.count == 0 and not self.image_index.load():
            return []

        text_vec = self.encoder.encode_text(query)
        if text_vec is None:
            return []

        hits = self.image_index.search(text_vec, top_k)
        return [{'image_path': img_id, 'score': score, 'metadata': meta}
                for img_id, score, meta in hits]

    def search_docs_by_image(self, image_path, doc_index, top_k=10):
        """图搜文: 上传图片 → 相关新闻文档"""
        if not self.encoder.ready:
            return []

        img_vec = self.encoder.encode_image(image_path)
        if img_vec is None:
            return []

        # 在文本向量索引中搜索
        if self.text_index.count == 0 and not self.text_index.load():
            # 如果没有文本索引，使用图片索引查找相似图片的关联文档
            if self.image_index.count == 0 and not self.image_index.load():
                return []
            hits = self.image_index.search(img_vec, top_k)
            doc_ids = set()
            for _, _, meta in hits:
                if meta.get('doc_id'):
                    doc_ids.add(meta['doc_id'])
            return [{'doc_id': did, 'score': 1.0, 'reason': 'image_similarity'}
                    for did in list(doc_ids)[:top_k]]

        # 如果有文本索引
        hits = self.text_index.search(img_vec, top_k)
        return [{'doc_id': meta.get('doc_id', ''), 'score': score,
                 'text': meta.get('text', '')[:100]}
                for _, score, meta in hits]

    def index_document_texts(self, doc_texts):
        """为文档的CLIP文本向量构建索引（用于图搜文）"""
        if not self.encoder.ready:
            return

        print(f'[跨模态] 编码 {len(doc_texts)} 条文档文本...')
        vectors = []
        ids = []
        for i, (doc_id, text) in enumerate(doc_texts.items()):
            vec = self.encoder.encode_text(text)
            if vec is not None:
                vectors.append(vec)
                ids.append(doc_id)

        if vectors:
            self.text_index.add(vectors, ids, [{'doc_id': did, 'text': doc_texts.get(did, '')} for did in ids])
            self.text_index.save()
            print(f'[跨模态] 文本索引完成: {len(vectors)} 条')

    def load_indices(self):
        """加载已有索引"""
        img_loaded = self.image_index.load()
        txt_loaded = self.text_index.load()
        return {'image_index': img_loaded, 'text_index': txt_loaded}
