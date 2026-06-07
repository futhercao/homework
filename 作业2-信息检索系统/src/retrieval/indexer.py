"""
倒排索引构建模块
- 全量索引 + 增量索引
- TF-IDF 权重计算
- 文档元信息存储
- 图片文本索引（多模态基础）
"""
import os
import sys
import json
import math
import pickle
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import INDEX_DIR
from src.nlp.pipeline import nlp
from src.content_filter import filter_content_images


class SearchIndex:
    """
    倒排索引 - 支持TF-IDF和BM25检索

    索引结构:
        inverted_index: {term: {doc_id: tf_count}}
        doc_freq: {term: document_frequency}
        doc_lengths: {doc_id: total_tokens}
        doc_info: {doc_id: {title, url, date, ...}}
        idf: {term: idf_value}
    """

    def __init__(self):
        self.inverted_index = defaultdict(dict)
        self.doc_freq = defaultdict(int)
        self.doc_lengths = {}
        self.doc_info = {}
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self.idf = {}
        self.vocabulary = set()

    def build(self, documents, use_images=True):
        """从文档列表构建倒排索引"""
        self.total_docs = len(documents)
        if self.total_docs == 0:
            return

        total_length = 0
        print(f"[索引] 开始构建，共 {self.total_docs} 篇文档...")

        for doc in documents:
            doc_id = doc['id']
            # 组合标题+正文+图片alt文本
            text_parts = [doc.get('title', '')]
            text_parts.append(doc.get('content', ''))

            if use_images:
                for img in doc.get('images', []):
                    alt = img.get('alt', '')
                    if alt and len(alt) >= 2:
                        text_parts.append(alt)
                # 本地图片路径也纳入索引
                for img in doc.get('local_images', []):
                    alt = img.get('alt', '')
                    if alt and len(alt) >= 2:
                        text_parts.append(alt)

            full_text = ' '.join(text_parts)
            tokens = nlp.tokenize(full_text)
            term_counts = Counter(tokens)
            doc_len = len(tokens)

            self.doc_lengths[doc_id] = doc_len
            total_length += doc_len
            # 仅统计/展示「内容图」: 复用 VLM 质检结论过滤掉 logo/广告/二维码/截图等无关图
            content_imgs = filter_content_images(doc.get('local_images') or doc.get('images') or [])
            self.doc_info[doc_id] = {
                'title': doc.get('title', ''),
                'url': doc.get('url', ''),
                'date': doc.get('date', ''),
                'content': doc.get('content', '')[:800],
                'source': doc.get('source', ''),
                'category': doc.get('category', ''),
                'images': content_imgs,
                'image_count': len(content_imgs),
            }

            for term, count in term_counts.items():
                self.inverted_index[term][doc_id] = count
                self.vocabulary.add(term)

        # 计算文档频率
        for term, postings in self.inverted_index.items():
            self.doc_freq[term] = len(postings)

        self.avg_doc_length = total_length / self.total_docs if self.total_docs else 1

        # 计算IDF
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

        print(f"[索引] 完成: {self.total_docs} 篇, {len(self.vocabulary)} 个词项, "
              f"平均长度 {self.avg_doc_length:.1f}")

    def get_postings(self, term):
        return self.inverted_index.get(term, {})

    def get_idf(self, term):
        return self.idf.get(term, 0)

    def get_doc_info(self, doc_id):
        return self.doc_info.get(doc_id, {})

    def get_doc_length(self, doc_id):
        return self.doc_lengths.get(doc_id, 1)

    def get_stats(self):
        return {
            'total_docs': self.total_docs,
            'vocabulary_size': len(self.vocabulary),
            'avg_doc_length': round(self.avg_doc_length, 2),
        }

    def save(self):
        """保存索引到磁盘"""
        os.makedirs(INDEX_DIR, exist_ok=True)
        data = {
            'total_docs': self.total_docs,
            'avg_doc_length': self.avg_doc_length,
            'vocabulary_size': len(self.vocabulary),
            'inverted_index': {t: dict(p) for t, p in self.inverted_index.items()},
            'doc_freq': dict(self.doc_freq),
            'doc_lengths': self.doc_lengths,
            'doc_info': self.doc_info,
            'idf': self.idf,
        }
        with open(os.path.join(INDEX_DIR, 'search_index.pkl'), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[索引] 已保存到磁盘")

    def load(self):
        """从磁盘加载索引"""
        path = os.path.join(INDEX_DIR, 'search_index.pkl')
        if not os.path.exists(path):
            return False

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.total_docs = data['total_docs']
        self.avg_doc_length = data['avg_doc_length']
        self.inverted_index = defaultdict(dict, {t: dict(p) for t, p in data['inverted_index'].items()})
        self.doc_freq = defaultdict(int, data['doc_freq'])
        self.doc_lengths = data['doc_lengths']
        self.doc_info = data['doc_info']
        self.idf = data['idf']
        self.vocabulary = set(self.inverted_index.keys())

        print(f"[索引] 已加载: {self.total_docs} 篇, {len(self.vocabulary)} 词项")
        return True
