"""
倒排索引构建模块
- 构建倒排索引（term → posting list）
- 计算TF-IDF权重
- 支持索引的保存与加载
"""
import os
import json
import math
from collections import defaultdict, Counter

from config import INDEX_DIR
from preprocessor import TextPreprocessor
from crawler import load_documents


class InvertedIndex:
    """
    倒排索引，支持TF-IDF加权。

    索引结构:
        inverted_index[term] = {doc_id: tf, ...}
        doc_lengths[doc_id] = 文档中的总词数
        doc_info[doc_id] = {title, url, date, ...}
        idf[term] = log(N / df)
        tfidf[term][doc_id] = tf * idf
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.inverted_index = defaultdict(dict)     # term -> {doc_id: tf_count}
        self.doc_freq = defaultdict(int)            # term -> 出现该词的文档数
        self.doc_lengths = {}                       # doc_id -> 文档总词数
        self.doc_info = {}                          # doc_id -> 文档元信息
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self.idf = {}                              # term -> IDF值
        self.tfidf_vectors = defaultdict(dict)      # doc_id -> {term: tfidf_weight}
        self.vocabulary = set()

    def build(self, documents=None):
        """从文档集合构建倒排索引"""
        if documents is None:
            documents = load_documents()

        if not documents:
            print("[索引] 没有找到文档，请先爬取或生成数据")
            return

        self.total_docs = len(documents)
        total_length = 0

        print(f"[索引] 开始构建索引，共 {self.total_docs} 篇文档")

        for doc in documents:
            doc_id = doc['id']
            text = doc.get('title', '') + ' ' + doc.get('content', '')

            # 对图片alt文本也进行索引（多媒体索引）
            for img in doc.get('images', []):
                alt = img.get('alt', '')
                if alt:
                    text += ' ' + alt

            tokens = self.preprocessor.tokenize_for_index(text)
            term_counts = Counter(tokens)
            doc_length = len(tokens)

            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length
            self.doc_info[doc_id] = {
                'title': doc.get('title', ''),
                'url': doc.get('url', ''),
                'date': doc.get('date', ''),
                'content': doc.get('content', '')[:500],
                'images': doc.get('images', []),
            }

            for term, count in term_counts.items():
                self.inverted_index[term][doc_id] = count
                self.vocabulary.add(term)

        # 计算文档频率
        for term, postings in self.inverted_index.items():
            self.doc_freq[term] = len(postings)

        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0

        # 计算IDF
        self._compute_idf()

        # 计算TF-IDF向量
        self._compute_tfidf()

        print(f"[索引] 索引构建完成: {self.total_docs} 篇文档, {len(self.vocabulary)} 个词项")

        self.save()

    def _compute_idf(self):
        """计算逆文档频率 IDF(t) = log(N / df(t))"""
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log(self.total_docs / df) if df > 0 else 0

    def _compute_tfidf(self):
        """计算TF-IDF权重：TF(t,d) = count(t,d) / |d|, TF-IDF = TF * IDF"""
        for term, postings in self.inverted_index.items():
            idf_val = self.idf.get(term, 0)
            for doc_id, tf_count in postings.items():
                doc_len = self.doc_lengths.get(doc_id, 1)
                tf = tf_count / doc_len
                self.tfidf_vectors[doc_id][term] = tf * idf_val

    def get_postings(self, term):
        """获取词项的倒排表"""
        return self.inverted_index.get(term, {})

    def get_idf(self, term):
        """获取词项的IDF值"""
        return self.idf.get(term, 0)

    def get_doc_tfidf(self, doc_id):
        """获取文档的TF-IDF向量"""
        return self.tfidf_vectors.get(doc_id, {})

    def save(self):
        """将索引保存到磁盘"""
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

        path = os.path.join(INDEX_DIR, 'index.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        print(f"[索引] 索引已保存到 {path}")

    def load(self):
        """从磁盘加载索引"""
        path = os.path.join(INDEX_DIR, 'index.json')
        if not os.path.exists(path):
            print(f"[索引] 索引文件不存在: {path}")
            return False

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.total_docs = data['total_docs']
        self.avg_doc_length = data['avg_doc_length']
        self.inverted_index = defaultdict(dict, {
            t: dict(p) for t, p in data['inverted_index'].items()
        })
        self.doc_freq = defaultdict(int, data['doc_freq'])
        self.doc_lengths = data['doc_lengths']
        self.doc_info = data['doc_info']
        self.idf = data['idf']
        self.vocabulary = set(self.inverted_index.keys())

        # 重新计算TF-IDF向量
        self._compute_tfidf()

        print(f"[索引] 已加载索引: {self.total_docs} 篇文档, {len(self.vocabulary)} 个词项")
        return True

    def get_stats(self):
        """返回索引统计信息"""
        return {
            'total_docs': self.total_docs,
            'vocabulary_size': len(self.vocabulary),
            'avg_doc_length': round(self.avg_doc_length, 2),
            'index_size_terms': len(self.inverted_index),
        }


if __name__ == '__main__':
    index = InvertedIndex()
    index.build()
    stats = index.get_stats()
    print(f"\n索引统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
