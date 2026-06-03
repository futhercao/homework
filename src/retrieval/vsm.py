"""
向量空间模型 (Vector Space Model)
TF-IDF加权 + 余弦相似度
"""
import math
from collections import Counter

from src.nlp.pipeline import nlp
from src.config import RETRIEVAL_CONFIG


class VSMRetriever:
    """向量空间模型检索器"""

    def __init__(self, index):
        self.index = index

    def search(self, query, top_k=20):
        query_tokens = nlp.tokenize(query)
        if not query_tokens:
            return []

        # 构建查询TF-IDF向量
        query_tf = Counter(query_tokens)
        query_vector = {}
        for term, count in query_tf.items():
            tf = count / len(query_tokens)
            idf = self.index.get_idf(term)
            if idf > 0:
                query_vector[term] = tf * idf

        if not query_vector:
            return []

        query_norm = math.sqrt(sum(v ** 2 for v in query_vector.values()))

        # 找候选文档
        candidate_docs = set()
        for term in query_vector:
            candidate_docs.update(self.index.get_postings(term).keys())

        # 计算余弦相似度
        scores = {}
        for doc_id in candidate_docs:
            # 使用倒排索引中的TF值 * IDF
            dot = 0.0
            doc_norm_sq = 0.0
            for term, q_weight in query_vector.items():
                postings = self.index.get_postings(term)
                tf_count = postings.get(doc_id, 0)
                if tf_count > 0:
                    doc_len = self.index.get_doc_length(doc_id)
                    tf = tf_count / doc_len
                    dot += q_weight * tf * self.index.get_idf(term)
                    doc_norm_sq += (tf * self.index.get_idf(term)) ** 2

            doc_norm = math.sqrt(doc_norm_sq)
            if doc_norm > 0 and query_norm > 0:
                scores[doc_id] = dot / (query_norm * doc_norm)

        return self._format(scores, query_tokens, top_k)

    def _format(self, scores, query_tokens, top_k):
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        snippet_len = RETRIEVAL_CONFIG.get('snippet_length', 250)

        for doc_id, score in ranked:
            if score <= 0:
                continue
            info = self.index.get_doc_info(doc_id)
            content = info.get('content', '')
            snippet = nlp.get_snippet(content, query_tokens, snippet_len)
            snippet_hl = nlp.highlight(snippet, query_tokens)

            results.append({
                'doc_id': doc_id,
                'score': round(float(score), 4),
                'algorithm': 'VSM',
                'title': info.get('title', ''),
                'snippet': snippet_hl,
                'url': info.get('url', ''),
                'date': info.get('date', ''),
                'source': info.get('source', ''),
                'category': info.get('category', ''),
                'images': info.get('images', [])[:3],
                'image_count': info.get('image_count', 0),
            })
        return results
