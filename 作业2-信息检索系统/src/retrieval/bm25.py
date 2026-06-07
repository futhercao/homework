"""
BM25 概率检索模型 (Okapi BM25)
业界标准的检索算法，比简单TF-IDF更优

Score(q,d) = Σ IDF(t) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × |d|/avgdl))
"""
import math
from collections import Counter

from src.nlp.pipeline import nlp
from src.config import RETRIEVAL_CONFIG


class BM25Retriever:
    """Okapi BM25检索器"""

    def __init__(self, index):
        self.index = index
        self.k1 = RETRIEVAL_CONFIG.get('bm25_k1', 1.5)
        self.b = RETRIEVAL_CONFIG.get('bm25_b', 0.75)

    def search(self, query, top_k=20):
        """执行BM25检索"""
        query_tokens = nlp.tokenize(query)
        if not query_tokens:
            return []

        scores = {}
        avgdl = self.index.avg_doc_length
        unique_terms = set(query_tokens)

        for term in unique_terms:
            postings = self.index.get_postings(term)
            if not postings:
                continue

            idf = self.index.get_idf(term)

            for doc_id, tf_count in postings.items():
                doc_len = self.index.get_doc_length(doc_id)
                numerator = tf_count * (self.k1 + 1)
                denominator = tf_count + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                tf_norm = numerator / denominator
                scores[doc_id] = scores.get(doc_id, 0) + idf * tf_norm

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
                'algorithm': 'BM25',
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
