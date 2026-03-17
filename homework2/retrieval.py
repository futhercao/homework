"""
信息检索模块
- VSMRetriever: 向量空间模型（TF-IDF + 余弦相似度）
- BM25Retriever: BM25概率检索模型（优化算法）
- QueryExpander: 查询扩展（创新点）
"""
import math
from collections import Counter

from preprocessor import TextPreprocessor
from config import RETRIEVAL_CONFIG


class VSMRetriever:
    """
    向量空间模型检索器

    核心算法：
    1. 将查询和文档都表示为TF-IDF加权的向量
    2. 使用余弦相似度计算查询与文档的匹配度
    3. 按相似度降序排列返回结果

    Cosine Similarity:
        sim(q, d) = Σ(w_q * w_d) / (|q| * |d|)
    """

    def __init__(self, index):
        self.index = index
        self.preprocessor = TextPreprocessor()

    def search(self, query, top_k=None):
        """执行VSM检索"""
        top_k = top_k or RETRIEVAL_CONFIG.get('top_k', 20)
        query_tokens = self.preprocessor.tokenize_query(query)

        if not query_tokens:
            return []

        # 构建查询向量（TF-IDF加权）
        query_tf = Counter(query_tokens)
        query_vector = {}
        for term, count in query_tf.items():
            tf = count / len(query_tokens)
            idf = self.index.get_idf(term)
            if idf > 0:
                query_vector[term] = tf * idf

        if not query_vector:
            return []

        # 计算查询向量的模
        query_norm = math.sqrt(sum(v ** 2 for v in query_vector.values()))

        # 对每个候选文档计算余弦相似度
        scores = {}
        candidate_docs = set()
        for term in query_vector:
            postings = self.index.get_postings(term)
            candidate_docs.update(postings.keys())

        for doc_id in candidate_docs:
            doc_vector = self.index.get_doc_tfidf(doc_id)
            dot_product = sum(
                query_vector.get(term, 0) * doc_vector.get(term, 0)
                for term in query_vector
            )
            doc_norm = math.sqrt(sum(v ** 2 for v in doc_vector.values()))
            if doc_norm > 0 and query_norm > 0:
                scores[doc_id] = dot_product / (query_norm * doc_norm)

        return self._format_results(scores, query_tokens, top_k)

    def _format_results(self, scores, query_tokens, top_k):
        """格式化检索结果"""
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        snippet_len = RETRIEVAL_CONFIG.get('snippet_length', 200)

        for doc_id, score in ranked:
            if score <= 0:
                continue
            info = self.index.doc_info.get(doc_id, {})
            content = info.get('content', '')
            snippet = self.preprocessor.get_snippet(content, query_tokens, snippet_len)
            snippet_highlighted = self.preprocessor.highlight(snippet, query_tokens)

            results.append({
                'doc_id': doc_id,
                'score': round(score, 6),
                'title': info.get('title', ''),
                'snippet': snippet_highlighted,
                'snippet_plain': snippet,
                'url': info.get('url', ''),
                'date': info.get('date', ''),
                'images': info.get('images', []),
            })

        return results


class BM25Retriever:
    """
    BM25概率检索模型（Okapi BM25）

    这是对基本VSM的优化算法（创新点之一）。BM25在信息检索领域被广泛
    认为优于基本的TF-IDF向量空间模型。

    核心公式：
        score(q, d) = Σ IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d| / avgdl))

    参数:
        k1: 词频饱和度参数 (默认1.5)
        b: 文档长度归一化参数 (默认0.75)
    """

    def __init__(self, index):
        self.index = index
        self.preprocessor = TextPreprocessor()
        self.k1 = RETRIEVAL_CONFIG.get('bm25_k1', 1.5)
        self.b = RETRIEVAL_CONFIG.get('bm25_b', 0.75)

    def search(self, query, top_k=None):
        """执行BM25检索"""
        top_k = top_k or RETRIEVAL_CONFIG.get('top_k', 20)
        query_tokens = self.preprocessor.tokenize_query(query)

        if not query_tokens:
            return []

        scores = {}
        avgdl = self.index.avg_doc_length

        for term in set(query_tokens):
            postings = self.index.get_postings(term)
            if not postings:
                continue

            # BM25 IDF: log((N - df + 0.5) / (df + 0.5))
            df = self.index.doc_freq.get(term, 0)
            N = self.index.total_docs
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf_count in postings.items():
                doc_len = self.index.doc_lengths.get(doc_id, 1)
                # BM25 TF归一化
                numerator = tf_count * (self.k1 + 1)
                denominator = tf_count + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                tf_norm = numerator / denominator

                score = idf * tf_norm
                scores[doc_id] = scores.get(doc_id, 0) + score

        return self._format_results(scores, query_tokens, top_k)

    def _format_results(self, scores, query_tokens, top_k):
        """格式化检索结果"""
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        snippet_len = RETRIEVAL_CONFIG.get('snippet_length', 200)

        for doc_id, score in ranked:
            if score <= 0:
                continue
            info = self.index.doc_info.get(doc_id, {})
            content = info.get('content', '')
            snippet = self.preprocessor.get_snippet(content, query_tokens, snippet_len)
            snippet_highlighted = self.preprocessor.highlight(snippet, query_tokens)

            results.append({
                'doc_id': doc_id,
                'score': round(score, 6),
                'title': info.get('title', ''),
                'snippet': snippet_highlighted,
                'snippet_plain': snippet,
                'url': info.get('url', ''),
                'date': info.get('date', ''),
                'images': info.get('images', []),
            })

        return results


class QueryExpander:
    """
    查询扩展模块（创新点之二）

    基于共现词分析实现查询扩展：
    1. 对原始查询词，在倒排索引中找到包含这些词的文档
    2. 统计这些文档中高频共现的其他词
    3. 选取最相关的共现词加入查询
    """

    def __init__(self, index):
        self.index = index
        self.preprocessor = TextPreprocessor()

    def expand(self, query, max_expansion=3):
        """对查询进行扩展"""
        query_tokens = self.preprocessor.tokenize_query(query)
        if not query_tokens:
            return query_tokens

        # 找到包含查询词的文档
        candidate_docs = set()
        for term in query_tokens:
            postings = self.index.get_postings(term)
            candidate_docs.update(postings.keys())

        if not candidate_docs:
            return query_tokens

        # 统计候选文档中的高频词
        cooccurrence = Counter()
        query_set = set(query_tokens)

        for doc_id in list(candidate_docs)[:50]:
            doc_vector = self.index.get_doc_tfidf(doc_id)
            for term, weight in doc_vector.items():
                if term not in query_set and weight > 0:
                    cooccurrence[term] += weight

        # 选取得分最高的词作为扩展词
        expansion_terms = [
            term for term, _ in cooccurrence.most_common(max_expansion)
        ]

        expanded = query_tokens + expansion_terms
        return expanded


def create_retriever(index, algorithm=None):
    """工厂方法：根据算法名称创建检索器"""
    algorithm = algorithm or RETRIEVAL_CONFIG.get('default_algorithm', 'bm25')
    if algorithm == 'vsm':
        return VSMRetriever(index)
    elif algorithm == 'bm25':
        return BM25Retriever(index)
    else:
        raise ValueError(f"不支持的检索算法: {algorithm}")
