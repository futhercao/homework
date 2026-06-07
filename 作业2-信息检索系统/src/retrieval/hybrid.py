"""
混合检索引擎 — 多路召回 + RRF融合排序

流程:
    用户查询 → BM25稀疏召回 ─┐
              → 语义稠密召回  ─┤→ RRF融合 → 去重排序 → Top-K结果
              → 查询扩展      ─┘

RRF (Reciprocal Rank Fusion):
    score(d) = Σ 1/(k + rank_i(d))
    其中k=60，rank_i(d)是文档在第i路召回中的排名
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import RETRIEVAL_CONFIG
from src.nlp.pipeline import nlp
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vsm import VSMRetriever


class HybridRetriever:
    """
    混合检索器 — 融合多路召回结果

    支持:
    - BM25 + VSM 双引擎
    - 查询扩展
    - RRF/加权融合
    - 结果去重
    """

    def __init__(self, index):
        self.index = index
        self.bm25 = BM25Retriever(index)
        self.vsm = VSMRetriever(index)

    def search(self, query, top_k=20, algorithm='hybrid', use_expansion=False):
        """执行混合检索"""
        query_tokens = nlp.tokenize(query)
        if not query_tokens:
            return []

        expanded_query = query
        expanded_terms = []
        if use_expansion:
            expanded_terms = self._expand_query(query_tokens)
            if expanded_terms:
                expanded_query = query + ' ' + ' '.join(expanded_terms)

        # 根据算法选择召回策略
        if algorithm == 'bm25':
            results = self.bm25.search(expanded_query, top_k)
        elif algorithm == 'vsm':
            results = self.vsm.search(expanded_query, top_k)
        elif algorithm == 'hybrid':
            # 三路召回: BM25 + VSM + Semantic
            bm25_results = self.bm25.search(query, top_k * 2)
            vsm_results = self.vsm.search(query, top_k * 2)

            # 语义召回（如果可用）
            semantic_results = self._semantic_recall(query, top_k * 2)
            has_semantic = bool(semantic_results)
            if has_semantic:
                results = self._rrf_fusion(
                    [bm25_results, vsm_results, semantic_results], top_k,
                    label='Hybrid (BM25+VSM+Semantic)')
            else:
                results = self._rrf_fusion(
                    [bm25_results, vsm_results], top_k,
                    label='Hybrid (BM25+VSM)')
        else:
            results = self.bm25.search(expanded_query, top_k)

        # 填充扩展词信息
        if expanded_terms:
            for r in results:
                r['expanded_terms'] = expanded_terms

        return results

    def _rrf_fusion(self, result_lists, top_k, k=60, label='Hybrid'):
        """RRF融合多路排序结果"""
        doc_scores = {}
        doc_data = {}

        for results in result_lists:
            for rank, r in enumerate(results):
                doc_id = r['doc_id']
                rrf_score = 1.0 / (k + rank + 1)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
                if doc_id not in doc_data:
                    doc_data[doc_id] = r

        # 按RRF分数排序
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        final_results = []
        for doc_id, rrf_score in ranked:
            r = doc_data[doc_id].copy()
            r['score'] = round(rrf_score, 4)
            r['algorithm'] = label
            final_results.append(r)

        return final_results

    def _semantic_recall(self, query, top_k):
        """语义向量召回 — 使用sentence-transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            # 延迟加载模型
            if not hasattr(self, '_sem_model'):
                self._sem_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                # 编码所有文档（首次）
                self._sem_docs = {}
                self._sem_vecs = []
                for doc_id in self.index.doc_info:
                    info = self.index.doc_info[doc_id]
                    text = f"{info.get('title', '')} {info.get('content', '')[:300]}"
                    self._sem_docs[doc_id] = len(self._sem_vecs)
                    self._sem_vecs.append(text)
                self._sem_embeddings = self._sem_model.encode(
                    self._sem_vecs, show_progress_bar=False, batch_size=32,
                    normalize_embeddings=True
                )
                print(f'[语义] 已编码 {len(self._sem_vecs)} 篇文档')

            # 编码查询
            q_vec = self._sem_model.encode([query], normalize_embeddings=True)[0]
            # 余弦相似度
            scores = np.dot(self._sem_embeddings, q_vec)
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    doc_id = list(self._sem_docs.keys())[idx]
                    info = self.index.doc_info.get(doc_id, {})
                    results.append({
                        'doc_id': doc_id,
                        'score': float(scores[idx]),
                        'algorithm': 'Semantic',
                        'title': info.get('title', ''),
                        'snippet': info.get('content', '')[:200],
                        'url': info.get('url', ''),
                        'date': info.get('date', ''),
                        'source': info.get('source', ''),
                        'category': info.get('category', ''),
                        'images': info.get('images', [])[:3],
                        'image_count': info.get('image_count', 0),
                    })
            return results
        except Exception:
            return []

    def _expand_query(self, query_tokens, max_terms=None):
        """基于伪相关反馈(PRF)的查询扩展

        思想 (Rocchio/RM 风格的轻量实现):
            1. 取与查询词命中数最高的前若干篇文档作为"伪相关"集；
            2. 在该集合内按 tf*idf 累计候选词权重 (排除查询词、单字词、停用噪声)；
            3. 返回权重最高的 max_terms 个词作为扩展词。
        相比简单共现计数, tf*idf 加权可抑制高频泛词, 扩展词更聚焦主题。
        """
        if max_terms is None:
            max_terms = RETRIEVAL_CONFIG.get('query_expansion_terms', 3)

        query_set = set(query_tokens)

        # 1) 候选文档按命中查询词数排序, 取前若干篇为伪相关集
        doc_hits = Counter()
        for term in query_set:
            for doc_id in self.index.get_postings(term):
                doc_hits[doc_id] += 1
        if not doc_hits:
            return []
        pseudo_set = {d for d, _ in doc_hits.most_common(10)}

        # 2) 在伪相关集内累计候选词 tf*idf 权重
        term_weights = defaultdict(float)
        for term, postings in self.index.inverted_index.items():
            if term in query_set or len(term) < 2:
                continue
            idf = self.index.get_idf(term)
            if idf <= 0:
                continue
            for doc_id in pseudo_set:
                tf = postings.get(doc_id)
                if tf:
                    term_weights[term] += tf * idf

        # 3) 取权重最高的 max_terms 个扩展词
        ranked = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ranked[:max_terms]]


def create_retriever(index, algorithm='hybrid'):
    """工厂方法"""
    if algorithm == 'bm25':
        return BM25Retriever(index)
    elif algorithm == 'vsm':
        return VSMRetriever(index)
    elif algorithm == 'hybrid':
        return HybridRetriever(index)
    else:
        return BM25Retriever(index)
