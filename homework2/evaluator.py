"""
检索结果评价模块
- 支持人工标注相关性判断
- 计算 Precision, Recall, F1, MAP, NDCG
- 保存和加载评价结果
"""
import os
import json
from datetime import datetime
from collections import defaultdict

from config import EVAL_DIR


class SearchEvaluator:
    """
    检索结果准确率人工评价系统

    工作流程:
    1. 用户执行查询，获取检索结果
    2. 对每个结果标注相关性（相关/不相关）
    3. 系统自动计算各项评价指标
    4. 保存评价记录用于后续分析

    评价指标:
    - Precision@K: 前K个结果中相关文档的比例
    - Recall@K: 前K个结果中相关文档占总相关文档的比例
    - F1@K: Precision和Recall的调和平均
    - AP (Average Precision): 平均精度
    - NDCG@K: 归一化折损累积增益
    """

    def __init__(self):
        self.evaluations = []
        self._load_history()

    def evaluate(self, query, results, judgments):
        """
        评价一次查询的检索结果

        参数:
            query: 查询字符串
            results: 检索结果列表 [{doc_id, score, title, ...}, ...]
            judgments: 相关性判断 {doc_id: relevance_score}
                       relevance_score: 0=不相关, 1=相关, 2=非常相关
        返回:
            评价指标字典
        """
        if not results or not judgments:
            return {}

        doc_ids = [r['doc_id'] for r in results]
        rel_scores = [judgments.get(did, 0) for did in doc_ids]

        total_relevant = sum(1 for v in judgments.values() if v > 0)

        metrics = {}
        for k in [5, 10, 20]:
            if k <= len(doc_ids):
                metrics[f'P@{k}'] = self._precision_at_k(rel_scores, k)
                metrics[f'R@{k}'] = self._recall_at_k(rel_scores, k, total_relevant)
                metrics[f'F1@{k}'] = self._f1_at_k(
                    metrics[f'P@{k}'], metrics[f'R@{k}']
                )

        metrics['AP'] = self._average_precision(rel_scores)
        metrics['NDCG@10'] = self._ndcg_at_k(rel_scores, 10)

        evaluation_record = {
            'query': query,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_results': len(results),
            'num_judged': len(judgments),
            'num_relevant': total_relevant,
            'metrics': {k: round(v, 4) for k, v in metrics.items()},
            'judgments': judgments,
        }
        self.evaluations.append(evaluation_record)
        self._save_history()

        return metrics

    @staticmethod
    def _precision_at_k(rel_scores, k):
        """Precision@K = 前K个结果中相关文档数 / K"""
        topk = rel_scores[:k]
        relevant = sum(1 for s in topk if s > 0)
        return relevant / k

    @staticmethod
    def _recall_at_k(rel_scores, k, total_relevant):
        """Recall@K = 前K个结果中相关文档数 / 总相关文档数"""
        if total_relevant == 0:
            return 0.0
        topk = rel_scores[:k]
        relevant = sum(1 for s in topk if s > 0)
        return relevant / total_relevant

    @staticmethod
    def _f1_at_k(precision, recall):
        """F1 = 2 * P * R / (P + R)"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _average_precision(rel_scores):
        """
        Average Precision (AP):
        AP = Σ(P@k * rel(k)) / 总相关文档数
        """
        relevant_count = 0
        precision_sum = 0.0
        total_relevant = sum(1 for s in rel_scores if s > 0)

        if total_relevant == 0:
            return 0.0

        for i, rel in enumerate(rel_scores):
            if rel > 0:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)

        return precision_sum / total_relevant

    @staticmethod
    def _ndcg_at_k(rel_scores, k):
        """
        NDCG@K (Normalized Discounted Cumulative Gain):
        DCG@K = Σ (2^rel_i - 1) / log2(i + 2)
        NDCG@K = DCG@K / IDCG@K
        """
        import math

        topk = rel_scores[:k]

        # DCG
        dcg = sum(
            (2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(topk)
        )

        # IDCG (理想排序)
        ideal = sorted(rel_scores, reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal)
        )

        return dcg / idcg if idcg > 0 else 0.0

    def get_summary(self):
        """获取所有评价的汇总统计"""
        if not self.evaluations:
            return {'message': '暂无评价记录'}

        all_metrics = defaultdict(list)
        for ev in self.evaluations:
            for k, v in ev['metrics'].items():
                all_metrics[k].append(v)

        summary = {
            'total_evaluations': len(self.evaluations),
            'average_metrics': {
                k: round(sum(v) / len(v), 4) for k, v in all_metrics.items()
            },
            'evaluations': self.evaluations,
        }
        return summary

    def _save_history(self):
        """保存评价历史"""
        os.makedirs(EVAL_DIR, exist_ok=True)
        path = os.path.join(EVAL_DIR, 'evaluation_history.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluations, f, ensure_ascii=False, indent=2)

    def _load_history(self):
        """加载评价历史"""
        path = os.path.join(EVAL_DIR, 'evaluation_history.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.evaluations = json.load(f)
