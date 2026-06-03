"""
评价指标体系
- 检索评价: P@K, R@K, F1, MAP, NDCG, MRR
- 抽取评价: per-field P/R/F1, exact match, partial match
"""
import math
import json
import os
from datetime import datetime
from collections import defaultdict

from src.config import EVAL_DIR


# ====== 检索评价 ======

def precision_at_k(relevance_scores, k):
    """Precision@K = 前K个结果中相关文档数 / K"""
    topk = relevance_scores[:k]
    relevant = sum(1 for s in topk if s > 0)
    return relevant / k if k > 0 else 0


def recall_at_k(relevance_scores, k, total_relevant):
    """Recall@K = 前K个结果中相关文档数 / 总相关文档数"""
    if total_relevant == 0:
        return 0
    topk = relevance_scores[:k]
    relevant = sum(1 for s in topk if s > 0)
    return relevant / total_relevant


def f1_at_k(precision, recall):
    """F1 = 2 * P * R / (P + R)"""
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def average_precision(relevance_scores):
    """Average Precision"""
    relevant_count = 0
    precision_sum = 0.0
    total_relevant = sum(1 for s in relevance_scores if s > 0)
    if total_relevant == 0:
        return 0
    for i, rel in enumerate(relevance_scores):
        if rel > 0:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    return precision_sum / total_relevant


def ndcg_at_k(relevance_scores, k):
    """NDCG@K"""
    topk = relevance_scores[:k]
    dcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(topk))
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0


def mrr(relevance_scores):
    """Mean Reciprocal Rank: 第一个相关文档排名的倒数"""
    for i, rel in enumerate(relevance_scores):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0


def evaluate_search(query, results, judgments):
    """
    完整检索评价
    judgments: {doc_id: relevance_score}  0=不相关, 1=部分相关, 2=非常相关
    """
    doc_ids = [r['doc_id'] for r in results]
    rel_scores = [judgments.get(did, 0) for did in doc_ids]
    total_relevant = sum(1 for v in judgments.values() if v > 0)

    metrics = {}
    for k in [5, 10, 20]:
        p = precision_at_k(rel_scores, k)
        r = recall_at_k(rel_scores, k, total_relevant)
        metrics[f'P@{k}'] = round(p, 4)
        metrics[f'R@{k}'] = round(r, 4)
        metrics[f'F1@{k}'] = round(f1_at_k(p, r), 4)

    metrics['MAP'] = round(average_precision(rel_scores), 4)
    metrics['NDCG@10'] = round(ndcg_at_k(rel_scores, 10), 4)
    metrics['MRR'] = round(mrr(rel_scores), 4)

    return metrics


# ====== 抽取评价 ======

def evaluate_extraction(predictions, ground_truth):
    """
    信息抽取评价
    predictions: {field: value}
    ground_truth: {field: value}
    """
    fields = ['disaster_type', 'event_time', 'event_location', 'casualties',
              'economic_loss', 'response_level', 'rescue_info']

    field_metrics = {}
    for field in fields:
        pred = predictions.get(field, '')
        gt = ground_truth.get(field, '')

        pred_str = str(pred).strip() if not isinstance(pred, list) else ' '.join(pred)
        gt_str = str(gt).strip() if not isinstance(gt, list) else ' '.join(gt)

        # 精确匹配
        exact = 1 if pred_str == gt_str else 0
        # 部分匹配（对长文本）
        partial = 0
        if not exact and pred_str and gt_str:
            if gt_str in pred_str or pred_str in gt_str:
                partial = 1
            elif len(set(pred_str) & set(gt_str)) / max(len(set(gt_str)), 1) > 0.5:
                partial = 1

        field_metrics[field] = {
            'exact_match': exact,
            'partial_match': partial,
            'prediction': pred_str[:100],
            'ground_truth': gt_str[:100],
        }

    # 整体统计
    total = len(fields)
    exact_count = sum(1 for f in field_metrics.values() if f['exact_match'])
    at_least_partial = sum(1 for f in field_metrics.values()
                           if f['exact_match'] or f['partial_match'])

    return {
        'per_field': field_metrics,
        'exact_accuracy': round(exact_count / total, 4),
        'partial_accuracy': round(at_least_partial / total, 4),
        'filled_count': sum(1 for f in field_metrics.values()
                           if f['prediction'] or f['ground_truth']),
    }


# ====== 评价记录管理 ======

class EvaluationStore:
    """评价结果持久化"""

    def __init__(self):
        self.search_records = []
        self.extraction_records = []
        os.makedirs(EVAL_DIR, exist_ok=True)
        self._load()

    def save_search_eval(self, query, algorithm, results, judgments, metrics):
        record = {
            'type': 'search',
            'query': query,
            'algorithm': algorithm,
            'num_results': len(results),
            'num_judgments': len(judgments),
            'num_relevant': sum(1 for v in judgments.values() if v > 0),
            'metrics': metrics,
            'judgments': judgments,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        self.search_records.append(record)
        self._save()
        return record

    def save_extraction_eval(self, doc_id, predictions, ground_truth, metrics):
        record = {
            'type': 'extraction',
            'doc_id': doc_id,
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        self.extraction_records.append(record)
        self._save()
        return record

    def get_search_summary(self):
        if not self.search_records:
            return {'message': '暂无检索评价记录'}
        all_metrics = defaultdict(list)
        for r in self.search_records:
            for k, v in r['metrics'].items():
                all_metrics[k].append(v)
        return {
            'total_evaluations': len(self.search_records),
            'average_metrics': {k: round(sum(v) / len(v), 4)
                               for k, v in all_metrics.items()},
            'records': self.search_records[-20:],  # 最近20条
        }

    def get_extraction_summary(self):
        if not self.extraction_records:
            return {'message': '暂无抽取评价记录'}
        accuracies = [r['metrics']['exact_accuracy'] for r in self.extraction_records
                      if 'exact_accuracy' in r.get('metrics', {})]
        return {
            'total_evaluations': len(self.extraction_records),
            'average_exact_accuracy': round(sum(accuracies) / len(accuracies), 4) if accuracies else 0,
            'records': self.extraction_records[-20:],
        }

    def _save(self):
        path = os.path.join(EVAL_DIR, 'evaluations.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'search': self.search_records,
                'extraction': self.extraction_records,
            }, f, ensure_ascii=False, indent=2)

    def _load(self):
        path = os.path.join(EVAL_DIR, 'evaluations.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.search_records = data.get('search', [])
            self.extraction_records = data.get('extraction', [])
