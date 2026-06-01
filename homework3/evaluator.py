"""
信息抽取评价模块
支持自动评价（与 ground truth 对比）和人工评价两种模式。
计算指标：Precision / Recall / F1（每个信息点及整体）
"""
import os
import json
import re
from datetime import datetime
from collections import defaultdict

from config import EVAL_DIR


class ExtractionEvaluator:
    """信息抽取结果评价器"""

    def __init__(self):
        self.evaluations = []
        self._load_history()

    # ── 自动评价（与 ground truth 对比）──────────────────────

    def auto_evaluate(self, doc_id, extraction, ground_truth):
        """
        自动评价单篇文档的抽取结果

        Args:
            doc_id: 文档 ID
            extraction: 抽取结果 dict
            ground_truth: 标准答案 dict

        Returns:
            dict: 逐信息点的 Precision / Recall / F1
        """
        points = [
            'disaster_type', 'event_time', 'event_location',
            'casualties', 'economic_loss', 'response_level', 'rescue_org',
        ]

        point_metrics = {}
        for point in points:
            pred = extraction.get(point, '')
            gold = ground_truth.get(point, '')
            metrics = self._compute_point_metrics(point, pred, gold)
            point_metrics[point] = metrics

        precisions = [m['precision'] for m in point_metrics.values()]
        recalls = [m['recall'] for m in point_metrics.values()]
        macro_p = sum(precisions) / len(precisions) if precisions else 0
        macro_r = sum(recalls) / len(recalls) if recalls else 0
        macro_f1 = (2 * macro_p * macro_r / (macro_p + macro_r)) if (macro_p + macro_r) > 0 else 0

        result = {
            'doc_id': doc_id,
            'point_metrics': point_metrics,
            'macro_precision': round(macro_p, 4),
            'macro_recall': round(macro_r, 4),
            'macro_f1': round(macro_f1, 4),
        }
        return result

    def auto_evaluate_batch(self, documents, extractions):
        """
        批量自动评价

        Args:
            documents: 文档列表（含 ground_truth）
            extractions: {doc_id: extraction_result}

        Returns:
            dict: 批量评价结果
        """
        all_results = []
        for doc in documents:
            doc_id = doc.get('id', '')
            gt = doc.get('ground_truth', {})
            if not gt:
                continue
            ext = extractions.get(doc_id, {})
            if isinstance(ext, dict) and 'event' in ext:
                ext = ext['event']
            elif isinstance(ext, dict) and 'fused' in ext:
                ext = ext['fused']
            result = self.auto_evaluate(doc_id, ext, gt)
            all_results.append(result)

        if not all_results:
            return {'message': '无可评价的文档'}

        point_aggregated = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})
        for r in all_results:
            for point, metrics in r['point_metrics'].items():
                point_aggregated[point]['precision'].append(metrics['precision'])
                point_aggregated[point]['recall'].append(metrics['recall'])
                point_aggregated[point]['f1'].append(metrics['f1'])

        avg_per_point = {}
        for point, lists in point_aggregated.items():
            avg_per_point[point] = {
                'avg_precision': round(sum(lists['precision']) / len(lists['precision']), 4),
                'avg_recall': round(sum(lists['recall']) / len(lists['recall']), 4),
                'avg_f1': round(sum(lists['f1']) / len(lists['f1']), 4),
                'count': len(lists['precision']),
            }

        overall = {
            'total_docs': len(all_results),
            'macro_precision': round(
                sum(r['macro_precision'] for r in all_results) / len(all_results), 4
            ),
            'macro_recall': round(
                sum(r['macro_recall'] for r in all_results) / len(all_results), 4
            ),
            'macro_f1': round(
                sum(r['macro_f1'] for r in all_results) / len(all_results), 4
            ),
        }

        batch_result = {
            'overall': overall,
            'per_point': avg_per_point,
            'details': all_results[:20],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        self.evaluations.append({
            'type': 'auto_batch',
            'timestamp': batch_result['timestamp'],
            'overall': overall,
            'per_point': avg_per_point,
        })
        self._save_history()

        return batch_result

    def _compute_point_metrics(self, point, pred, gold):
        """计算单个信息点的精确率、召回率、F1（增强版，含字段特定匹配）"""
        if point == 'rescue_org':
            return self._compute_set_metrics(pred, gold)

        pred_str = str(pred).strip() if pred else ''
        gold_str = str(gold).strip() if gold else ''

        if not gold_str:
            return {'precision': 1.0 if not pred_str else 0.0,
                    'recall': 1.0, 'f1': 1.0 if not pred_str else 0.0,
                    'pred': pred_str, 'gold': gold_str, 'match': 'no_gold'}

        if not pred_str:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                    'pred': pred_str, 'gold': gold_str, 'match': 'miss'}

        if pred_str == gold_str:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                    'pred': pred_str, 'gold': gold_str, 'match': 'exact'}

        # ── 字段特定匹配策略 ──

        if point == 'event_time':
            return self._compute_time_metrics(pred_str, gold_str)

        if point == 'event_location':
            return self._compute_location_metrics(pred_str, gold_str)

        if point == 'disaster_type':
            return self._compute_disaster_type_metrics(pred_str, gold_str)

        if pred_str in gold_str or gold_str in pred_str:
            overlap = min(len(pred_str), len(gold_str)) / max(len(pred_str), len(gold_str))
            return {'precision': overlap, 'recall': overlap,
                    'f1': overlap, 'pred': pred_str, 'gold': gold_str, 'match': 'partial'}

        if point in ('casualties', 'economic_loss'):
            pred_nums = set(re.findall(r'\d+(?:\.\d+)?', pred_str))
            gold_nums = set(re.findall(r'\d+(?:\.\d+)?', gold_str))
            if pred_nums and gold_nums:
                overlap = len(pred_nums & gold_nums) / max(len(pred_nums | gold_nums), 1)
                return {'precision': overlap, 'recall': overlap,
                        'f1': overlap, 'pred': pred_str, 'gold': gold_str, 'match': 'numeric'}

        pred_chars = set(pred_str)
        gold_chars = set(gold_str)
        overlap = len(pred_chars & gold_chars) / max(len(pred_chars | gold_chars), 1)
        return {'precision': overlap, 'recall': overlap,
                'f1': overlap, 'pred': pred_str, 'gold': gold_str, 'match': 'fuzzy'}

    def _compute_time_metrics(self, pred, gold):
        """时间字段专用匹配：基于日期组件比较"""
        # 先在gold和pred中查找所有日期片段
        time_pat_loose = re.compile(r'(\d{4})?[年/-]?(\d{1,2})[月/-](\d{1,2})日?')
        pm = time_pat_loose.search(pred)
        gm = time_pat_loose.search(gold)

        if pm and gm:
            py, pm_m, pd_d = pm.groups()
            gy, gm_m, gd_d = gm.groups()

            # 标准化月份（去掉前导0）
            pm_m = str(int(pm_m)) if pm_m else None
            gm_m = str(int(gm_m)) if gm_m else None
            pd_d = str(int(pd_d)) if pd_d else None
            gd_d = str(int(gd_d)) if gd_d else None

            # 如果月+日相同，给予高基础分
            month_day_match = (pm_m and gm_m and pd_d and gd_d
                              and pm_m == gm_m and pd_d == gd_d)

            if month_day_match:
                # 月日相同 → 基础分 0.8
                score = 0.8
                if py and gy and py == gy:
                    score = 1.0  # 年月日全匹配
                elif py and gy:
                    score = 0.85  # 年月匹配但年不同
                # 如果没有年份信息，维持 0.8
            else:
                score = 0.0
                parts_matched = 0

                if pm_m and gm_m and pm_m == gm_m:
                    score += 0.35
                    parts_matched += 1
                elif pm_m and gm_m:
                    score += 0.1

                if pd_d and gd_d and pd_d == gd_d:
                    score += 0.35
                    parts_matched += 1
                elif pd_d and gd_d:
                    score += 0.1

                if py and gy and py == gy:
                    score += 0.3
                    parts_matched += 1
                elif py and gy:
                    score += 0.05

                if parts_matched == 0:
                    if pred in gold or gold in pred:
                        score = min(len(pred), len(gold)) / max(len(pred), len(gold)) * 0.5
                    else:
                        score = len(set(pred) & set(gold)) / max(len(set(pred) | set(gold)), 1) * 0.5

            return {'precision': round(score, 4), 'recall': round(score, 4),
                    'f1': round(score, 4), 'pred': pred, 'gold': gold,
                    'match': 'time_component'}

        # 回退：包含关系
        if pred in gold or gold in pred:
            overlap = min(len(pred), len(gold)) / max(len(pred), len(gold))
            return {'precision': overlap, 'recall': overlap,
                    'f1': overlap, 'pred': pred, 'gold': gold, 'match': 'time_partial'}

        # 完全失败
        pred_chars = set(pred)
        gold_chars = set(gold)
        if pred_chars and gold_chars:
            overlap = len(pred_chars & gold_chars) / max(len(pred_chars | gold_chars), 1)
        else:
            overlap = 0
        return {'precision': round(overlap, 4), 'recall': round(overlap, 4),
                'f1': round(overlap, 4), 'pred': pred, 'gold': gold, 'match': 'time_fuzzy'}

    def _compute_location_metrics(self, pred, gold):
        """地点字段专用匹配：基于行政层级比较（宽容版）"""
        # Step 0: 直接相等
        if pred == gold:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                    'pred': pred, 'gold': gold, 'match': 'loc_exact'}

        # Step 0.5: 包含关系 - 增强版
        if pred in gold:
            # pred 比 gold 更少 → 部分匹配
            overlap = len(pred) / len(gold) if len(gold) > 0 else 0
            return {'precision': 1.0, 'recall': overlap,
                    'f1': 2*overlap/(1+overlap) if overlap > 0 else 0,
                    'pred': pred, 'gold': gold, 'match': 'loc_pred_in_gold'}
        if gold in pred:
            # pred 比 gold 更多 → 精确率惩罚但召回满分
            overlap = len(gold) / len(pred) if len(pred) > 0 else 0
            return {'precision': overlap, 'recall': 1.0,
                    'f1': 2*overlap/(1+overlap) if overlap > 0 else 0,
                    'pred': pred, 'gold': gold, 'match': 'loc_gold_in_pred'}

        # Step 1: 提取层级
        def parse_location(loc_str):
            """解析地名字符串为 (province, city, county)"""
            province = ''
            city = ''
            county = ''
            # 省级（包括自治区、特别行政区、直辖市）
            pm = re.search(r'([一-龥]{2,12}(?:省|自治区|壮族自治区|回族自治区|维吾尔自治区|特别行政区))', loc_str)
            if pm:
                province = pm.group(1)
            # 市级（市/自治州/地区/盟）
            cm = re.search(r'([一-龥]{2,15}(?:市|自治州|州|地区|盟))', loc_str)
            if cm:
                city = cm.group(1)
                if province and city == province:
                    city = ''
            # 县级
            xm = re.search(r'([一-龥]{2,15}(?:县|区|镇|乡|村))', loc_str)
            if xm:
                county = xm.group(1)
            return province, city, county

        p_prov, p_city, p_county = parse_location(pred)
        g_prov, g_city, g_county = parse_location(gold)

        # Step 2: 如果都无法解析或一个是外国地名，回退到字符重叠
        has_pred = bool(p_prov or p_city or p_county)
        has_gold = bool(g_prov or g_city or g_county)

        if not has_pred and not has_gold:
            # 两个都没有层级 → 尝试字符级匹配
            pred_chars = set(pred)
            gold_chars = set(gold)
            if pred_chars and gold_chars:
                overlap = len(pred_chars & gold_chars) / max(len(pred_chars | gold_chars), 1)
            else:
                overlap = 0
            return {'precision': round(overlap, 4), 'recall': round(overlap, 4),
                    'f1': round(overlap, 4), 'pred': pred, 'gold': gold, 'match': 'loc_char'}

        if not has_pred:
            # pred 无层级但 gold 有 → pred 可能只提取了具体地名
            # 检查 pred 是否出现在 gold 中，或字符重叠
            if pred and gold:
                pred_chars = set(pred)
                gold_chars = set(gold)
                overlap = len(pred_chars & gold_chars) / max(len(pred_chars | gold_chars), 1) if pred_chars and gold_chars else 0
            else:
                overlap = 0
            return {'precision': round(overlap, 4), 'recall': round(overlap, 4),
                    'f1': round(overlap, 4), 'pred': pred, 'gold': gold, 'match': 'loc_pred_no_hierarchy'}
        if not has_gold:
            # gold 无层级但 pred 有
            if pred and gold:
                pred_chars = set(pred)
                gold_chars = set(gold)
                overlap = len(pred_chars & gold_chars) / max(len(pred_chars | gold_chars), 1) if pred_chars and gold_chars else 0
            else:
                overlap = 0
            return {'precision': round(overlap, 4), 'recall': round(overlap, 4),
                    'f1': round(overlap, 4), 'pred': pred, 'gold': gold, 'match': 'loc_gold_no_hierarchy'}

        # Step 3: 行政层级匹配
        score = 0.0
        total_weight = 0.0

        # 省匹配（权重 0.45）
        if g_prov:
            total_weight += 0.45
            if p_prov and (p_prov == g_prov or p_prov in g_prov or g_prov in p_prov or p_prov[:3] == g_prov[:3]):
                score += 0.45
            elif p_prov:
                # 省份部分匹配
                common = len(set(p_prov) & set(g_prov))
                if common >= 2:
                    score += 0.45 * (common / max(len(p_prov), len(g_prov)))

        # 市匹配（权重 0.35）
        if g_city:
            total_weight += 0.35
            if p_city and (p_city == g_city or p_city in g_city or g_city in p_city):
                score += 0.35
            elif p_city:
                common = len(set(p_city) & set(g_city))
                if common >= 2:
                    score += 0.35 * (common / max(len(p_city), len(g_city)))

        # 县级匹配（权重 0.20）
        if g_county:
            total_weight += 0.20
            if p_county and (p_county == g_county or p_county in g_county or g_county in p_county):
                score += 0.20
            elif p_county:
                common = len(set(p_county) & set(g_county))
                if common >= 2:
                    score += 0.20 * (common / max(len(p_county), len(g_county)))

        # Bonus: pred 层级比 gold 更细（如 pred 含 county 但 gold 不含）
        if p_county and not g_county and p_prov == g_prov:
            score = min(1.0, score + 0.1)

        if total_weight == 0:
            overlap = len(set(pred) & set(gold)) / max(len(set(pred) | set(gold)), 1)
            return {'precision': round(overlap, 4), 'recall': round(overlap, 4),
                    'f1': round(overlap, 4), 'pred': pred, 'gold': gold, 'match': 'loc_fallback'}

        final_score = score / total_weight if total_weight > 0 else 0
        return {'precision': round(final_score, 4), 'recall': round(final_score, 4),
                'f1': round(final_score, 4), 'pred': pred, 'gold': gold,
                'match': 'location_hierarchy'}

    def _compute_disaster_type_metrics(self, pred, gold):
        """灾害类型专用匹配：考虑类型间语义相关性"""
        # 类型相关性矩阵（相关类型给予部分分数）
        related_groups = [
            {'泥石流', '山体滑坡', '崩塌', '塌方'},           # 地质类
            {'洪水', '洪涝', '暴雨', '山洪'},                 # 水文类
            {'火灾', '森林火灾', '草原火灾', '山林火灾'},      # 火灾类
            {'台风', '飓风', '龙卷风'},                       # 风灾类
            {'暴雪', '雪灾', '低温雨雪冰冻'},                  # 雪灾类
            {'干旱', '旱灾'},                                 # 干旱类
            {'冰雹', '风雹'},                                 # 雹灾类
        ]

        if pred == gold:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                    'pred': pred, 'gold': gold, 'match': 'exact'}

        # 包含关系
        if pred in gold or gold in pred:
            overlap = min(len(pred), len(gold)) / max(len(pred), len(gold))
            return {'precision': overlap, 'recall': overlap,
                    'f1': overlap, 'pred': pred, 'gold': gold, 'match': 'partial'}

        # 检查是否属于同一相关组
        for group in related_groups:
            if pred in group and gold in group:
                # 同组不同类型 → 部分匹配
                return {'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
                        'pred': pred, 'gold': gold, 'match': 'related_type'}

        # 无关类型
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'pred': pred, 'gold': gold, 'match': 'different_type'}

    def _compute_set_metrics(self, pred, gold):
        """对列表类型（如救援组织）计算集合级别指标"""
        pred_set = set(pred) if isinstance(pred, list) else set()
        gold_set = set(gold) if isinstance(gold, list) else set()

        if not gold_set:
            p = 1.0 if not pred_set else 0.0
            return {'precision': p, 'recall': 1.0, 'f1': p,
                    'pred': list(pred_set), 'gold': list(gold_set), 'match': 'no_gold'}
        if not pred_set:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                    'pred': [], 'gold': list(gold_set), 'match': 'miss'}

        pred_hits = 0
        for p_item in pred_set:
            for g_item in gold_set:
                if p_item == g_item or p_item in g_item or g_item in p_item:
                    pred_hits += 1
                    break

        gold_matched = set()
        for g_item in gold_set:
            for p_item in pred_set:
                if p_item == g_item or p_item in g_item or g_item in p_item:
                    gold_matched.add(g_item)
                    break

        precision = pred_hits / len(pred_set) if pred_set else 0
        recall = len(gold_matched) / len(gold_set) if gold_set else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        return {'precision': round(precision, 4), 'recall': round(recall, 4),
                'f1': round(f1, 4), 'pred': list(pred_set), 'gold': list(gold_set),
                'match': 'set'}

    # ── 人工评价 ─────────────────────────────────────────────

    def human_evaluate(self, doc_id, extraction, judgments):
        """
        人工评价：用户对每个信息点标注 正确/部分正确/错误

        Args:
            doc_id: 文档 ID
            extraction: 抽取结果
            judgments: {info_point: score}  score ∈ {0, 0.5, 1}

        Returns:
            dict: 人工评价指标
        """
        scores = list(judgments.values())
        accuracy = sum(scores) / len(scores) if scores else 0

        result = {
            'doc_id': doc_id,
            'type': 'human',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'judgments': judgments,
            'accuracy': round(accuracy, 4),
            'total_points': len(scores),
            'correct': sum(1 for s in scores if s == 1),
            'partial': sum(1 for s in scores if s == 0.5),
            'wrong': sum(1 for s in scores if s == 0),
        }

        self.evaluations.append(result)
        self._save_history()
        return result

    def get_summary(self):
        """获取评价历史摘要"""
        if not self.evaluations:
            return {'message': '暂无评价记录', 'evaluations': []}

        auto_evals = [e for e in self.evaluations if e.get('type') == 'auto_batch']
        human_evals = [e for e in self.evaluations if e.get('type') == 'human']

        summary = {
            'total_evaluations': len(self.evaluations),
            'auto_evaluations': len(auto_evals),
            'human_evaluations': len(human_evals),
            'evaluations': self.evaluations[-20:],
        }

        if human_evals:
            avg_acc = sum(e['accuracy'] for e in human_evals) / len(human_evals)
            summary['avg_human_accuracy'] = round(avg_acc, 4)

        if auto_evals:
            latest = auto_evals[-1]
            summary['latest_auto'] = latest.get('overall', {})

        return summary

    def _save_history(self):
        os.makedirs(EVAL_DIR, exist_ok=True)
        path = os.path.join(EVAL_DIR, 'extraction_eval_history.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluations, f, ensure_ascii=False, indent=2)

    def _load_history(self):
        path = os.path.join(EVAL_DIR, 'extraction_eval_history.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.evaluations = json.load(f)
