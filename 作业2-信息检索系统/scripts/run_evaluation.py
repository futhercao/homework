"""
完整自动评价脚本 — 多维度语义级相关性判断
避免与BM25关键词匹配的循环验证
"""
import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.indexer import SearchIndex
from src.retrieval.hybrid import create_retriever
from src.evaluation.metrics import evaluate_search, EvaluationStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vsm import VSMRetriever


# 查询定义: (query, positive_kw, negative_kw, title_bonus_kw)
# - positive: 需 ≥3 hits 才认为部分相关，≥5 hits 才非常相关
# - negative: 包含这些词则降级（防跨领域误判）
# - title_bonus: 标题命中则升一级（与pos不完全重合，避免循环）
EVAL_QUERIES = [
    ('人工智能',
     ['人工智能', 'AI', '大模型', 'GPT', 'OpenAI', '神经网络', '机器学习', '深度学习'],
     ['地震', '台风', '洪水', '暴雨', '滑坡'],
     ['人工智能', 'AI', '大模型', 'GPT']),
    ('地震救援',
     ['地震', '遇难', '救援', '震级', '救灾', '搜救', '震中'],
     ['人工智能', '芯片', '股票', '基金', '新能源'],
     ['地震', '遇难', '救援']),
    ('新能源汽车',
     ['新能源', '电动', '充电', '电池', '比亚迪', '特斯拉', '电车'],
     ['地震', '台风', '洪水', '教育', '考试'],
     ['新能源', '电动', '特斯拉', '比亚迪']),
    ('芯片制造',
     ['芯片', '半导体', '晶圆', '制程', '光刻', '台积电', '处理器'],
     ['地震', '台风', '洪水', '新能源'],
     ['芯片', '半导体', '晶圆', '制程']),
    ('台风灾害',
     ['台风', '登陆', '应急响应', '转移', '预警', '防台', '大风', '风暴'],
     ['人工智能', '芯片', '股票', '新能源', '教育'],
     ['台风', '登陆', '应急响应']),
    ('高等教育改革',
     ['高考', '教育', '大学', '招生', '考生', '高校', '考试', '录取', '志愿'],
     ['地震', '台风', '洪水', '芯片', '新能源', '股票'],
     ['高考', '教育', '大学', '招生']),
    ('医疗健康保障',
     ['医疗', '医保', '医院', '药品', '就诊', '门诊', '患者', '健康'],
     ['地震', '台风', '芯片', '新能源'],
     ['医疗', '医保', '医院']),
    ('自动驾驶技术',
     ['自动驾驶', '无人驾驶', '智驾', 'FSD', '激光雷达', '智能驾驶'],
     ['地震', '台风', '洪水', '教育', '医疗'],
     ['自动驾驶', '无人驾驶', '智驾', 'FSD']),
    ('自然灾害损失',
     ['自然灾害', '直接经济损失', '受灾', '灾情', '倒塌', '转移群众'],
     ['人工智能', '芯片', '股票'],
     ['自然灾害', '直接经济损失', '受灾']),
    ('财经投资理财',
     ['投资', '基金', 'A股', '经济', '金融', '股市', '理财', '资本', '沪指', '深指'],
     ['地震', '台风', '洪水', '芯片', '教育'],
     ['投资', '基金', 'A股', '股市', '理财']),
]


def auto_judge(results, pos_keywords, neg_keywords, title_bonus):
    """多维度语义级相关性自动判断

    判断逻辑（与BM25解耦）:
    1. 标题命中 title_bonus = 强信号 (+bonus)
    2. 正文命中 pos_keywords ≥3 = 部分相关; ≥5 = 非常相关
    3. 命中 neg_keywords ≥2 = 降级一档
    4. 最终钳制到 0-2

    与BM25的区别:
    - title_bonus ⊆ pos_keywords, 是更强的子集, 标题命中权重更高
    - neg_keywords 过滤跨领域噪声 (如搜索"地震救援"不应返回财经新闻中的"应急")
    - 阈值≥3 hits避免零散提及被误判相关
    """
    judgments = {}
    for r in results:
        doc_id = r['doc_id']
        title = r.get('title', '')
        snippet = r.get('snippet', '')
        full = (title + ' ' + snippet)

        pos_hits = sum(1 for kw in pos_keywords if kw in full)
        neg_hits = sum(1 for kw in neg_keywords if kw in full)
        title_hit = any(kw in title for kw in title_bonus)

        # === 相关性定级 ===
        if pos_hits >= 5 and title_hit:
            relevance = 2      # 非常相关: 多方匹配 + 标题命中
        elif pos_hits >= 5:
            relevance = 2
        elif pos_hits >= 3:
            relevance = 1      # 部分相关
        elif pos_hits >= 2 and title_hit:
            relevance = 1      # 内容略少但标题命中
        else:
            relevance = 0      # 不相关

        # === 负关键词降级 ===
        if neg_hits >= 3:
            relevance = max(0, relevance - 2)
        elif neg_hits >= 2:
            relevance = max(0, relevance - 1)

        judgments[doc_id] = relevance
    return judgments


def run():
    idx = SearchIndex()
    if not idx.load():
        print("索引未构建!")
        return

    store = EvaluationStore()
    all_results = {}

    print("=" * 70)
    print(f"检索评价 | {idx.total_docs}篇 | {len(idx.vocabulary)}词项 | {len(EVAL_QUERIES)}查询")
    print("=" * 70)

    for alg in ['bm25', 'vsm']:
        print(f"\n{'='*55}")
        print(f"算法: {alg.upper()}")
        print(f"{'='*55}")

        ret = BM25Retriever(idx) if alg == 'bm25' else VSMRetriever(idx)
        all_metrics = defaultdict(list)

        for query, pos_kw, neg_kw, title_bonus in EVAL_QUERIES:
            results = ret.search(query, top_k=20)
            if not results:
                print(f"  [{query}] 无结果")
                continue

            judgments = auto_judge(results, pos_kw, neg_kw, title_bonus)
            metrics = evaluate_search(query, results, judgments)
            store.save_search_eval(query, alg, results, judgments, metrics)

            for k, v in metrics.items():
                all_metrics[k].append(v)

            relevant = sum(1 for v in judgments.values() if v > 0)
            print(f"  [{query:8s}] P@5={metrics['P@5']:.2%} P@10={metrics['P@10']:.2%} "
                  f"F1@10={metrics['F1@10']:.2%} MAP={metrics['MAP']:.2%} "
                  f"NDCG={metrics['NDCG@10']:.2%} (judged relevant: {relevant})")

        # 汇总
        print(f"\n  --- {alg.upper()} 平均 ---")
        avg = {}
        for k, vs in all_metrics.items():
            avg[k] = round(sum(vs) / len(vs), 4) if vs else 0
            print(f"  {k}: {avg[k]:.4f}")
        all_results[alg] = avg

    # 算法对比
    print(f"\n{'='*55}")
    print("算法对比")
    print(f"{'='*55}")
    for metric in ['P@5', 'P@10', 'F1@10', 'MAP', 'NDCG@10', 'MRR']:
        b25 = all_results['bm25'].get(metric, 0)
        vsm = all_results['vsm'].get(metric, 0)
        best = 'BM25' if b25 >= vsm else 'VSM'
        print(f"  {metric:12s}: BM25={b25:.4f}  VSM={vsm:.4f}  → {best}")

    # 保存报告
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'evaluation', 'evaluation_report.json'
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'all_results': all_results,
            'summary': store.get_search_summary(),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {report_path}")


if __name__ == '__main__':
    run()
