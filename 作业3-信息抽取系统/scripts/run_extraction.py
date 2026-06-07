"""
批量信息抽取 → 灾害事件 → 知识图谱

- 对每篇灾害文档运行 EnsembleExtractor (regex + 触发词 + NER 加权投票, 见 ensemble.py);
  若文档已有 OCR 文本(ocr_text, 由 run_ocr.py 生成), 追加到正文后再抽取,
  实现"图片中的灾情文字也参与抽取"(多媒体信息抽取);
- 用 EventBuilder 组装为结构化事件(7信息点 + 严重度 + 完整度 + 摘要);
- 全部事件 → KnowledgeGraphBuilder → data/knowledge_graph.json (ECharts力导图数据);
- 抽取结果落盘 data/extractions.json, 供 Web 抽取/评价页直接加载, 避免重复计算。

用法: python scripts/run_extraction.py
"""
import os
import sys
import glob
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR
from src.extraction.ensemble import EnsembleExtractor
from src.extraction.event_builder import EventBuilder
from src.extraction.kg_builder import KnowledgeGraphBuilder

DISASTER_DIR = os.path.join(DATA_DIR, 'disaster_docs')
SEVEN = ['disaster_type', 'event_time', 'event_location', 'casualties',
         'economic_loss', 'response_level', 'rescue_info']


def main():
    ens = EnsembleExtractor()
    eb = EventBuilder()
    kg = KnowledgeGraphBuilder()

    files = sorted(glob.glob(os.path.join(DISASTER_DIR, 'dis_*.json')))
    events = []
    extractions = {}
    used_ocr = 0
    sev_dist = Counter()
    filled_total = 0

    for i, fp in enumerate(files, 1):
        try:
            doc = json.load(open(fp, 'r', encoding='utf-8'))
        except Exception:
            continue

        # OCR 文本回灌: 追加到正文供 regex/NER/trigger 抽取
        aug = dict(doc)
        if doc.get('ocr_text'):
            aug['content'] = doc.get('content', '') + '\n【图片文字】' + doc['ocr_text']
            used_ocr += 1

        result = ens.extract(aug)
        fields = {k: result[k] for k in SEVEN}
        event = eb.build(doc, fields)
        events.append(event)

        sev_dist[event['severity']] += 1
        filled_total += event['filled_count']
        extractions[doc['id']] = {
            'fields': fields,
            'event': event,
            'ensemble': result.get('_ensemble', {}),
            'used_ocr': bool(doc.get('ocr_text')),
        }

        if i % 40 == 0:
            print(f'  [{i}/{len(files)}] 已抽取')

    # 知识图谱
    kg.build(events)
    kg.save()

    # 抽取结果落盘
    with open(os.path.join(DATA_DIR, 'extractions.json'), 'w', encoding='utf-8') as f:
        json.dump(extractions, f, ensure_ascii=False, indent=2)

    valid = sum(1 for e in events if e['is_valid_event'])
    n = len(events) or 1
    print(f'\n[抽取] 完成: {len(events)} 篇文档')
    print(f'  有效事件(≥3信息点): {valid} ({valid/n*100:.1f}%)')
    print(f'  平均信息完整度: {filled_total/n:.2f}/7')
    print(f'  使用OCR文本的文档: {used_ocr}')
    print(f'  严重度分布: {dict(sev_dist)}')
    print(f'  结果: data/extractions.json, data/knowledge_graph.json')


if __name__ == '__main__':
    main()
