"""
多模态自然灾害事件信息抽取系统 - 命令行入口

用法:
    python main.py generate     生成150篇灾害新闻样本数据（含信息图图片）
    python main.py extract      对所有文档执行多模态信息抽取
    python main.py evaluate     运行自动评价（与 ground truth 对比）
    python main.py kg           构建并保存事件知识图谱
    python main.py web          启动 Web 界面
    python main.py demo         一键演示：生成 → 抽取 → 评价 → 知识图谱 → Web
    python main.py crawl [url]  从指定URL爬取真实新闻
"""
import sys
import json
import os
import time


def cmd_generate():
    """生成样本数据"""
    from crawler import DisasterDataGenerator
    print("=" * 60)
    print("  Step 1: 生成灾害新闻样本数据")
    print("=" * 60)
    docs = DisasterDataGenerator.generate(150)
    image_count = sum(1 for d in docs if d.get('images'))
    print(f"  共生成 {len(docs)} 篇文档, {image_count} 篇含信息图图片")


def cmd_extract():
    """执行批量信息抽取"""
    from crawler import load_documents
    from text_extractor import create_extractor
    from image_extractor import ImageExtractor
    from fusion import MultiModalFusion

    print("=" * 60)
    print("  Step 2: 多模态信息抽取")
    print("=" * 60)

    documents = load_documents()
    if not documents:
        print("  [错误] 无数据，请先运行: python main.py generate")
        return {}

    text_ext = create_extractor('ensemble')
    img_ext = ImageExtractor()
    fusion = MultiModalFusion()
    extractions = {}

    t0 = time.time()
    for i, doc in enumerate(documents):
        text_result = text_ext.extract(doc.get('content', ''), doc.get('title', ''))

        image_results = []
        for img_info in doc.get('images', []):
            img_result = img_ext.extract_from_image(img_info)
            image_results.append(img_result)

        fused = fusion.fuse(text_result, image_results)

        extractions[doc['id']] = {
            'doc_id': doc['id'],
            'text_result': text_result,
            'image_results': image_results,
            'fused_result': fused,
            'event': fused.get('event', {}),
            'confidence': fused.get('confidence', {}),
            'modalities': fused.get('modalities_used', []),
        }

        if (i + 1) % 20 == 0 or (i + 1) == len(documents):
            print(f"  [{i+1}/{len(documents)}] 已抽取")

    elapsed = time.time() - t0
    multi_count = sum(1 for e in extractions.values() if len(e['modalities']) > 1)
    print(f"  完成! 耗时 {elapsed:.1f}s, 多模态融合文档 {multi_count} 篇")

    from config import BASE_DIR
    out_path = os.path.join(BASE_DIR, 'data', 'extractions.json')
    save_data = {}
    for doc_id, ext in extractions.items():
        save_data[doc_id] = {
            'event': ext['event'],
            'confidence': ext['confidence'],
            'modalities': ext['modalities'],
        }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"  抽取结果已保存至 {out_path}")

    return extractions


def cmd_evaluate(extractions=None):
    """运行自动评价"""
    from crawler import load_documents
    from evaluator import ExtractionEvaluator

    print("=" * 60)
    print("  Step 3: 自动评价")
    print("=" * 60)

    documents = load_documents()
    if not documents:
        print("  [错误] 无数据")
        return

    if not extractions:
        from config import BASE_DIR
        ext_path = os.path.join(BASE_DIR, 'data', 'extractions.json')
        if os.path.exists(ext_path):
            with open(ext_path, 'r', encoding='utf-8') as f:
                extractions = json.load(f)
        else:
            print("  [错误] 无抽取结果，请先运行: python main.py extract")
            return

    evaluator = ExtractionEvaluator()
    result = evaluator.auto_evaluate_batch(documents, extractions)

    overall = result.get('overall', {})
    print(f"\n  ┌────────────────────────────────────────┐")
    print(f"  │  评价文档数: {overall.get('total_docs', 0):>5}                      │")
    print(f"  │  Macro Precision: {overall.get('macro_precision', 0)*100:>6.1f}%               │")
    print(f"  │  Macro Recall:    {overall.get('macro_recall', 0)*100:>6.1f}%               │")
    print(f"  │  Macro F1:        {overall.get('macro_f1', 0)*100:>6.1f}%               │")
    print(f"  └────────────────────────────────────────┘")

    per_point = result.get('per_point', {})
    if per_point:
        print(f"\n  {'信息点':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
        names = {
            'disaster_type': '灾害类型', 'event_time': '发生时间',
            'event_location': '发生地点', 'casualties': '伤亡情况',
            'economic_loss': '经济损失', 'response_level': '响应级别',
            'rescue_org': '救援组织',
        }
        for point, m in per_point.items():
            name = names.get(point, point)
            print(f"  {name:<10} {m['avg_precision']*100:>9.1f}% {m['avg_recall']*100:>9.1f}% {m['avg_f1']*100:>9.1f}%")


def cmd_knowledge_graph(extractions=None):
    """构建知识图谱"""
    from knowledge_graph import EventKnowledgeGraph

    print("=" * 60)
    print("  Step 4: 构建事件知识图谱")
    print("=" * 60)

    if not extractions:
        from config import BASE_DIR
        ext_path = os.path.join(BASE_DIR, 'data', 'extractions.json')
        if os.path.exists(ext_path):
            with open(ext_path, 'r', encoding='utf-8') as f:
                extractions = json.load(f)
        else:
            print("  [错误] 无抽取结果")
            return

    kg = EventKnowledgeGraph()
    for doc_id, ext in extractions.items():
        kg.add_event(doc_id, ext)

    kg.save()
    stats = kg.get_statistics()
    print(f"  节点: {stats['total_nodes']}, 边: {stats['total_edges']}, 事件: {stats['total_events']}")
    print(f"  灾害类型分布: {stats.get('disaster_type_distribution', {})}")


def cmd_web():
    """启动 Web 界面"""
    from app import app
    from config import FLASK_CONFIG
    print("=" * 60)
    print("  多模态灾害事件信息抽取系统 - Web 界面")
    print(f"  访问: http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    print("=" * 60)
    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug'],
    )


def cmd_demo():
    """一键演示全流程"""
    print("\n" + "=" * 60)
    print("  多模态自然灾害事件信息抽取系统 - 完整演示")
    print("=" * 60 + "\n")

    cmd_generate()
    print()
    extractions = cmd_extract()
    print()
    cmd_evaluate(extractions)
    print()
    cmd_knowledge_graph(extractions)
    print()
    cmd_web()


def cmd_crawl():
    """爬取真实新闻"""
    from crawler import WebCrawler
    urls = sys.argv[2:] if len(sys.argv) > 2 else None
    crawler = WebCrawler()
    crawler.crawl(seed_urls=urls)


COMMANDS = {
    'generate': cmd_generate,
    'extract': cmd_extract,
    'evaluate': cmd_evaluate,
    'kg': cmd_knowledge_graph,
    'web': cmd_web,
    'demo': cmd_demo,
    'crawl': cmd_crawl,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("可用命令:", ', '.join(COMMANDS.keys()))
        return
    COMMANDS[sys.argv[1]]()


if __name__ == '__main__':
    main()
