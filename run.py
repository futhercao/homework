"""
信息检索与抽取平台 — 主入口

用法:
    python run.py crawl [--max 800]    运行爬虫
    python run.py build                 构建索引
    python run.py web                   启动Web界面
    python run.py demo                  一键演示 (crawl + build + web)
    python run.py extract [doc_id]      信息抽取
"""
import os
import sys
import asyncio
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def cmd_crawl(args):
    print("=" * 60)
    print("  爬虫模式")
    print("=" * 60)
    from src.crawler import crawl
    docs = crawl(max_docs=args.max or 800)
    print(f"\n爬取完成: {len(docs)} 篇")
    print("接下来运行: python run.py build")


def cmd_build(args):
    print("=" * 60)
    print("  索引构建模式")
    print("=" * 60)
    from src.retrieval.indexer import SearchIndex
    from src.crawler.engine import AsyncCrawlerEngine
    from src.storage.doc_store import DocumentStore

    docs = AsyncCrawlerEngine.load_documents()
    if not docs:
        print("错误: 未找到文档。请先运行: python run.py crawl")
        return

    store = DocumentStore()
    for doc in docs:
        try:
            store.save(doc)
        except Exception:
            pass

    print(f"已加载 {len(docs)} 篇文档")
    index = SearchIndex()
    index.build(docs)
    index.save()
    print("完成! 运行: python run.py web")


def cmd_web(args):
    print("=" * 60)
    print("  Web服务模式")
    print("=" * 60)
    from src.web.app import run
    run()


def cmd_demo(args):
    print("=" * 60)
    print("  一键演示")
    print("=" * 60)
    cmd_crawl(args)
    cmd_build(args)
    cmd_web(args)


def cmd_extract(args):
    doc_id = args.doc_id
    doc_path = os.path.join('data', 'documents', f'{doc_id}.json')
    if not os.path.exists(doc_path):
        print(f"文档不存在: {doc_path}")
        return
    import json
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc = json.load(f)

    from src.extraction.regex_engine import RegexExtractor
    from src.extraction.ner_engine import NEREnhancer
    from src.extraction.event_builder import EventBuilder

    extractor = RegexExtractor()
    enhancer = NEREnhancer()
    builder = EventBuilder()

    result = extractor.extract(doc)
    result = enhancer.enhance(doc, result)
    event = builder.build(doc, result)

    print("=" * 50)
    print(f"  文档: {doc.get('title', '')[:60]}")
    print("=" * 50)
    print(f"  灾害类型: {event['disaster']['type'] or '未识别'}")
    print(f"  发生时间: {event['disaster']['time'] or '未知'}")
    print(f"  发生地点: {event['disaster']['location'] or '未知'}")
    print(f"  伤亡情况: {event['disaster']['casualties'] or '未提及'}")
    print(f"  经济损失: {event['disaster']['economic_loss'] or '未提及'}")
    print(f"  响应级别: {event['disaster']['response_level'] or '未提及'}")
    print(f"  救援信息: {', '.join(event['disaster']['rescue_info']) or '未提及'}")
    print(f"  严重程度: {event['severity']}")
    print(f"  信息完整度: {event['filled_count']}/7")
    print(f"\n  摘要: {event['summary']}")


def main():
    parser = argparse.ArgumentParser(description='信息检索与抽取平台')
    sub = parser.add_subparsers(dest='command')

    sp1 = sub.add_parser('crawl', help='运行爬虫')
    sp1.add_argument('--max', type=int, default=800)

    sub.add_parser('build', help='构建索引')

    sub.add_parser('web', help='启动Web界面')

    sub.add_parser('demo', help='一键演示')

    sp2 = sub.add_parser('extract', help='信息抽取')
    sp2.add_argument('doc_id', help='文档ID')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    globals()[f'cmd_{args.command}'](args)


if __name__ == '__main__':
    main()
