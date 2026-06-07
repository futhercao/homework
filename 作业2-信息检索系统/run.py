"""
信息检索系统 (作业2) — 主入口

用法:
    python run.py crawl [--max 800]   异步爬取真实新闻语料 (多来源)
    python run.py build               构建倒排索引 (BM25 / VSM / 语义)
    python run.py multimodal          构建 CLIP 图像向量索引 (以文搜图/以图搜文)
    python run.py web                 启动 Web 界面 (检索 / 评价 / 多模态 / 看板)
    python run.py demo                一键: crawl + build + web

全流程纯 CPU 可跑, 无需 GPU。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def cmd_crawl(args):
    print("=" * 60)
    print("  爬虫模式 — 异步抓取真实新闻语料")
    print("=" * 60)
    from src.crawler import crawl
    docs = crawl(max_docs=args.max or 800)
    print(f"\n爬取完成: {len(docs)} 篇")
    print("接下来运行: python run.py build")


def cmd_build(args):
    print("=" * 60)
    print("  索引构建模式 — 倒排索引 (BM25 / VSM / 语义)")
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
    print("完成! 运行: python run.py multimodal (可选) 或 python run.py web")


def cmd_multimodal(args):
    print("=" * 60)
    print("  多模态索引构建 — CLIP 图像编码")
    print("=" * 60)
    import runpy
    runpy.run_path(os.path.join(os.path.dirname(__file__), 'scripts', 'build_multimodal.py'),
                   run_name='__main__')


def cmd_web(args):
    print("=" * 60)
    print("  Web 服务模式")
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


def main():
    parser = argparse.ArgumentParser(description='信息检索系统 (作业2)')
    sub = parser.add_subparsers(dest='command')

    sp1 = sub.add_parser('crawl', help='运行爬虫')
    sp1.add_argument('--max', type=int, default=800)

    sub.add_parser('build', help='构建倒排索引')
    sub.add_parser('multimodal', help='构建 CLIP 图像索引')
    sub.add_parser('web', help='启动 Web 界面')
    sub.add_parser('demo', help='一键演示')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    globals()[f'cmd_{args.command}'](args)


if __name__ == '__main__':
    main()
