"""
构建倒排索引脚本
从 data/documents/ 加载所有文档，构建倒排索引
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.indexer import SearchIndex
from src.crawler.engine import AsyncCrawlerEngine
from src.storage.doc_store import DocumentStore


def main():
    print("=" * 60)
    print("  构建倒排索引")
    print("=" * 60)

    # 加载文档
    docs = AsyncCrawlerEngine.load_documents()
    if not docs:
        print("[错误] 未找到任何文档，请先运行爬虫: python scripts/crawl.py")
        return

    # 同步到SQLite
    print(f"[加载] {len(docs)} 篇文档")
    store = DocumentStore()
    for doc in docs:
        try:
            store.save(doc)
        except Exception:
            pass

    # 显示数据概况
    categories = {}
    sources = {}
    for d in docs:
        cat = d.get('category', '未知')
        src = d.get('source', '未知')
        categories[cat] = categories.get(cat, 0) + 1
        sources[src] = sources.get(src, 0) + 1

    print("\n数据概况:")
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")
    print(f"\n来源数: {len(sources)}, 图片数: {sum(len(d.get('local_images', [])) for d in docs)}")

    # 构建索引
    index = SearchIndex()
    index.build(docs)
    index.save()

    print(f"\n索引已保存，可通过 Web 界面搜索。")


if __name__ == '__main__':
    main()
