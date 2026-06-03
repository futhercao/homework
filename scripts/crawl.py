"""
爬虫运行脚本
用法:
    python scripts/crawl.py              # 默认800篇
    python scripts/crawl.py --max 500    # 指定数量
    python scripts/crawl.py --resume     # 断点续爬
"""
import os
import sys
import json
import asyncio
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.crawler import crawl_async
from src.config import CRAWLER_CONFIG, DOC_DIR


async def main():
    parser = argparse.ArgumentParser(description='信息检索系统 - 多源新闻爬虫')
    parser.add_argument('--max', type=int, default=800, help='目标文档数')
    parser.add_argument('--concurrent', type=int, default=15, help='并发数')
    args = parser.parse_args()

    print("=" * 60)
    print("  生产级多源新闻爬虫")
    print(f"  目标: {args.max} 篇 | 并发: {args.concurrent}")
    print("=" * 60)

    docs = await crawl_async(
        seed_urls=CRAWLER_CONFIG['seed_urls'],
        max_docs=args.max,
        max_concurrent=args.concurrent,
    )

    print(f"\n最终: {len(docs)} 篇文档已保存到 {DOC_DIR}")

    # 统计
    categories = {}
    sources = {}
    for d in docs:
        cat = d.get('category', '未知')
        src = d.get('source', '未知')
        categories[cat] = categories.get(cat, 0) + 1
        sources[src] = sources.get(src, 0) + 1

    print("\n类别分布:")
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    print("\n来源分布:")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1])[:15]:
        print(f"  {src}: {cnt}")

    total_images = sum(len(d.get('local_images', [])) for d in docs)
    print(f"\n图片: {total_images} 张已下载到本地")


if __name__ == '__main__':
    asyncio.run(main())
