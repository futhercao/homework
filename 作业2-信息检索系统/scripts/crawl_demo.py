"""
现场演示脚本 —— 证明本系统的爬虫"真实可爬" (作业验收"爬取数据"项 + 现场演示要求)

为什么单独提供本脚本:
  `python run.py crawl` 会把新抓到的文章写入交付语料目录 data/documents/ 并追加进
  断点文件, 从而改变随报告交付的"802 篇"语料规模。验收时只是想看"爬虫当场能不能爬",
  并不希望污染已交付、已建索引的语料。故本脚本把落盘目录重定向到一个**临时隔离目录**
  (系统 temp, 用完即弃), 用全新空断点, 当场实时抓取数毫无几篇真实新闻并打印出来,
  绝不触碰 data/documents/ 与已构建的索引。

演示讲解要点 (对应"现场解释"):
  1. 异步高并发:   aiohttp + Semaphore 控制并发, 单域名限流;
  2. 反反爬:       15 个 User-Agent 轮换 + Referer 伪装 + 失败重试/指数退避;
  3. 多源种子:     新华网/人民网/央视网/中新网/光明网/中国经济网/科技日报 等权威采编源;
  4. 两阶段抓取:   Phase 1 从频道页收集候选链接 → Phase 2 并发抓正文 + 浅层 BFS 发现新链接;
  5. 去重:         SimHash 近似去重, 避免转载重复入库;
  6. 真实多媒体:   顺带下载文章配图到本地 (供以文搜图/以图搜文)。

用法:
    python scripts/crawl_demo.py            # 默认现场抓 5 篇
    python scripts/crawl_demo.py --max 8    # 自定义篇数
"""
import os
import sys
import asyncio
import tempfile
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.crawler.engine as eng


async def _demo(max_docs):
    # 把落盘目录重定向到隔离临时目录, 用全新空断点 → 绝不污染交付语料
    tmp = tempfile.mkdtemp(prefix='crawl_demo_')
    eng.DOC_DIR = tmp
    eng.IMAGE_DIR = os.path.join(tmp, 'images')
    os.makedirs(eng.IMAGE_DIR, exist_ok=True)

    engine = eng.AsyncCrawlerEngine()
    engine.checkpoint_file = os.path.join(tmp, '.checkpoint.json')
    engine.visited_urls = set()

    print('=' * 64)
    print('  现场爬虫演示 — 实时抓取真实新闻 (隔离目录, 不动交付语料)')
    print('=' * 64)
    docs = await engine.crawl(max_docs=max_docs, max_concurrent=10)

    print('\n' + '=' * 64)
    print(f'  本次现场抓取 {len(docs)} 篇 (失败 {engine.stats.get("failed", 0)}, '
          f'图片 {engine.stats.get("images_downloaded", 0)} 张)')
    print('=' * 64)
    for i, d in enumerate(docs, 1):
        print(f'[{i}] 来源={d.get("source", "?")}  日期={d.get("date", "?")}  '
              f'正文 {len(d.get("content", ""))} 字  配图 {len(d.get("images", []))} 张')
        print(f'    标题: {d.get("title", "")[:50]}')
        print(f'    URL : {d.get("url", "")}')
    print(f'\n隔离输出目录(用完即弃, 不影响交付语料): {tmp}')
    print('交付语料仍为 data/documents/ 下的 802 篇, 索引未变。')


def main():
    ap = argparse.ArgumentParser(description='现场爬虫演示 (安全隔离)')
    ap.add_argument('--max', type=int, default=5, help='本次现场抓取篇数 (默认 5)')
    args = ap.parse_args()
    asyncio.run(_demo(args.max))


if __name__ == '__main__':
    main()
