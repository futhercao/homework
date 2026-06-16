"""
现场演示脚本 —— 证明本系统的灾害爬虫"真实可爬" (作业验收"爬取数据"项 + 现场演示要求)

为什么单独提供本脚本:
  `python run.py crawl` 会把新抓到的灾害报道写入交付语料目录 data/disaster_docs/ 并重建
  metadata.db, 从而改变随报告交付的"118 篇真实灾害事件"语料规模与抽取结果。验收时只是
  想看"爬虫当场能不能爬", 并不希望污染已交付、已抽取、已建图谱的语料。故本脚本把落盘目录
  重定向到一个**临时隔离目录**(系统 temp, 用完即弃), 当场实时抓取数毫篇真实灾害新闻并打印,
  绝不触碰 data/disaster_docs/ / metadata.db / extractions.json / 知识图谱。

演示讲解要点 (对应"现场解释"):
  1. 多入口召回:   百度新闻搜索(13 个灾害关键词) + 应急部/消防局/水利部/气象局 等权威频道种子;
  2. 权威源白名单: 仅保存 gov.cn/新华/人民/央视/中新/光明/央广/澎湃 等权威源, 滤掉门户与自媒体;
  3. 三重过滤:     关键词初筛 → 正则引擎校验真实灾害类型 → 事件性过滤(剔除党务/会议/科普/导航);
  4. 浅层 BFS:     在已确认的灾害页面顺带发现更多文章链接, 自然提升数量与域名多样性;
  5. 单源配额:     单一来源最多 35 篇, 强制来源/题材多样性;
  6. 真实多媒体:   按文件魔数判定下载文章真实配图 (供后续 EasyOCR 抽取)。

用法:
    python scripts/crawl_demo.py            # 默认现场抓 5 篇灾害新闻
    python scripts/crawl_demo.py --max 8    # 自定义篇数
"""
import os
import sys
import asyncio
import tempfile
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scripts.crawl_disaster as cd


async def _demo(max_docs):
    # 把落盘目录重定向到隔离临时目录 → 绝不污染交付语料 / 抽取结果 / 知识图谱
    tmp = tempfile.mkdtemp(prefix='discrawl_demo_')
    cd.DATA_DIR = tmp
    cd.IMAGE_DIR = os.path.join(tmp, 'images')
    os.makedirs(cd.IMAGE_DIR, exist_ok=True)

    print('=' * 64)
    print('  现场灾害爬虫演示 — 实时抓取真实灾害新闻 (隔离目录, 不动交付语料)')
    print('=' * 64)
    docs = await cd.crawl_disaster_news(max_docs=max_docs, max_concurrent=10)

    print('\n' + '=' * 64)
    print(f'  本次现场抓取 {len(docs)} 篇真实灾害事件报道')
    print('=' * 64)
    for i, d in enumerate(docs, 1):
        print(f'[{i}] 来源={d.get("source", "?")}  类型={d.get("disaster_type", "?")}  '
              f'日期={d.get("date", "?")}  正文 {len(d.get("content", ""))} 字  '
              f'配图 {len(d.get("local_images", []))} 张')
        print(f'    标题: {d.get("title", "")[:50]}')
        print(f'    URL : {d.get("url", "")}')
    print(f'\n隔离输出目录(用完即弃, 不影响交付语料): {tmp}')
    print('交付语料仍为 data/disaster_docs/ 下的 118 篇, 抽取结果/知识图谱未变。')


def main():
    ap = argparse.ArgumentParser(description='现场灾害爬虫演示 (安全隔离)')
    ap.add_argument('--max', type=int, default=5, help='本次现场抓取篇数 (默认 5)')
    args = ap.parse_args()
    asyncio.run(_demo(args.max))


if __name__ == '__main__':
    main()
