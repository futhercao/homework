"""
灾害新闻专用爬虫 — 为作业三收集真实灾害事件报道
通过百度新闻搜索 + 已知灾害新闻源采集

用法: python scripts/crawl_disaster.py --max 500
"""
import os
import sys
import json
import re
import random
import hashlib
import asyncio
from datetime import datetime
from urllib.parse import urljoin, urlparse
from collections import deque

import aiohttp
import aiofiles

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.crawler.parser import ArticleParser
from src.crawler.dedup import ContentDeduplicator

# 百度新闻搜索关键词（覆盖所有灾害类型）
DISASTER_QUERIES = [
    '地震 遇难 救援',
    '地震 震级 应急响应',
    '台风 登陆 经济损失',
    '台风 应急响应 转移',
    '洪水 暴雨 遇难',
    '暴雨 山洪 经济损失',
    '山体滑坡 遇难 救援',
    '泥石流 遇难 经济损失',
    '森林火灾 过火面积',
    '自然灾害 遇难 应急管理部',
    '受灾 灾情 救灾',
]

# 种子URL（灾害新闻聚合）
SEED_URLS = [
    'https://news.sina.com.cn/society/',
    'https://society.people.com.cn/',
    'https://www.thepaper.cn/',
    'https://news.163.com/domestic/',
]

# 百度新闻URL模板
BAIDU_NEWS_URL = 'https://www.baidu.com/s?tn=news&rtt=1&wd={query}'

HEADERS_TEMPLATE = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1',
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'disaster_docs')
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'images')


def is_disaster_article(title, content):
    """判断是否为灾害新闻"""
    text = (title + content[:1000]).lower()
    keywords = ['地震', '台风', '洪水', '暴雨', '滑坡', '泥石流', '火灾', '干旱',
                '暴雪', '冰雹', '海啸', '龙卷风', '遇难', '受灾', '救灾', '灾情',
                '抢险', '救援', '应急响应', '转移群众', '倒塌', '经济损失']
    return any(kw in text for kw in keywords)


def extract_article_urls_from_baidu(html):
    """从百度新闻搜索页面提取文章URL"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    urls = set()
    for a in soup.find_all('a', href=True):
        href = a.get('href', '')
        # 百度新闻搜索结果中的真实URL通常在result中
        if href.startswith('http') and 'baidu.com' not in href and len(href) > 20:
            urls.add(href)
    return urls


async def crawl_disaster_news(max_docs=500, max_concurrent=10):
    """主爬取函数"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    dedup = ContentDeduplicator(threshold=3)
    parser = ArticleParser()
    documents = []
    visited = set()
    all_article_urls = set()

    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=3)
    timeout = aiohttp.ClientTimeout(total=25)
    session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    sem = asyncio.Semaphore(max_concurrent)

    async def fetch(url, referer=''):
        headers = {
            'User-Agent': random.choice(HEADERS_TEMPLATE),
            'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9',
        }
        if referer:
            headers['Referer'] = referer
        try:
            async with session.get(url, headers=headers, allow_redirects=True, ssl=False) as resp:
                if resp.status == 200:
                    return await resp.text()
        except Exception:
            pass
        return None

    async def collect_from_search(query):
        """从百度新闻搜索收集文章URL"""
        search_url = BAIDU_NEWS_URL.format(query=query)
        html = await fetch(search_url)
        if html:
            return extract_article_urls_from_baidu(html)
        return set()

    async def collect_from_seed(seed_url):
        """从种子页面收集文章链接"""
        html = await fetch(seed_url)
        if not html:
            return set()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'lxml')
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            if not href or not text or len(text) < 5:
                continue
            if any(href.startswith(p) for p in ['javascript:', '#', 'mailto:']):
                continue
            if not href.startswith('http'):
                href = urljoin(seed_url, href)
            if re.search(r'/\d{4}[-/]\d{2}[-/]\d{2}', href) or '.html' in href:
                links.add(href)
        return links

    async def process_article(url):
        """抓取并处理单篇文章"""
        if url in visited:
            return None
        async with sem:
            html = await fetch(url)
            if not html:
                visited.add(url)
                return None

            article = parser.parse(html, url)
            if not article:
                visited.add(url)
                return None

            if not is_disaster_article(article['title'], article['content']):
                visited.add(url)
                return None

            doc_id = 'dis_' + hashlib.md5(url.encode()).hexdigest()[:12]
            article['id'] = doc_id
            article['crawl_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            article['category'] = '灾害'

            # 去重
            if dedup.is_duplicate(article['content'], doc_id)[0]:
                visited.add(url)
                return None

            # 下载图片
            local_images = []
            for img in article.get('images', [])[:5]:
                src = img.get('src', '')
                if not src.startswith('http'):
                    continue
                try:
                    img_data = await fetch(src)
                    # Actually this should be raw bytes, but we use fetch which returns text
                    # For simplicity, use aiohttp directly for images
                    async with session.get(src, ssl=False) as resp:
                        if resp.status == 200:
                            img_bytes = await resp.read()
                            if len(img_bytes) > 1000:
                                img_id = hashlib.md5(src.encode()).hexdigest()[:12]
                                img_path = os.path.join(IMAGE_DIR, f'{img_id}.jpg')
                                async with aiofiles.open(img_path, 'wb') as f:
                                    await f.write(img_bytes)
                                local_images.append({
                                    'local_path': img_path,
                                    'original_src': src,
                                    'alt': img.get('alt', ''),
                                })
                except Exception:
                    pass

            article['local_images'] = local_images

            # 保存
            filepath = os.path.join(DATA_DIR, f'{doc_id}.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article, f, ensure_ascii=False, indent=2)

            visited.add(url)
            return article

    print(f'[灾害爬虫] Phase 1: 收集文章URL...')

    # 从百度新闻搜索收集
    for q in DISASTER_QUERIES:
        urls = await collect_from_search(q)
        all_article_urls.update(urls)
        print(f'  搜索 [{q}] → {len(urls)} URL')
        await asyncio.sleep(1)

    # 从种子页面收集
    for seed in SEED_URLS:
        urls = await collect_from_seed(seed)
        all_article_urls.update(urls)
        print(f'  种子 [{seed}] → {len(urls)} URL')

    print(f'  共收集 {len(all_article_urls)} 个候选URL')

    # Phase 2: 并发爬取
    print(f'[灾害爬虫] Phase 2: 并发爬取 ({max_concurrent}线程)...')

    pending = deque(all_article_urls)
    tasks = set()

    while (pending or tasks) and len(documents) < max_docs:
        while pending and len(tasks) < max_concurrent * 2:
            url = pending.popleft()
            if url not in visited:
                tasks.add(asyncio.create_task(process_article(url)))

        if not tasks:
            break

        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            result = await task
            if result:
                documents.append(result)
                if len(documents) % 25 == 0:
                    print(f'  [{len(documents)}/{max_docs}] {result["title"][:60]}')

    await session.close()

    # 统计
    types = {}
    for d in documents:
        t = d.get('content', '')[:500]
        for dt in ['地震', '台风', '洪水', '暴雨', '山体滑坡', '泥石流', '火灾', '干旱', '暴雪']:
            if dt in d['title'] or dt in t:
                types[dt] = types.get(dt, 0) + 1
                break

    print(f'\n[完成] {len(documents)} 篇灾害新闻')
    print(f'类型分布: {dict(sorted(types.items(), key=lambda x: -x[1]))}')
    return documents


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=500)
    await crawl_disaster_news(max_docs=parser.parse_args().max)


if __name__ == '__main__':
    asyncio.run(main())
