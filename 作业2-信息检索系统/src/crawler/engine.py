"""
异步分布式爬虫引擎
- aiohttp 异步高并发
- UA轮换 + 自适应限速
- 失败重试 + 指数退避
- SimHash去重
- 断点续爬
- 图片本地存储
"""
import os
import sys
import json
import re
import time
import uuid
import random
import hashlib
import asyncio
from datetime import datetime
from urllib.parse import urljoin, urlparse
from collections import deque

import aiohttp
import aiofiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import CRAWLER_CONFIG, DOC_DIR, IMAGE_DIR
from src.crawler.parser import ArticleParser
from src.crawler.dedup import ContentDeduplicator
from loguru import logger


class AsyncCrawlerEngine:
    """异步爬虫引擎 — 支持多站点、高并发、智能反反爬"""

    def __init__(self, config=None):
        self.config = config or CRAWLER_CONFIG
        self.parser = ArticleParser()
        self.dedup = ContentDeduplicator(threshold=3)
        self.documents = []
        self.visited_urls = set()
        self.failed_urls = {}
        self.stats = {'total': 0, 'success': 0, 'failed': 0, 'duplicate': 0,
                       'images_downloaded': 0}
        self.semaphore = None
        self.session = None
        self.checkpoint_file = os.path.join(DOC_DIR, '.crawler_checkpoint.json')
        os.makedirs(DOC_DIR, exist_ok=True)
        os.makedirs(IMAGE_DIR, exist_ok=True)

    # ============ 主入口 ============

    async def crawl(self, seed_urls=None, max_docs=800, max_concurrent=20,
                    focus_categories=None):
        """主爬取方法"""
        seed_urls = seed_urls or self.config['seed_urls']
        self.semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"[引擎] 启动: {len(seed_urls)} 种子, 目标 {max_docs} 篇, {max_concurrent} 并发")

        # 恢复断点
        self._load_checkpoint()

        # 创建会话
        connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=5, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=self.config['request_timeout'])
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        # Phase 1: 从种子页收集文章链接
        article_urls = set()
        for seed_url in seed_urls:
            links = await self._collect_links(seed_url)
            article_urls.update(links)
            logger.info(f"  [种子] {seed_url} → {len(links)} 个链接")

        logger.info(f"[Phase 1] 共收集 {len(article_urls)} 个候选链接")

        # Phase 2: 并发爬取文章 + 发现新链接
        pending = deque(article_urls)
        tasks = set()

        while (pending or tasks) and len(self.documents) < max_docs:
            # 填满任务槽
            while pending and len(tasks) < max_concurrent * 2:
                url = pending.popleft()
                if url not in self.visited_urls:
                    task = asyncio.create_task(self._crawl_article(url))
                    tasks.add(task)

            if not tasks:
                break

            # 等待任意任务完成
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    result = await task
                    if result:
                        doc, new_links = result
                        if not self.dedup.is_duplicate(doc['content'], doc['id'])[0]:
                            self._save_document(doc)
                            self.documents.append(doc)
                            self.visited_urls.add(doc['url'])
                            self.stats['success'] += 1
                            if len(self.documents) % 50 == 0:
                                self._save_checkpoint()
                                logger.info(f"  [{len(self.documents)}/{max_docs}] {doc['title'][:50]}")
                        else:
                            self.stats['duplicate'] += 1

                        # 从已爬文章中提取新链接加入队列
                        for link in new_links:
                            if link not in self.visited_urls and len(pending) < max_docs * 3:
                                pending.append(link)
                    else:
                        self.stats['failed'] += 1
                except Exception as e:
                    self.stats['failed'] += 1

            self.stats['total'] = self.stats['success'] + self.stats['failed'] + self.stats['duplicate']

        # 清理
        await self.session.close()
        self._save_checkpoint()

        logger.info(f"[完成] 成功 {len(self.documents)} 篇, "
                     f"失败 {self.stats['failed']}, 去重 {self.stats['duplicate']}")
        return self.documents

    # ============ 核心方法 ============

    async def _crawl_article(self, url):
        """爬取单篇文章，返回 (doc, new_links) 或 None"""
        async with self.semaphore:
            html = await self._fetch(url)
            if not html:
                return None

            article = self.parser.parse(html, url)
            if not article:
                return None

            doc_id = 'doc_' + hashlib.md5(url.encode()).hexdigest()[:12]
            article['id'] = doc_id
            article['crawl_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 提取页面中的新链接
            new_links = self._extract_internal_links(html, url)

            # 下载图片到本地
            await self._download_images(article)

            return article, new_links

    async def _fetch(self, url):
        """HTTP GET with 重试和退避"""
        for attempt in range(self.config['max_retries']):
            try:
                headers = {
                    'User-Agent': random.choice(self.config['user_agents']),
                    'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Referer': self._get_referer(url),
                }
                async with self.session.get(url, headers=headers, allow_redirects=True,
                                             ssl=False) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    elif resp.status in (429, 503):
                        wait = 2 ** attempt + random.uniform(0, 1)
                        await asyncio.sleep(wait)
                    elif resp.status in (403, 404):
                        return None
            except Exception:
                await asyncio.sleep(1 + attempt * 0.5)

        self.failed_urls[url] = 'max_retries_exceeded'
        return None

    async def _collect_links(self, page_url):
        """从列表页收集文章URL"""
        links = set()
        try:
            html = await self._fetch(page_url)
            if not html:
                return links
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'lxml')
            base_domain = urlparse(page_url).netloc

            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text(strip=True)

                if self._is_article_url(href, text, base_domain):
                    if not href.startswith('http'):
                        href = urljoin(page_url, href)
                    if self._is_valid_article_url(href):
                        links.add(href)

            # 尝试翻页收集更多
            for i in range(2, 6):
                for fmt in [f'index_{i}.html', f'index_{i}.shtml', f'_{i}.html',
                           f'/page/{i}', f'?page={i}', f'list_{i}.shtml']:
                    try:
                        page = urljoin(page_url, fmt)
                        html2 = await self._fetch(page)
                        if html2:
                            soup2 = BeautifulSoup(html2, 'lxml')
                            for a in soup2.find_all('a', href=True):
                                href = a['href']
                                if self._is_article_url(href, a.get_text(strip=True), base_domain):
                                    if not href.startswith('http'):
                                        href = urljoin(page_url, href)
                                    if self._is_valid_article_url(href):
                                        links.add(href)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"  [链接收集失败] {page_url}: {e}")

        return links

    def _is_article_url(self, href, text, base_domain):
        """判断URL是否为文章页"""
        if not href or not text:
            return False
        if any(href.startswith(p) for p in ['javascript:', '#', 'mailto:', 'tel:']):
            return False
        if len(text) < 5:
            return False
        # 文章URL特征 (通用, 适配门户与权威采编源)
        patterns = [
            r'/\d{4,}/\d{2}/\d{2}/', r'/article/', r'/a/\d+',
            r'/\d{6,}/\d{2}/', r'/news/\d', r'\.s?html?',   # .htm/.html/.shtm/.shtml
            r'/detail/', r'/content/', r'/story/', r'/read/',
            # 权威源文章 id / 日期式: 新华&央视 c_<id>; 光明&经济网 content_<id>; 经济网 t20240615_; 光明网 /2024-06/15/
            r'/c_\d+', r'/content_\d+', r'/t\d{8}', r'/\d{4}-\d{2}/\d{2}/',
        ]
        return any(re.search(p, href) for p in patterns)

    def _is_valid_article_url(self, url):
        """过滤非文章URL"""
        exclude = ['search', 'tag', 'category', 'author', 'about', 'login',
                    'register', 'video', 'live', 'topic', 'special', 'photo',
                    'comment', 'reply', '举报', '反馈']
        url_lower = url.lower()
        return not any(e in url_lower for e in exclude)

    def _extract_internal_links(self, html, base_url):
        """从文章页提取同站链接"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'lxml')
        base_domain = urlparse(base_url).netloc
        links = set()
        for a in soup.find_all('a', href=True)[:30]:
            href = a['href']
            text = a.get_text(strip=True)
            if self._is_article_url(href, text, base_domain):
                if not href.startswith('http'):
                    href = urljoin(base_url, href)
                if self._is_valid_article_url(href) and href not in self.visited_urls:
                    links.add(href)
        return links

    def _get_referer(self, url):
        domain = urlparse(url).netloc
        return f'https://{domain}/'

    # ============ 图片下载 ============

    async def _download_images(self, doc):
        """下载文章中的图片到本地存储"""
        images = doc.get('images', [])
        local_images = []

        for img in images[:5]:  # 每篇最多5张
            src = img.get('src', '')
            if not src or not src.startswith('http'):
                continue
            try:
                img_id = hashlib.md5(src.encode()).hexdigest()[:12]
                ext = os.path.splitext(urlparse(src).path)[1] or '.jpg'
                if ext not in ('.jpg', '.jpeg', '.png', '.webp', '.gif'):
                    ext = '.jpg'
                local_path = os.path.join(IMAGE_DIR, f'{img_id}{ext}')

                if not os.path.exists(local_path):
                    async with self.session.get(src, ssl=False) as resp:
                        if resp.status == 200 and int(resp.headers.get('Content-Length', 0)) > 1000:
                            content = await resp.read()
                            if len(content) > 1000:
                                async with aiofiles.open(local_path, 'wb') as f:
                                    await f.write(content)
                                self.stats['images_downloaded'] += 1

                if os.path.exists(local_path):
                    local_images.append({
                        'local_path': local_path,
                        'original_src': src,
                        'alt': img.get('alt', ''),
                    })
            except Exception:
                pass

        doc['local_images'] = local_images
        if local_images:
            doc['images'] = [{'src': li['local_path'], 'alt': li['alt'],
                              'original_src': li['original_src']} for li in local_images]

    # ============ 存储 ============

    def _save_document(self, doc):
        """保存文档到JSON文件"""
        path = os.path.join(DOC_DIR, f"{doc['id']}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    def _save_checkpoint(self):
        """保存断点"""
        cp = {
            'num_documents': len(self.documents),
            'visited_urls': list(self.visited_urls),
            'failed_urls': self.failed_urls,
            'stats': self.stats,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(cp, f, ensure_ascii=False)

    def _load_checkpoint(self):
        """恢复断点"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                cp = json.load(f)
            self.visited_urls = set(cp.get('visited_urls', []))
            self.failed_urls = cp.get('failed_urls', {})
            self.stats = cp.get('stats', self.stats)
            logger.info(f"[断点恢复] 已爬 {cp.get('num_documents', 0)} 篇, "
                        f"已访问 {len(self.visited_urls)} URL")

    @staticmethod
    def load_documents(doc_dir=None):
        """静态方法：加载所有已爬取文档"""
        doc_dir = doc_dir or DOC_DIR
        docs = []
        if not os.path.exists(doc_dir):
            return docs
        for fname in sorted(os.listdir(doc_dir)):
            if fname.endswith('.json') and not fname.startswith('.'):
                path = os.path.join(doc_dir, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        docs.append(json.load(f))
                except Exception:
                    pass
        return docs


# ============ 便捷入口 ============

async def crawl_async(seed_urls=None, max_docs=800, max_concurrent=20,
                      focus_categories=None):
    """异步爬虫入口"""
    engine = AsyncCrawlerEngine()
    return await engine.crawl(seed_urls, max_docs, max_concurrent, focus_categories)


def crawl(seed_urls=None, max_docs=800):
    """同步爬虫入口（从命令行调用）"""
    return asyncio.run(crawl_async(seed_urls, max_docs))
