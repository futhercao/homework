"""
真实新闻爬虫 - 从163.com爬取科技新闻和社会新闻
分别用于作业2（信息检索）和作业3（信息抽取）的真实数据补充

用法:
    python real_crawler.py tech 50     # 爬取50篇科技新闻
    python real_crawler.py society 50  # 爬取50篇社会新闻
"""
import os
import sys
import json
import time
import uuid
import re
import hashlib
import random
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}

# 科技新闻种子页面
TECH_SEEDS = [
    'https://tech.163.com/',
    'https://www.163.com/tech/',
]

# 社会新闻种子页面（可能含灾害新闻）
SOCIETY_SEEDS = [
    'https://news.163.com/domestic/',
    'https://news.163.com/shehui/',
]

# 额外科技新闻种子
EXTRA_TECH_SEEDS = [
    'https://www.36kr.com/newsflashes',
    'https://www.ithome.com/',
]


class RealNewsCrawler:
    """从真实新闻网站爬取文章"""

    def __init__(self, category='tech'):
        self.category = category
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.documents = []
        self.visited_urls = set()
        self.article_links = set()

    def crawl(self, max_docs=50):
        """爬取指定数量的文档"""
        if self.category == 'tech':
            seeds = TECH_SEEDS
        else:
            seeds = SOCIETY_SEEDS

        print(f"[爬虫] 类别: {self.category}, 目标: {max_docs} 篇")
        print(f"[爬虫] 种子页面: {seeds}")

        # Phase 1: Collect article links from seed pages
        print("\n[Phase 1] 收集文章链接...")
        for seed_url in seeds:
            self._collect_links_from_seed(seed_url)
            time.sleep(1)

        # Also crawl a few pages deeper
        article_list = list(self.article_links - self.visited_urls)
        print(f"  共收集到 {len(article_list)} 个文章链接")

        # Phase 2: Extract articles
        print(f"\n[Phase 2] 提取文章内容...")
        for url in article_list:
            if len(self.documents) >= max_docs:
                break
            if url in self.visited_urls:
                continue

            try:
                self.visited_urls.add(url)
                article = self._extract_article(url)
                if article:
                    self.documents.append(article)
                    print(f"  [{len(self.documents)}/{max_docs}] {article['title'][:50]}")
                time.sleep(random.uniform(1.0, 2.5))  # 礼貌延迟
            except Exception as e:
                print(f"  [跳过] {url[:60]}: {str(e)[:80]}")
                continue

        print(f"\n[爬虫] 完成! 共爬取 {len(self.documents)} 篇{self._category_name()}新闻")
        return self.documents

    def _collect_links_from_seed(self, seed_url):
        """从种子页面收集文章链接"""
        try:
            resp = self.session.get(seed_url, timeout=20)
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'lxml')

            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text(strip=True)

                # 筛选文章链接
                if self._is_article_url(href) and len(text) >= 5:
                    if not href.startswith('http'):
                        href = urljoin(seed_url, href)
                    if href not in self.visited_urls:
                        self.article_links.add(href)
        except Exception as e:
            print(f"  [警告] 收集链接失败 {seed_url}: {str(e)[:80]}")

    def _is_article_url(self, url):
        """判断URL是否是文章页面"""
        if not url or url.startswith(('javascript:', '#', 'mailto:', 'tel:')):
            return False
        # 163.com article pattern
        if '163.com' in url and ('/article/' in url or '/tech/' in url or '/news/' in url):
            return True
        # Generic article pattern
        if re.search(r'/\d{6,}/\w+\.html?$', url):
            return True
        if re.search(r'/article/\w+\.html?$', url):
            return True
        return False

    def _extract_article(self, url):
        """从文章页面提取标题和内容"""
        try:
            resp = self.session.get(url, timeout=20, allow_redirects=True)
            resp.encoding = 'utf-8'
        except Exception:
            try:
                resp = self.session.get(url, timeout=20)
                resp.encoding = 'utf-8'
            except Exception:
                return None

        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'lxml')

        # 移除无关元素
        for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'aside', 'iframe', 'noscript']):
            tag.decompose()

        # 提取标题
        title = self._extract_title(soup)
        if not title or len(title) < 5:
            return None

        # 提取正文
        content = self._extract_content(soup, url)
        if not content or len(content) < 100:
            return None

        # 提取日期
        date_str = self._extract_date(soup, resp.text)

        # 提取图片
        images = self._extract_images(soup, url)

        doc_id = hashlib.md5(url.encode()).hexdigest()[:10]

        return {
            'id': f'real_{doc_id}',
            'title': title.strip(),
            'content': content.strip(),
            'url': url,
            'date': date_str,
            'images': images,
            'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'real_crawl',
        }

    def _extract_title(self, soup):
        """提取文章标题"""
        for sel in ['h1', '.post_title', '.article-title', '.headline', '.title']:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(strip=True)
                if len(text) >= 5:
                    return text
        # Fallback: <title> tag
        title_tag = soup.find('title')
        if title_tag:
            text = title_tag.get_text(strip=True)
            # Remove site suffix
            for sep in ['_', '-', '|', '–']:
                if sep in text:
                    parts = [p.strip() for p in text.split(sep)]
                    # Return the longest part that's not purely a site name
                    parts = [p for p in parts if len(p) >= 5]
                    if parts:
                        return max(parts, key=len)
            if len(text) >= 5:
                return text
        return None

    def _extract_content(self, soup, url):
        """提取文章正文"""
        # 特定于163.com的选择器
        selectors_163 = [
            '.post_body', '.post_text', '.article-body',
            '.content', '#article-content', '.entry-content',
            '.post-content', '.article-content', '.art_content',
            '#endText', '.article', 'article',
        ]

        domain = urlparse(url).netloc

        # 先尝试常见选择器
        for sel in selectors_163:
            el = soup.select_one(sel)
            if el:
                # 移除内部无关元素
                for tag in el.find_all(['script', 'style', 'iframe']):
                    tag.decompose()
                text = el.get_text(separator='\n', strip=True)
                # 清理空行
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                text = '\n'.join(lines)
                if len(text) >= 100:
                    return text

        # Fallback: 取最长的文本块
        candidates = []
        for tag_name in ['div', 'section', 'main', 'article']:
            for el in soup.find_all(tag_name):
                text = el.get_text(separator='\n', strip=True)
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                text = '\n'.join(lines)
                if 100 < len(text) < 50000:
                    candidates.append((len(text), text))

        if candidates:
            # 选择长度适中的（不是导航栏等超长元素）
            candidates.sort(key=lambda x: x[0], reverse=True)
            # 取前3个候选中的第二个（避免导航栏等）
            best = candidates[min(1, len(candidates)-1)][1]
            return best

        return None

    def _extract_date(self, soup, html):
        """提取发布日期"""
        # 常见日期选择器
        for sel in ['.date', '.time', '.pub-date', '.publish-time',
                    'time', '.article-time', '.post_time', '.info']:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(strip=True)
                date_match = re.search(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})', text)
                if date_match:
                    return date_match.group(1)

        # 从HTML中搜索日期模式
        for pat in [
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{1,2})',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
        ]:
            m = re.search(pat, html[:10000])
            if m:
                return m.group(1)

        return datetime.now().strftime('%Y-%m-%d')

    def _extract_images(self, soup, base_url):
        """提取文章图片"""
        images = []
        content_area = None
        for sel in ['.post_body', '.article-body', 'article', '.content']:
            content_area = soup.select_one(sel)
            if content_area:
                break

        search_in = content_area if content_area else soup
        for img in search_in.find_all('img', src=True)[:3]:
            src = img.get('src', '')
            alt = img.get('alt', '')
            if src and not src.endswith(('.gif', '.svg', '.ico', 'beacon', 'pixel')):
                if not src.startswith('http'):
                    src = urljoin(base_url, src)
                # 过滤太小的图片
                images.append({'src': src, 'alt': alt or ''})
        return images

    def _category_name(self):
        return '科技' if self.category == 'tech' else '社会'

    def save(self, output_dir):
        """保存文档到指定目录"""
        os.makedirs(output_dir, exist_ok=True)
        for doc in self.documents:
            path = os.path.join(output_dir, f"{doc['id']}.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

        manifest = {
            'total': len(self.documents),
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': '163.com real crawl',
            'category': self._category_name(),
            'docs': [{'id': d['id'], 'title': d['title'], 'url': d['url']}
                     for d in self.documents],
        }
        with open(os.path.join(output_dir, 'manifest_real.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"[爬虫] 已保存 {len(self.documents)} 篇文档到 {output_dir}")


def crawl_for_homework2(num=50):
    """为作业2爬取科技新闻，混入现有数据"""
    crawler = RealNewsCrawler('tech')
    docs = crawler.crawl(max_docs=num)

    if not docs:
        print("[错误] 未能爬取到文章")
        return []

    # 保存到作业2的data目录
    hw2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'homework2', 'data', 'documents')
    crawler.save(hw2_dir)
    print(f"\n真实科技新闻已保存到: {hw2_dir}")
    print("请运行 homework2 的索引重建脚本以包含真实数据")
    return docs


def crawl_for_homework3(num=50):
    """为作业3爬取社会新闻，混入现有数据"""
    crawler = RealNewsCrawler('society')
    docs = crawler.crawl(max_docs=num)

    if not docs:
        print("[错误] 未能爬取到文章")
        return []

    hw3_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'homework3', 'data', 'documents')
    hw3_dir = os.path.abspath(hw3_dir)
    crawler.save(hw3_dir)
    print(f"\n真实社会新闻已保存到: {hw3_dir}")
    print("请运行 homework3 的抽取脚本以处理真实数据")
    return docs


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    category = sys.argv[1]
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    if category == 'tech':
        crawl_for_homework2(num)
    elif category == 'society':
        crawl_for_homework3(num)
    else:
        print(f'未知类别: {category}，可选: tech / society')
