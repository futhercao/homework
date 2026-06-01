"""
大规模真实新闻爬虫 - 目标150篇，并发无延迟
"""
import os, sys, json, time, re, hashlib, random
from datetime import datetime, timedelta
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

TECH_SOURCES = [
    'https://tech.163.com/',
    'https://www.163.com/tech/',
]
SOCIETY_SOURCES = [
    'https://news.163.com/domestic/',
    'https://news.163.com/shehui/',
    'https://www.163.com/news/',
    'https://news.163.com/world/',
    'https://news.163.com/special/',
]

session = requests.Session()
session.headers.update(HEADERS)


def _is_article_url(url):
    if not url or any(url.startswith(p) for p in ['javascript:', '#', 'mailto:']):
        return False
    if '163.com' in url and re.search(r'/(article|tech|news|dy/article|domestic|shehui|world)/', url):
        return True
    if re.search(r'/\d{4,}/\d{2}/\d{2}/', url):
        return True
    if re.search(r'/article/\w+\.html?$', url):
        return True
    if '.html' in url and '163.com' in url:
        return True
    return False


def collect_links(seed_urls):
    links = set()
    for url in seed_urls:
        try:
            resp = session.get(url, timeout=15)
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'lxml')
            for a in soup.find_all('a', href=True):
                href = a['href']
                if _is_article_url(href) and len(a.get_text(strip=True)) >= 5:
                    if not href.startswith('http'):
                        href = urljoin(url, href)
                    links.add(href)
        except:
            pass
    return links


def extract_article(url):
    try:
        resp = session.get(url, timeout=15, allow_redirects=True)
        resp.encoding = 'utf-8'
    except:
        return None
    if resp.status_code != 200 or len(resp.text) < 1000:
        return None

    soup = BeautifulSoup(resp.text, 'lxml')
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'aside', 'iframe']):
        tag.decompose()

    # Title
    title = None
    for sel in ['h1', '.post_title', '.article-title', '.headline']:
        el = soup.select_one(sel)
        if el:
            title = el.get_text(strip=True)
            if len(title) >= 5:
                break
    if not title:
        t = soup.find('title')
        if t:
            title = t.get_text(strip=True)
            for sep in ['_', '-', '|']:
                if sep in title:
                    parts = [p.strip() for p in title.split(sep) if len(p.strip()) >= 5]
                    if parts:
                        title = max(parts, key=len)
                        break
    if not title or len(title) < 5:
        return None

    # Content
    content = None
    for sel in ['.post_body', '.post_text', '.article-body', '.article-content',
                 '.content', '#endText', 'article']:
        el = soup.select_one(sel)
        if el:
            for t in el.find_all(['script', 'style']):
                t.decompose()
            text = el.get_text(separator='\n', strip=True)
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            text = '\n'.join(lines)
            if len(text) >= 100:
                content = text
                break
    if not content:
        candidates = [(len(el.get_text(strip=True)), el.get_text(separator='\n', strip=True))
                      for el in soup.find_all(['div', 'section'])
                      if 100 < len(el.get_text(strip=True)) < 50000]
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            content = candidates[min(1, len(candidates)-1)][1]
    if not content or len(content) < 100:
        return None

    # Date
    date_str = datetime.now().strftime('%Y-%m-%d')
    for pat in [r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', r'(\d{4}年\d{1,2}月\d{1,2}日)']:
        m = re.search(pat, resp.text[:15000])
        if m:
            date_str = m.group(1)
            break

    # Images
    images = []
    content_el = soup.select_one('.post_body') or soup.select_one('article') or soup
    for img in (content_el.find_all('img', src=True) if content_el else [])[:3]:
        src = img.get('src', '')
        if src and not any(src.endswith(x) for x in ['.gif', '.svg', '.ico']):
            if not src.startswith('http'):
                src = urljoin(url, src)
            images.append({'src': src, 'alt': img.get('alt', '')})

    return {
        'id': 'real_' + hashlib.md5(url.encode()).hexdigest()[:10],
        'title': title.strip(),
        'content': content.strip(),
        'url': url,
        'date': date_str,
        'images': images,
        'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'real_crawl',
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python bulk_crawl.py tech|society [数量默认150] [并发默认20]")
        return

    cat = sys.argv[1]
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    seeds = TECH_SOURCES if cat == 'tech' else SOCIETY_SOURCES
    cat_name = '科技' if cat == 'tech' else '社会'

    print(f"[爬虫] {cat_name}新闻, 目标{num}篇, {workers}线程并发")
    t0 = time.time()

    # Phase 1: 收集链接
    print("[Phase 1] 收集链接...")
    links = collect_links(seeds)
    # 也收集翻页（尝试多种分页格式）
    for page in range(2, 11):
        for seed in seeds:
            for fmt in [f"/index_{page:02d}.html", f"/index_{page}.html", f"/{page}.html", f"_0{page}.html"]:
                try:
                    resp = session.get(f"{seed.rstrip('/')}{fmt}", timeout=10)
                    resp.encoding = 'utf-8'
                    soup = BeautifulSoup(resp.text, 'lxml')
                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        if _is_article_url(href) and len(a.get_text(strip=True)) >= 5:
                            if not href.startswith('http'):
                                href = urljoin(seed, href)
                            links.add(href)
                except:
                    pass
    print(f"  收集到 {len(links)} 个链接")

    # Phase 2: 并发提取
    print(f"[Phase 2] 并发提取 ({workers}线程)...")
    docs = []
    pending = list(links)
    random.shuffle(pending)
    pending = pending[:num * 3]  # 多取一些以防失败

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_article, url): url for url in pending}
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            if result:
                docs.append(result)
                if len(docs) % 25 == 0 or len(docs) <= 5:
                    print(f"  [{len(docs)}/{num}] {result['title'][:50]}")
            if len(docs) >= num:
                break

    elapsed = time.time() - t0
    print(f"\n[完成] {len(docs)}篇, 耗时{elapsed:.0f}s ({elapsed/len(docs):.1f}s/篇)")

    # Save
    base = os.path.dirname(os.path.abspath(__file__))
    if cat == 'tech':
        out = os.path.join(base, 'homework2', 'data', 'documents')
    else:
        out = os.path.join(base, 'homework3', 'data', 'documents')
    os.makedirs(out, exist_ok=True)
    for doc in docs:
        path = os.path.join(out, f"{doc['id']}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"[保存] → {out}")


if __name__ == '__main__':
    main()
