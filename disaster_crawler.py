"""
真实灾害新闻爬虫 - 通过关键词搜索+多站点适配爬取真实灾害报道
支持: sina.com.cn, sohu.com, 163.com, news.qq.com, thepaper.cn
"""
import os, sys, json, time, re, hashlib, random
from datetime import datetime
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

# 灾害关键词搜索
DISASTER_KEYWORDS = [
    '地震 遇难 应急响应',
    '台风 登陆 经济损失',
    '洪水 暴雨 遇难 救援',
    '山体滑坡 遇难 救援',
    '泥石流 遇难 经济损失',
    '森林火灾 过火面积',
    '暴雪 受灾 经济损失',
    '干旱 受灾 饮水困难',
]

# 搜索源（百度新闻、搜狗新闻等）
SEARCH_URLS = [
    'https://www.baidu.com/s?tn=news&word={keyword}',
]

# 已收集的灾害新闻URL（通过搜索引擎逐一确认）
KNOWN_ARTICLES = [
    # === 地震类 ===
    # 广西柳州5.2级地震 (2026年5月18日)
    'http://news.china.com.cn/2026-05/18/content_118499615.html',
    'http://www.hn.chinanews.com.cn/news/gnxw/2026/0522/528761.html',
    'https://news.cnr.cn/native/gd/kx/20260518/t20260518_527624321.shtml',
    'https://www.thepaper.cn/newsDetail_forward_33193192',
    'https://news.cri.cn/2026-05-18/94df1305-55c4-9707-daf1-6112baa01f2c.html',
    'https://www.thepaper.cn/newsDetail_forward_33192927',
    'http://www.gxzf.gov.cn/zzqzyxx/t27626466.shtml',
    'http://news.china.com.cn/2026-05/19/content_118502658.html',
    'http://www.bjnews.com.cn/detail-1780010673129416.html',
    'https://www.thepaper.cn/newsDetail_forward_33267168',
    'http://www.163.com/dy/article/KT6NLLR6051497H3.html',
    'http://lzzx.liuzhou.gov.cn/xw/202605/t20260520_3753432.html',
    'http://zjt.gxzf.gov.cn/xwdt/btdt/t27682813.shtml',
    # 菲律宾7.6级地震 (2025年10月10日)
    'https://news.sina.com.cn/w/2025-10-10/doc-inftkkxn9095843.shtml',
    # 日本7.5级地震 (2025年12月10日)
    'https://news.sina.com.cn/w/2025-12-10/doc-inhahnvm0721555.shtml',
    # 智利6.0级地震 (2026年6月1日)
    'https://news.sina.com.cn/w/2026-06-01/doc-inhzvxau5582445.shtml',

    # === 洪水/暴雨类 ===
    # 湖南石门暴雨 (2026年5月)
    'https://news.inewsweek.cn/society/2026-05-21/30270.shtml',
    'https://www.cqcb.com/shixiang/2026-05-20/6143485_pc.html',
    'https://news.sina.com.cn/c/2026-05-22/doc-inhytqkr5850526.shtml',
    'https://www.163.com/dy/article/KTG1QDQL055040N3.html',
    'http://ctdsb.net/c1476_202605/2747512.html',
    'https://m.gmw.cn/2026-05/19/content_1304462999.htm',
    'https://news.qq.com/rain/a/20260522V069MC00',
    'https://static.cdsb.com/micropub/Articles/202605/8f189ca8fbe5eb1579ab4b8409153803.html',
    'https://news.qq.com/rain/a/20260530V075RN00',
    'https://news.qq.com/rain/a/20260521V09MZL00',
    'https://k.sina.cn/article_1720962692_m6693ce84020032a88.html',
    'https://m.gmw.cn/2026-05/21/content_1304465183.htm',
    'http://news.anhuinews.com/shehui/202605/t20260520_9281544.html',
    'https://news.sina.com.cn/c/2026-05-20/doc-inhyqcsk6353617.shtml',
    # 重庆永川暴雨 (2026年5月)
    'https://cbgc.scol.com.cn/news/7617715',
    # 广东平远强降雨 (2024年6月 - 重要参考)
    'https://5g.dahe.cn/news/202406211775007',

    # === 山体滑坡/泥石流类 ===
    # 四川宜宾筠连山体滑坡 (2025年2月)
    'https://www.sohu.com/a/984997659_114988',
    'https://www.sohu.com/a/985043671_163278',
    'http://1rd.cregc.com.cn/eportal/ui?pageId=635936&articleKey=668558&columnId=636074',
    # 贵州贵定铁锁岩村山体滑坡 (2026年5月)
    'https://www.sohu.com/a/1025420176_119038',
    'https://china.cnr.cn/gdgg/20260521/t20260521_527629914.shtml',
    'https://t.m.youth.cn/transfer/index/url/news.youth.cn/gn/202605/t20260521_16671599.htm',
    'https://m.gmw.cn/2026-05/21/content_1304464772.htm',
    # 云南文山马关泥石流 (2026年5月)
    'https://www.sohu.com/a/1030511828_114988',
    'https://www.sohu.com/a/1030460009_121347613',
    'https://www.ztnews.net/article/show-487375.html',
    'http://yn.people.com.cn/BIG5/n2/2026/0524/c372456-41589485.html',
    'https://app.xinhuanet.com/news/article.html?articleId=20260528f52934f0dd8f4ba2b4f0dbcab074cb5f',
    'https://finance.sina.com.cn/jjxw/2026-06-01/doc-inhzwpyu4083679.shtml',
    'https://k.sina.com.cn/article_2090512390_7c9ab006020039eo0.html',
    # 广东信宜滑坡 (2025年6月)
    'http://k.sina.com.cn/article_5787187353_158f17899020023z8q.html',
    # 广西百色泥石流 (2025年9月)
    'https://www.sohu.com/a/1021546853_121019331',

    # === 台风类 ===
    # 超强台风"杨柳" (2025年8月)
    'https://103.30.70.4/sc/informtc/podul25/report.html',
    # 台风"Ragasa" (2026年5月, 广东登陆)

    # === 综合灾情报告 ===
    'https://www.163.com/dy/article/KJDNFN8D0514D3UH.html',  # 2025全国自然灾害
    'https://news.sina.com.cn/c/2025-10-17/doc-infuehut6381008.shtml',  # 前三季度自然灾害
    'https://finance.sina.com.cn/jjxw/2026-05-09/doc-inhxhzaz1384514.shtml',  # 2026年4月自然灾害
    'https://news.qq.com/rain/a/20260423A03D1600',  # 2026一季度自然灾害
    'https://www.thepaper.cn/newsDetail_forward_33035289',  # 2026一季度自然灾害
    'https://m.gmw.cn/2026-04/23/content_1304430393.htm',  # 一季度自然灾害
]


def search_baidu_news(keyword, session):
    """通过百度新闻搜索获取更多文章URL"""
    urls = set()
    try:
        url = f'https://www.baidu.com/s?tn=news&rtt=1&bsst=1&cl=2&wd={keyword}'
        resp = session.get(url, timeout=15, headers=HEADERS)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')
        for a in soup.find_all('a', href=True):
            href = a['href']
            # 提取百度新闻搜索结果中的真实URL
            if 'article' in href or '/a/' in href or '/c/' in href:
                if not href.startswith('http'):
                    continue
                if any(domain in href for domain in ['sina.com.cn', 'sohu.com', '163.com', 'qq.com', 'thepaper.cn']):
                    urls.add(href)
    except Exception as e:
        print(f'  [搜索异常] {keyword}: {e}')
    return urls


def search_bing_news(keyword, session):
    """通过Bing新闻搜索获取文章URL"""
    urls = set()
    try:
        url = f'https://www.bing.com/news/search?q={keyword}&qft=interval%3d%225%22&form=YFNR'
        resp = session.get(url, timeout=15, headers=HEADERS)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')
        for a in soup.find_all('a', href=True):
            href = a['href']
            if not href.startswith('http'):
                continue
            if any(domain in href for domain in ['sina.com.cn', 'sohu.com', '163.com', 'qq.com', 'thepaper.cn',
                                                   'people.com.cn', 'xinhuanet.com', 'chinanews.com']):
                urls.add(href)
    except Exception as e:
        print(f'  [Bing搜索异常] {keyword}: {e}')
    return urls


def extract_generic(url, session):
    """通用文章提取，适配大多数中文新闻站点"""
    try:
        resp = session.get(url, timeout=15, headers=HEADERS, allow_redirects=True)
        resp.encoding = resp.apparent_encoding or 'utf-8'
    except:
        return None

    if resp.status_code != 200 or len(resp.text) < 500:
        return None

    soup = BeautifulSoup(resp.text, 'lxml')
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'aside', 'iframe', 'noscript', 'header']):
        tag.decompose()

    # === 标题提取 ===
    title = ''
    # 尝试常见标题选择器
    for sel in ['h1', '.article-title', '.post-title', '.news-title', '.main-title',
                '.title', '#title', '.art_title', '.con_title', 'h2.title',
                '.text-title', '.content-title', '.entry-title', '.page-title']:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(strip=True)
            if len(t) >= 5 and len(t) < 200:
                title = t
                break

    if not title:
        t = soup.find('title')
        if t:
            title = t.get_text(strip=True)
            # 分割取最长片段
            for sep in ['_', '-', '|', '–']:
                if sep in title:
                    parts = [p.strip() for p in title.split(sep) if len(p.strip()) >= 5]
                    if parts:
                        title = max(parts, key=len)
                        break

    if not title or len(title) < 5 or len(title) > 300:
        return None

    # === 正文提取 ===
    content = ''
    # 尝试常见正文选择器
    content_selectors = [
        '.article-content', '.article-body', '.post-content', '.post_body',
        '#article', '.article', '.content', '#content', '.main-content',
        '.news-content', '.entry-content', '.post-text', '#endText',
        '.text-content', '.detail-content', '.art_content', '.con_text',
        '#Cnt-Main-Article-QQ', '.TRS_Editor', '.Custom_UnionStyle',
        '.show_text', '.news_txt', '.content-article', '.article-con',
    ]
    for sel in content_selectors:
        el = soup.select_one(sel)
        if el:
            for t in el.find_all(['script', 'style']):
                t.decompose()
            text = el.get_text(separator='\n', strip=True)
            lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 5]
            if len(lines) >= 3:
                content = '\n'.join(lines)
                break

    # Fallback: 找最长的<div>或<section>
    if not content:
        candidates = []
        for tag_name in ['div', 'section', 'main', 'article']:
            for el in soup.find_all(tag_name):
                text = el.get_text(separator='\n', strip=True)
                lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 5]
                text = '\n'.join(lines)
                if 200 < len(text) < 100000:
                    candidates.append((len(text), text))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            # 取第二长的（最长可能是导航）
            idx = min(1, len(candidates) - 1)
            content = candidates[idx][1]

    if not content or len(content) < 100:
        return None

    # === 日期提取 ===
    date_str = datetime.now().strftime('%Y-%m-%d')
    date_patterns = [
        r'(\d{4}年\d{1,2}月\d{1,2}日)',
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})',
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{4}/\d{2}/\d{2})',
    ]
    html_snippet = resp.text[:15000]
    for pat in date_patterns:
        matches = re.findall(pat, html_snippet)
        if matches:
            date_str = matches[0] if isinstance(matches[0], str) else matches[0][0]
            break

    # 同时找时间标签
    if date_str == datetime.now().strftime('%Y-%m-%d'):
        for sel in ['.date', '.time', '.pub-date', '.publish-time', 'time', '.article-time',
                     '.post_time', '.info', '.source', '.time-source']:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(strip=True)
                for pat in date_patterns:
                    m = re.search(pat, text)
                    if m:
                        date_str = m.group(1)
                        break
                if date_str != datetime.now().strftime('%Y-%m-%d'):
                    break

    doc_id = 'real_' + hashlib.md5(url.encode()).hexdigest()[:10]
    return {
        'id': doc_id, 'title': title.strip(), 'content': content.strip(),
        'url': url, 'date': date_str, 'images': [],
        'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'real_crawl',
    }


# 站点特定的提取器别名
extract_sina = extract_generic
extract_sohu = extract_generic


def is_disaster_article(title, content):
    """判断文章是否真的是灾害新闻（较宽松，保留灾害相关报道）"""
    text = f'{title} {content[:1000]}'
    disaster_keywords = ['地震', '台风', '飓风', '洪水', '洪涝', '暴雨', '山洪',
                          '山体滑坡', '滑坡', '泥石流', '森林火灾', '森林大火',
                          '草原火灾', '火灾', '干旱', '暴雪', '雪灾', '低温雨雪',
                          '冰雹', '海啸', '火山', '龙卷风', '风雹', '沙尘暴',
                          '遇难', '受灾', '灾情', '救灾', '抢险', '救援队',
                          '应急响应', '转移群众', '倒塌', '经济损失', '失联']
    has_keyword = any(kw in text for kw in disaster_keywords)
    if not has_keyword:
        return False
    exclude_patterns = [
        r'防灾减灾.*小知识', r'科普.*灾害', r'灾害.*科普',
    ]
    for pat in exclude_patterns:
        if re.search(pat, text):
            return False
    return True


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"[爬虫] 开始爬取 {len(KNOWN_ARTICLES)} 个已知灾害新闻URL...")
    docs = []
    seen_urls = set()
    failed_urls = []

    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(extract_generic, url, session): url for url in KNOWN_ARTICLES}
        for fut in as_completed(futures):
            url = futures[fut]
            result = fut.result()
            if result:
                if is_disaster_article(result['title'], result['content']):
                    docs.append(result)
                    seen_urls.add(url)
                    dtype = ''
                    for dt in ['地震', '台风', '洪水', '山体滑坡', '泥石流', '火灾', '干旱', '暴雪', '暴雨']:
                        if dt in result['title'] or dt in result['content'][:200]:
                            dtype = dt
                            break
                    print(f"  [{len(docs)}] [{dtype or '灾害'}] {result['title'][:65]}")
                else:
                    print(f"  [跳过非灾害] {result['title'][:55]}")
            else:
                failed_urls.append(url)

    # 也尝试从一些新闻聚合页面获取更多链接
    print(f"\n[Phase 2] 尝试从已爬取页面发现更多链接...")
    # 添加一些种子页面的链接发现
    additional_seeds = [
        'https://news.163.com/domestic/',
        'https://news.sina.com.cn/society/',
        'https://www.thepaper.cn/',
    ]
    extra_links = set()
    for seed in additional_seeds:
        try:
            resp = session.get(seed, timeout=10, headers=HEADERS)
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'lxml')
            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text(strip=True)
                if not href.startswith('http'):
                    href = urljoin(seed, href)
                if href not in seen_urls and len(text) >= 8:
                    for kw in ['地震', '台风', '洪水', '滑坡', '泥石流', '火灾', '干旱', '暴雪', '暴雨', '遇难']:
                        if kw in text:
                            extra_links.add(href)
                            break
        except:
            pass
    print(f"  发现 {len(extra_links)} 个额外灾害链接")

    # 爬取额外链接
    if extra_links:
        with ThreadPoolExecutor(max_workers=30) as pool:
            futures = {pool.submit(extract_generic, url, session): url for url in list(extra_links)[:200]}
            for fut in as_completed(futures):
                result = fut.result()
                if result and result['url'] not in seen_urls:
                    seen_urls.add(result['url'])
                    if is_disaster_article(result['title'], result['content']):
                        docs.append(result)
                        print(f"  [{len(docs)}] [+] {result['title'][:65]}")

    # 分类统计
    print(f"\n[统计]")
    disaster_stats = {}
    for doc in docs:
        text = doc['title'] + doc['content'][:300]
        found = False
        for dtype in ['地震', '台风', '洪水', '暴雨', '山体滑坡', '泥石流', '火灾', '干旱', '暴雪', '雪灾']:
            if dtype in text:
                disaster_stats[dtype] = disaster_stats.get(dtype, 0) + 1
                found = True
                break
        if not found:
            disaster_stats['其他'] = disaster_stats.get('其他', 0) + 1

    print(f"  成功: {len(docs)} 篇灾害新闻")
    print(f"  失败: {len(failed_urls)} 个URL")
    print(f"  类型分布: {dict(sorted(disaster_stats.items(), key=lambda x: -x[1]))}")

    # Save
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, 'homework3', 'data', 'documents')
    os.makedirs(out_dir, exist_ok=True)

    for doc in docs:
        path = os.path.join(out_dir, f"{doc['id']}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"\n[保存] → {out_dir}")
    print(f"[完成] {len(docs)} 篇真实灾害新闻已保存")


if __name__ == '__main__':
    main()
