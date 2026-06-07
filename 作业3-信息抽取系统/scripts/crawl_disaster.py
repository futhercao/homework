"""
灾害新闻专用爬虫 — 为作业三收集真实灾害事件报道

设计要点 (对应"爬虫质量 + 数据源多样性"评分点):
  1. 多入口召回:  百度新闻搜索(多关键词) + 多家主流媒体灾害/社会频道种子;
  2. 浅层 BFS 扩展: 在已确认的灾害页面上顺带发现更多文章链接, 自然提升数量与域名多样性;
  3. 双重过滤:     关键词初筛 is_disaster_article + 正则引擎 RegexExtractor 二次校验
                   (要求能识别出真实灾害类型, 复用 _is_real_disaster, 排除隐喻/列表式提及);
  4. 礼貌抓取:     UA 轮换 + 并发上限 + 单域名限流 + 请求间隔 (亦服务于可持续性章节);
  5. 真实多媒体:   下载文章真实配图 (按文件魔数判定 png/jpg), 供后续 EasyOCR 抽取;
  6. 断点友好:     逐篇落盘 dis_<md5>.json, 重跑同 URL 覆盖同名文件, 不产生重复。

用法: python scripts/crawl_disaster.py --max 200
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
from src.extraction.regex_engine import RegexExtractor

# 百度新闻搜索关键词（覆盖主要灾害类型）
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
    '暴雪 冰冻 灾害',
    '地质灾害 转移 安置',
]

# 种子URL — 灾害领域权威采编源(部委/应急/气象/地震/水利 + 国家通讯社/党媒/央媒)的频道与列表页。
# 由原门户聚合站(新浪/网易/搜狐/环球)升级而来: 信源权威、灾害现场报道集中、转载与标题党少。
# 配合下方 SOURCE_WHITELIST, 百度搜索/链接扩展带出的门户与自媒体一律不入库。
SEED_URLS = [
    # 部委/应急/气象/地震/水利 (灾害领域最权威源)
    'https://www.mem.gov.cn/xw/yjglbgzdt/',        # 应急管理部 工作动态
    'https://www.mem.gov.cn/xw/bndt/',             # 应急管理部 部内动态
    'https://www.cma.gov.cn/2011xwzx/2011xqxxw/',  # 中国气象局 气象新闻
    'https://www.119.gov.cn/',                     # 国家消防救援局
    'http://www.mwr.gov.cn/xw/',                   # 水利部 新闻
    # 注: 不种子化中国地震局/省级地震局站点 —— 其"地震速报"为震级/经纬度技术公报, 缺伤亡/响应/
    #     经济等事件要素, 易刷屏且抽取价值低; 地震事件改由通讯社/应急部的灾情报道获取(信息更完整)。
    # 国家通讯社 / 党媒 / 央媒 社会·国内频道 (灾害现场报道集中地)
    'http://www.news.cn/local/',                   # 新华网 地方
    'http://www.news.cn/politics/',                # 新华网 时政
    'http://society.people.com.cn/',               # 人民网 社会
    'https://news.cctv.com/society/',              # 央视网 社会
    'https://news.cctv.com/china/',                # 央视网 国内
    'https://www.chinanews.com.cn/society/',       # 中国新闻网 社会
    'https://www.chinanews.com.cn/scroll-news/news1.html',  # 中新网 滚动(链接海量)
    'https://www.chinanews.com.cn/gn/',            # 中新网 国内
    'https://politics.gmw.cn/',                    # 光明网 时政
    'https://www.cnr.cn/',                         # 央广网
    # 优质深度报道 (非门户; 灾害/社会调查强)
    'https://www.thepaper.cn/',                    # 澎湃新闻
]

# 权威源白名单 (子串匹配 netloc): 仅保存来自这些域名的文章, 滤掉百度搜索与链接扩展带出的
# 门户/自媒体(网易163/中华网/搜狐/新浪/腾讯/环球等), 确保灾害语料信源的权威性与可信度。
SOURCE_WHITELIST = (
    'gov.cn',                          # 各级政府/部委: mem(应急)/cma(气象)/119(消防)/mwr(水利)/地方 *.gov.cn
    'ceic.ac.cn', 'cea.gov.cn',        # 中国地震台网 / 地震局
    'news.cn', 'xinhuanet.com',        # 新华网 / 新华社
    'people.com.cn',                   # 人民网
    'cctv.com', 'cctv.cn', 'cntv.cn',  # 央视网
    'chinanews.com',                   # 中国新闻网
    'gmw.cn',                          # 光明网
    'cnr.cn',                          # 央广网
    'ce.cn',                           # 中国经济网
    'stdaily.com',                     # 科技日报
    'thepaper.cn',                     # 澎湃新闻 (优质深度报道)
)


def is_authoritative(url):
    """URL 是否来自权威源白名单 (按 netloc 子串匹配)"""
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(key in netloc for key in SOURCE_WHITELIST)

# 百度新闻URL模板
BAIDU_NEWS_URL = 'https://www.baidu.com/s?tn=news&rtt=1&wd={query}'

# 单源配额: 单一来源最多入库的文档数, 避免某站点(如地震局速报)刷屏, 强制来源/题材多样性。
SOURCE_CAP = 35

HEADERS_TEMPLATE = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1',
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'disaster_docs')
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'images')

# 像"文章"的URL特征 (日期路径 / .htm(l)/.shtml / /a/123 / /article/ / /news/ / 新华c_ / 光明content_ / 部委t20240615_)
NEWS_LINK_RE = re.compile(
    r'/\d{4}[-/]?\d{2}[-/]?\d{2}'      # 日期路径
    r'|\.s?html?(?:$|\?)'             # .htm/.html/.shtm/.shtml
    r'|/a/\d|/article/|/news/\w'      # 文章式路径
    r'|/c_\d+|/content_\d+|/t\d{8}'   # 新华c_/光明content_/部委 t20240615_
)


def is_disaster_article(title, content):
    """关键词初筛: 是否疑似灾害新闻"""
    text = (title + content[:1000]).lower()
    keywords = ['地震', '台风', '洪水', '暴雨', '滑坡', '泥石流', '火灾', '干旱',
                '暴雪', '冰雹', '海啸', '龙卷风', '遇难', '受灾', '救灾', '灾情',
                '抢险', '救援', '应急响应', '转移群众', '倒塌', '经济损失']
    return any(kw in text for kw in keywords)


# 机构/党务/导航/解读/活动/误分类 等"非灾害事件"标题信号 (命中即剔除)
_INSTITUTIONAL = [
    '党组', '党委', '党风', '廉政', '纪委', '纪检', '巡视', '巡察', '警示教育', '读书班',
    '学习贯彻', '理论学习', '中心组', '主题教育', '政绩观', '述职', '述廉', '离退休', '任免',
    '会见', '调研', '座谈', '交流会', '局长会议', '工作会议', '工作会召开', '视频会议',
    '专题视频会', '督导工作', '动员会',
    '网站地图', '站点地图', '信息公开', '公开指南', '公开内容', '年度报告', '年度报表',
    '建设报告', '工作报告', '简介', '概述', '监测中心站', '预警站网', '站网规划', '研究室',
    '发展战略', '处罚裁量', '行政处罚',
    '解读', '通告', '行业标准', '国家标准', '在线申请', '管理规定', '履职管理', '答复的函',
    '提案', '征求', '意见的函', '消防术语',
    '科普', '宣传活动', '知识培训', '培训走进', '周年', '博物馆', '大讲堂', '少年说', '复测',
    '志愿服务', '劳动模范', '先进工作者', '服务队', '公益事业', '安全教育活动', '展演',
    '守夜人', '点灯人', '保驾护航', '为群众办实事', '民生工程', '工作法', '安全屏障',
    '专项整治', '隐患自查', '智能体', '顺风耳', '高质量发展', '初心',
    '热带雨林', '珊瑚', '航母', '儿童剧', '电动自行车', '三伏天', '生态文明', '数字水网',
    '金融保险', '物价监测', '安全防范工作', '清明节前后',
]
# 具体"灾害事件/响应"信号 (标题须命中其一, 区别于"地震工作/防震减灾能力"等机构话术)
_EVENT_SIGNAL = re.compile(
    r'\d(?:\.\d)?\s*级|级地震|余震|震中|震群|烈度|震感'
    r'|应急响应|启动.{0,12}响应|响应.{0,4}(?:提升|启动|至|为)'
    r'|遇难|死亡|受伤|失踪|伤亡|被困|罹难'
    r'|受灾|灾区|抢险|搜救|转移群众|疏散|安置'
    r'|登陆|强降雨|暴雨|特大暴雨|大暴雨|降雨量|降水量|降雨|降水|雷暴'
    r'|防汛|汛期|主汛|龙舟水|山洪|洪峰|溃堤|决口|内涝|洪涝'
    r'|救灾资金|救灾物资|预拨|调拨|飓风|台风'
    r'|过火面积|森林火灾|森林草原火灾|草原火灾'
    r'|次生灾害|地质灾害|水旱灾害|自然灾害|经济损失|直接损失'
    r'|泥石流|滑坡|崩塌|旱情|暴雪|雪灾|冰雹')
_DISASTER_KW = ('地震', '台风', '洪水', '暴雨', '强降雨', '滑坡', '泥石流', '火灾', '干旱',
                '暴雪', '冰雹', '海啸', '龙卷风', '飓风', '遇难', '受灾', '救灾', '灾情',
                '汛期', '防汛', '山洪', '灾区', '震级', '余震', '登陆')


def is_disaster_event(title, content):
    """事件性过滤: 仅保留真实"灾害事件/响应"报道, 剔除机构党务/会议/解读/导航/人名/科普等。
    判据: (1)标题不含机构话术词; (2)标题非纯人名/超短导航; (3)标题须含具体灾害事件/响应信号。"""
    t = title or ''
    if any(k in t for k in _INSTITUTIONAL):
        return False
    cjk = sum(1 for ch in t if '一' <= ch <= '鿿')
    if cjk <= 4 and not any(k in t for k in _DISASTER_KW):
        return False
    if not _EVENT_SIGNAL.search(t):
        return False
    return True



def extract_article_urls_from_baidu(html):
    """从百度新闻搜索页面提取候选文章URL"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    urls = set()
    for a in soup.find_all('a', href=True):
        href = a.get('href', '')
        if href.startswith('http') and 'baidu.com' not in href and len(href) > 20:
            urls.add(href)
    return urls


def extract_news_links(html, base_url):
    """从任意页面抽取像文章的链接 (浅层BFS的扩展来源)"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        if not href or len(text) < 6:
            continue
        if any(href.startswith(p) for p in ('javascript:', '#', 'mailto:', 'tel:')):
            continue
        if not href.startswith('http'):
            href = urljoin(base_url, href)
        if not href.startswith('http'):
            continue
        if NEWS_LINK_RE.search(href):
            links.add(href.split('#')[0])
    return links


async def crawl_disaster_news(max_docs=200, max_concurrent=10):
    """主爬取函数 (两阶段: 召回候选URL → 并发抓取+浅层扩展)"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    dedup = ContentDeduplicator(threshold=3)
    parser = ArticleParser()
    regex = RegexExtractor()
    documents = []
    visited = set()
    all_article_urls = set()
    saved_per_source = {}   # 来源 → 已入库数, 用于单源配额

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

    async def fetch_bytes(url):
        try:
            async with session.get(url, headers={'User-Agent': random.choice(HEADERS_TEMPLATE)},
                                   allow_redirects=True, ssl=False) as resp:
                if resp.status == 200:
                    return await resp.read()
        except Exception:
            pass
        return None

    async def download_images(article):
        """下载文章真实配图; 按文件魔数判定扩展名"""
        local_images = []
        for img in article.get('images', [])[:5]:
            src = img.get('src', '')
            if not src.startswith('http'):
                continue
            img_bytes = await fetch_bytes(src)
            if not img_bytes or len(img_bytes) < 2000:
                continue
            ext = '.png' if img_bytes[:8].startswith(b'\x89PNG') else '.jpg'
            img_id = hashlib.md5(src.encode()).hexdigest()[:12]
            img_path = os.path.join(IMAGE_DIR, f'{img_id}{ext}')
            try:
                async with aiofiles.open(img_path, 'wb') as f:
                    await f.write(img_bytes)
                local_images.append({
                    # 存相对路径(data/images/<名>), 不写机器绝对路径, 保证整体可迁移;
                    # 读取端(web/_resolve_image、run_ocr)一律按 basename 在 IMAGE_DIR 解析。
                    'local_path': f'data/images/{img_id}{ext}',
                    'original_src': src,
                    'alt': img.get('alt', ''),
                })
            except Exception:
                pass
        return local_images

    async def collect_from_search(query):
        html = await fetch(BAIDU_NEWS_URL.format(query=query))
        return extract_article_urls_from_baidu(html) if html else set()

    async def collect_from_seed(seed_url):
        html = await fetch(seed_url)
        return extract_news_links(html, seed_url) if html else set()

    async def process_article(url):
        """抓取并处理单篇; 返回 (保存的文章 or None, 本页发现的新链接集合)"""
        if url in visited:
            return None, set()
        visited.add(url)
        # 权威源白名单: 百度搜索/链接扩展会带出大量门户与自媒体URL, 非权威源直接跳过(不抓不存),
        # 既保证灾害语料信源权威, 又省带宽; BFS 链接扩展因此只在权威站内进行。
        if not is_authoritative(url):
            return None, set()
        # 单源配额: 该来源已达上限则跳过(不抓不存), 把名额让给其他来源 → 来源/题材更均衡
        src_guess = ArticleParser._extract_domain(url)
        if saved_per_source.get(src_guess, 0) >= SOURCE_CAP:
            return None, set()
        async with sem:
            html = await fetch(url)
        if not html:
            return None, set()

        article = parser.parse(html, url)
        if not article:
            return None, set()

        # 关键词初筛
        if not is_disaster_article(article['title'], article['content']):
            return None, set()

        # 在灾害相关页面顺带发现更多文章链接 (浅层BFS)
        discovered = extract_news_links(html, url)

        # 正则二次校验: 必须识别出真实灾害类型 (复用 _is_real_disaster)
        full_text = f"{article['title']}\n{article['content']}"
        dtype = regex._extract_disaster_type(full_text)
        if not dtype or not regex._is_real_disaster(full_text, dtype):
            return None, discovered

        # 事件性过滤: 剔除机构党务/会议/解读/导航/人名/科普等"非灾害事件"内容
        # (标题须含具体灾害事件/响应信号; 权威部门站点首页多为工作动态, 此处保证语料为真事件)
        if not is_disaster_event(article['title'], full_text):
            return None, discovered

        doc_id = 'dis_' + hashlib.md5(url.encode()).hexdigest()[:12]
        if dedup.is_duplicate(article['content'], doc_id)[0]:
            return None, discovered

        article['id'] = doc_id
        article['crawl_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        article['category'] = '灾害'
        article['disaster_type'] = dtype
        article['local_images'] = await download_images(article)

        filepath = os.path.join(DATA_DIR, f'{doc_id}.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=2)
        saved_per_source[src_guess] = saved_per_source.get(src_guess, 0) + 1
        return article, discovered

    # ---------- Phase 1: 召回候选URL ----------
    print('[灾害爬虫] Phase 1: 收集候选文章URL...')
    for q in DISASTER_QUERIES:
        urls = await collect_from_search(q)
        all_article_urls.update(urls)
        print(f'  搜索 [{q}] → {len(urls)} URL')
        await asyncio.sleep(1)
    for seed in SEED_URLS:
        urls = await collect_from_seed(seed)
        all_article_urls.update(urls)
        print(f'  种子 [{seed}] → {len(urls)} URL')
    print(f'  共收集 {len(all_article_urls)} 个候选URL')

    # ---------- Phase 2: 并发抓取 + 浅层扩展 ----------
    print(f'[灾害爬虫] Phase 2: 并发抓取 + 浅层BFS扩展 ({max_concurrent}并发)...')
    pending = deque(all_article_urls)
    tasks = set()
    max_visits = max_docs * 40   # 访问上限, 防止扩展失控

    while (pending or tasks) and len(documents) < max_docs:
        while pending and len(tasks) < max_concurrent * 2 and len(visited) < max_visits:
            url = pending.popleft()
            if url not in visited:
                tasks.add(asyncio.create_task(process_article(url)))
        if not tasks:
            break
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            result, discovered = await task
            for u in discovered:
                if u not in visited:
                    pending.append(u)
            if result:
                documents.append(result)
                if len(documents) % 10 == 0:
                    ndom = len({d.get('source') for d in documents})
                    print(f'  [{len(documents)}/{max_docs}] 域名{ndom} | {result["title"][:50]}')

    await session.close()

    # ---------- 统计: 类型分布 + 域名多样性 ----------
    types, domains = {}, {}
    for d in documents:
        dt = d.get('disaster_type') or '其他'
        types[dt] = types.get(dt, 0) + 1
        dom = d.get('source', '?')
        domains[dom] = domains.get(dom, 0) + 1
    print(f'\n[完成] 本次新增 {len(documents)} 篇灾害新闻 (访问 {len(visited)} 个URL)')
    print(f'类型分布: {dict(sorted(types.items(), key=lambda x: -x[1]))}')
    print(f'域名多样性: {len(domains)} 个 → {dict(sorted(domains.items(), key=lambda x: -x[1]))}')

    # 以落盘 JSON 为准重建 SQLite 索引 (爬虫直接写 JSON, Web 端按索引列目录)
    try:
        from src.storage.doc_store import DocumentStore
        n_idx = DocumentStore(doc_dir=DATA_DIR).reindex_from_files()
        print(f'[索引] 已按 {n_idx} 篇 JSON 重建 metadata.db')
    except Exception as e:
        print(f'[索引] 重建失败(不影响语料, Web 端首次启动会自动重建): {e}')
    return documents


async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--max', type=int, default=200)
    ap.add_argument('--concurrent', type=int, default=10)
    args = ap.parse_args()
    await crawl_disaster_news(max_docs=args.max, max_concurrent=args.concurrent)


if __name__ == '__main__':
    asyncio.run(main())
