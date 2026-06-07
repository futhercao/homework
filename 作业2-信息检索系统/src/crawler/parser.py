"""
智能HTML解析器
- readability-lxml 提取正文（优先）
- 自定义选择器回退（多站点兼容）
- 元数据提取（标题/时间/作者/来源）
- 图片链接提取
"""
import re
from datetime import datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from readability import Document as ReadabilityDoc


class ArticleParser:
    """通用新闻文章解析器"""

    # 正文选择器（按优先级）
    CONTENT_SELECTORS = [
        'article', '.article-content', '.article-body', '.post-content',
        '.post_body', '#article', '.article', '.content', '#content',
        '.main-content', '.news-content', '.entry-content', '.post-text',
        '#endText', '.text-content', '.detail-content', '.art_content',
        '.con_text', '#Cnt-Main-Article-QQ', '.TRS_Editor',
        '.show_text', '.news_txt', '.content-article', '.article-con',
        '.rich_media_content', '.js_content', '#js_content',
    ]

    # 标题选择器
    TITLE_SELECTORS = [
        'h1', '.article-title', '.post-title', '.news-title',
        '.main-title', '.title', '#title', '.art_title', '.con_title',
        'h2.title', '.text-title', '.content-title', '.entry-title',
    ]

    # 时间选择器
    DATE_SELECTORS = [
        '.date', '.time', '.pub-date', '.publish-time', 'time',
        '.article-time', '.post_time', '.info .time', '.time-source',
        '.article-info .time', '.news-time',
    ]

    @classmethod
    def parse(cls, html, url, source=''):
        """解析HTML提取文章"""
        if not html or len(html) < 500:
            return None

        soup = BeautifulSoup(html, 'lxml')

        # 清除干扰元素
        for tag in soup.find_all(['script', 'style', 'nav', 'footer',
                                   'aside', 'iframe', 'noscript', 'header']):
            tag.decompose()

        # 1. 尝试 readability-lxml
        title = content = None
        try:
            rd = ReadabilityDoc(html)
            title = rd.title()
            content_html = rd.summary()
            if content_html:
                content_soup = BeautifulSoup(content_html, 'lxml')
                content = content_soup.get_text(separator='\n', strip=True)
        except Exception:
            pass

        # 2. 自定义提取回退
        if not title or len(title) < 5:
            title = cls._extract_title(soup)

        if not content or len(content) < cls.MIN_LEN:
            fb = cls._extract_content(soup)
            if fb and len(fb) > len(content or ''):   # 仅当回退提取到更完整正文时才替换
                content = fb

        if not content or len(content) < cls.MIN_LEN:
            return None

        # 清理
        title = cls._clean_text(title)[:200]
        content = cls._clean_lines(content)

        # 质量闸: 丢弃残页/占位/错误页(正文中文过少, 或含错误页文案且正文偏短)
        if cls._looks_like_junk(title, content):
            return None

        date_str = cls._extract_date(soup, html, url)
        images = cls._extract_images(soup, url)
        source_domain = cls._extract_domain(url)
        category = cls._classify(url, title, content)

        return {
            'title': title,
            'content': content,
            'url': url,
            'date': date_str,
            'images': images,
            'source': source or source_domain,
            'category': category,
            'content_length': len(content),
            'image_count': len(images),
        }

    MIN_LEN = 250          # 正文最短字符数(质量闸; 由门户语料的 100 抬高, 滤快讯/导语残页)
    MAX_LEN = 100000

    @classmethod
    def _extract_title(cls, soup):
        for sel in cls.TITLE_SELECTORS:
            el = soup.select_one(sel)
            if el:
                t = el.get_text(strip=True)
                if 5 <= len(t) <= 200:
                    return t
        t = soup.find('title')
        if t:
            title_text = t.get_text(strip=True)
            for sep in ['_', '-', '|', '–']:
                if sep in title_text:
                    parts = [p.strip() for p in title_text.split(sep) if len(p.strip()) >= 5]
                    if parts:
                        return max(parts, key=len)
            return title_text
        return ''

    @classmethod
    def _extract_content(cls, soup):
        for sel in cls.CONTENT_SELECTORS:
            el = soup.select_one(sel)
            if el:
                for t in el.find_all(['script', 'style']):
                    t.decompose()
                text = el.get_text(separator='\n', strip=True)
                if len(text) >= cls.MIN_LEN:
                    return text[:cls.MAX_LEN]

        # Fallback: 找最长文本块
        candidates = []
        for tag_name in ['div', 'section', 'main', 'article']:
            for el in soup.find_all(tag_name):
                text = el.get_text(separator='\n', strip=True)
                if cls.MIN_LEN < len(text) < cls.MAX_LEN:
                    link_density = len(el.find_all('a')) / max(len(text), 1)
                    if link_density < 0.3:  # 过滤导航链接
                        candidates.append((len(text) - link_density * 1000, text))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            idx = min(1, len(candidates) - 1)
            return candidates[idx][1]
        return ''

    @classmethod
    def _extract_date(cls, soup, html, url):
        # 选择器提取
        for sel in cls.DATE_SELECTORS:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(strip=True)
                d = cls._parse_date(text)
                if d:
                    return d

        # 正则匹配
        patterns = [
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{4}/\d{2}/\d{2})',
            r'(\d{1,2}月\d{1,2}日)',
        ]
        snippet = html[:15000]
        for pat in patterns:
            m = re.search(pat, snippet)
            if m:
                d = cls._parse_date(m.group(1))
                if d:
                    return d

        # URL中提取
        url_m = re.search(r'/(\d{4})[-/](\d{2})[-/](\d{2})', url)
        if url_m:
            return f'{url_m.group(1)}-{url_m.group(2)}-{url_m.group(3)}'

        return datetime.now().strftime('%Y-%m-%d')

    @classmethod
    def _parse_date(cls, text):
        patterns = [
            (r'(\d{4})年(\d{1,2})月(\d{1,2})日', 1),
            (r'(\d{4})-(\d{2})-(\d{2})', 2),
            (r'(\d{4})/(\d{2})/(\d{2})', 3),
        ]
        for pat, style in patterns:
            m = re.search(pat, text)
            if m:
                y, mo, d = m.group(1), m.group(2), m.group(3)
                try:
                    return f'{int(y):04d}-{int(mo):02d}-{int(d):02d}'
                except ValueError:
                    pass
        return None

    @classmethod
    def _extract_images(cls, soup, base_url):
        images = []
        seen = set()
        for img in soup.find_all('img', src=True)[:10]:
            src = img.get('src', '')
            if not src or src in seen:
                continue
            if any(src.endswith(x) for x in ['.gif', '.svg', '.ico', 'data:']):
                continue
            if not src.startswith('http'):
                src = urljoin(base_url, src)
            if src.startswith('http'):
                seen.add(src)
                images.append({
                    'src': src,
                    'alt': img.get('alt', '')[:200],
                    'width': img.get('width', ''),
                    'height': img.get('height', ''),
                })
        return images

    # 域名 → 正式媒体名: 权威源给出可读来源名, 而非 netloc 首段(如 "society"/"finance")。
    # 按子串匹配, 故 chinanews.com 同时覆盖 chinanews.com.cn 与 www.chinanews.com。
    DOMAIN_SOURCE_MAP = [
        ('xinhuanet.com', '新华网'), ('news.cn', '新华网'),
        ('people.com.cn', '人民网'),
        ('cctv.com', '央视网'), ('cctv.cn', '央视网'), ('cnr.cn', '央广网'),
        ('chinanews.com', '中国新闻网'),
        ('gmw.cn', '光明网'),
        ('ce.cn', '中国经济网'),
        ('stdaily.com', '科技日报'),
        ('thepaper.cn', '澎湃新闻'),
        ('jiemian.com', '界面新闻'),
    ]

    @classmethod
    def _extract_domain(cls, url):
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc.lower()
        for key, name in cls.DOMAIN_SOURCE_MAP:
            if key in netloc:
                return name
        domain = re.sub(r'^www\.', '', netloc)
        return domain.split('.')[0] if '.' in domain else domain

    @classmethod
    def _classify(cls, url, title, content):
        text = url + title + content[:500]
        patterns = {
            '科技': ['科技', 'AI', '人工智能', '芯片', '5G', '手机', '互联网', '软件', '算法'],
            '社会': ['社会', '民生', '事故', '安全', '社区', '居民'],
            '财经': ['财经', '股票', '基金', '经济', '投资', '金融', 'A股'],
            '教育': ['教育', '学校', '大学', '考试', '招生', '学生'],
            '健康': ['健康', '医疗', '医院', '疾病', '医保', '药品'],
            '体育': ['体育', '比赛', '足球', '篮球', '奥运', '冠军'],
            '军事': ['军事', '军队', '武器', '国防', '演习'],
            '娱乐': ['娱乐', '电影', '综艺', '明星', '音乐'],
            '灾害': ['地震', '台风', '洪水', '暴雨', '滑坡', '泥石流', '火灾', '遇难'],
        }
        scores = {}
        for cat, keywords in patterns.items():
            scores[cat] = sum(1 for kw in keywords if kw in text)
        if scores:
            best = max(scores, key=scores.get)
            return best if scores[best] >= 2 else '综合'
        return '综合'

    @classmethod
    def _clean_text(cls, text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @classmethod
    def _clean_lines(cls, text):
        lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 2]
        return '\n'.join(lines)

    # 错误页/占位页/JS墙 文案: 命中且正文偏短时判为非真实新闻(澎湃/界面等 JS 渲染失败常见)
    _JUNK_MARKERS = ('请升级浏览器', '您的浏览器版本', '下载客户端', '页面不存在', '页面找不到',
                     '该页面不存在', '内容已删除', '请开启javascript', 'enable javascript',
                     '正在加载', 'page not found', '404 not found')

    @classmethod
    def _looks_like_junk(cls, title, content):
        """保守的垃圾页判定: 正文中文字数过少, 或(正文偏短 且 含错误页/占位文案)。
        300+ 中文字的正常长文永不误杀。"""
        text = content or ''
        cjk = sum(1 for ch in text if '一' <= ch <= '鿿')
        if cjk < 120:
            return True
        if cjk < 300:
            low = (title + ' ' + text).lower()
            if any(m.lower() in low for m in cls._JUNK_MARKERS):
                return True
        return False
