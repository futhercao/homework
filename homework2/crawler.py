"""
网络爬虫模块
- WebCrawler: 通用网页爬虫，支持从任意新闻网站爬取文章
- SampleDataGenerator: 样本数据生成器，用于生成演示用中文科技新闻数据
"""
import os
import json
import time
import uuid
import re
import random
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import DATA_DIR, CRAWLER_CONFIG


class WebCrawler:
    """通用网页爬虫，从指定网站爬取文章并本地存储"""

    def __init__(self, config=None):
        self.config = config or CRAWLER_CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config['user_agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
        })
        self.documents = []
        self.visited = set()

    def crawl(self, seed_urls=None, max_docs=None):
        """从种子URL开始爬取文档"""
        seed_urls = seed_urls or self.config['seed_urls']
        max_docs = max_docs or self.config['max_documents']
        queue = list(seed_urls)

        print(f"[爬虫] 开始爬取，种子URL: {len(seed_urls)} 个，目标: {max_docs} 篇文档")

        while queue and len(self.documents) < max_docs:
            url = queue.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)

            try:
                html = self._fetch(url)
                if not html:
                    continue

                article = self._extract_article(html, url)
                if article and len(article['content']) >= 80:
                    self.documents.append(article)
                    print(f"  [{len(self.documents)}/{max_docs}] {article['title'][:40]}")

                links = self._extract_links(html, url)
                for link in links:
                    if link not in self.visited:
                        queue.append(link)

                time.sleep(self.config.get('request_delay', 1))
            except Exception as e:
                print(f"  [错误] {url}: {e}")

        self._save_documents()
        print(f"[爬虫] 完成，共爬取 {len(self.documents)} 篇文档")
        return self.documents

    def _fetch(self, url):
        try:
            resp = self.session.get(
                url, timeout=self.config.get('request_timeout', 15)
            )
            resp.encoding = resp.apparent_encoding or 'utf-8'
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            print(f"  [请求失败] {e}")
        return None

    def _extract_article(self, html, url):
        soup = BeautifulSoup(html, 'lxml')
        for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'aside', 'iframe']):
            tag.decompose()

        title = self._get_title(soup)
        content = self._get_content(soup)
        if not title or not content:
            return None

        return {
            'id': uuid.uuid4().hex[:8],
            'title': title.strip(),
            'content': content.strip(),
            'url': url,
            'date': self._get_date(soup, html),
            'images': self._get_images(soup, url),
            'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def _get_title(self, soup):
        for sel in ['h1', '.title', '.article-title', '.post-title']:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                return el.get_text(strip=True)
        t = soup.find('title')
        return t.get_text(strip=True).split('-')[0].strip() if t else None

    def _get_content(self, soup):
        selectors = [
            'article', '.article-content', '.article_content',
            '.post-content', '.entry-content', '.content',
            '.art_content', '.article-body', '#artibody',
            '.main-content', '.text', '#content',
        ]
        for sel in selectors:
            el = soup.select_one(sel)
            if el and len(el.get_text(strip=True)) >= 80:
                return el.get_text(separator='\n', strip=True)

        divs = soup.find_all(['div', 'section', 'main'])
        if divs:
            best = max(divs, key=lambda d: len(d.get_text(strip=True)))
            text = best.get_text(separator='\n', strip=True)
            if len(text) >= 80:
                return text
        return None

    def _get_date(self, soup, html):
        for sel in ['.date', '.time', '.pub-date', '.publish-time', 'time', '.article-time']:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                return el.get_text(strip=True)[:20]
        for pat in [r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', r'\d{4}年\d{1,2}月\d{1,2}日']:
            m = re.search(pat, html[:5000])
            if m:
                return m.group()
        return datetime.now().strftime('%Y-%m-%d')

    def _get_images(self, soup, base_url):
        images = []
        for img in soup.find_all('img', src=True)[:5]:
            src = img.get('src', '')
            alt = img.get('alt', '')
            if src and not src.endswith(('.gif', '.svg', '.ico')):
                if not src.startswith('http'):
                    src = urljoin(base_url, src)
                images.append({'src': src, 'alt': alt})
        return images

    def _extract_links(self, html, base_url):
        soup = BeautifulSoup(html, 'lxml')
        links = []
        domain = urlparse(base_url).netloc
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith(('javascript', '#', 'mailto')):
                continue
            if not href.startswith('http'):
                href = urljoin(base_url, href)
            if urlparse(href).netloc == domain and href not in self.visited:
                links.append(href)
        return links[:30]

    def _save_documents(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        for doc in self.documents:
            path = os.path.join(DATA_DIR, f"{doc['id']}.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
        manifest = {
            'total': len(self.documents),
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'docs': [{'id': d['id'], 'title': d['title']} for d in self.documents],
        }
        with open(os.path.join(DATA_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[爬虫] 已保存 {len(self.documents)} 篇文档到 {DATA_DIR}")


class SampleDataGenerator:
    """
    样本数据生成器：生成中文科技新闻文档用于系统演示与测试。
    涵盖20个科技领域，每个领域生成多篇文档，总计150+篇。
    """

    TOPICS = [
        {
            'name': '人工智能',
            'subs': ['深度学习', '机器学习', '强化学习', '迁移学习', '联邦学习', '大语言模型', '知识图谱', '多模态AI'],
            'terms': ['神经网络', '算法优化', '模型训练', '数据集', '推理加速', '参数调优', '梯度下降', '注意力机制'],
            'apps': ['医疗诊断', '智能客服', '金融风控', '教育个性化', '智能制造', '内容生成'],
        },
        {
            'name': '自然语言处理',
            'subs': ['文本分类', '情感分析', '机器翻译', '问答系统', '文本生成', '命名实体识别', '信息抽取'],
            'terms': ['词嵌入', 'Transformer', '预训练模型', '语义理解', '序列标注', '文本表示', '分词算法'],
            'apps': ['搜索引擎', '舆情监控', '智能写作', '跨语言交流', '法律文书分析', '智能翻译'],
        },
        {
            'name': '计算机视觉',
            'subs': ['目标检测', '图像分割', '人脸识别', '图像生成', '视频分析', '三维重建', 'OCR识别'],
            'terms': ['卷积神经网络', '特征提取', '目标跟踪', '语义分割', '图像增强', '姿态估计'],
            'apps': ['安防监控', '自动驾驶', '医学影像', '工业检测', '增强现实', '无人机'],
        },
        {
            'name': '大数据技术',
            'subs': ['分布式计算', '数据挖掘', '流式处理', '数据仓库', '数据可视化', '图计算', '数据治理'],
            'terms': ['Hadoop', 'Spark', '数据清洗', '特征工程', '实时分析', '数据湖', '批处理'],
            'apps': ['用户画像', '推荐系统', '精准营销', '城市管理', '交通优化', '供应链管理'],
        },
        {
            'name': '云计算',
            'subs': ['容器技术', '微服务架构', '无服务器计算', '混合云', '边缘计算', '云原生', '云安全'],
            'terms': ['虚拟化', '弹性扩展', '负载均衡', '资源调度', '服务网格', '持续集成'],
            'apps': ['企业数字化', '在线教育', '远程办公', '电商平台', '游戏云化', '政务云'],
        },
        {
            'name': '物联网',
            'subs': ['传感器网络', '智能家居', '工业物联网', '车联网', '可穿戴设备', '智慧城市', '智能农业'],
            'terms': ['嵌入式系统', 'MQTT协议', '低功耗通信', '边缘网关', '数据采集', '设备管理'],
            'apps': ['环境监测', '健康管理', '能源管理', '智慧交通', '智能仓储', '精准灌溉'],
        },
        {
            'name': '5G与6G通信',
            'subs': ['毫米波技术', '网络切片', '大规模MIMO', '太赫兹通信', '卫星互联网', '通感一体化'],
            'terms': ['频谱效率', '低延迟', '高带宽', '基站部署', '波束赋形', '信道编码'],
            'apps': ['远程手术', '工业自动化', '全息通信', '虚拟现实', '智能交通', '无人机通信'],
        },
        {
            'name': '网络安全',
            'subs': ['入侵检测', '数据加密', '隐私保护', '零信任架构', '安全审计', '恶意软件分析', '漏洞挖掘'],
            'terms': ['防火墙', '身份认证', '密码学', '安全协议', '风险评估', '态势感知'],
            'apps': ['金融安全', '个人隐私', '企业防护', '国家安全', '电子政务', '供应链安全'],
        },
        {
            'name': '区块链',
            'subs': ['智能合约', '共识机制', '跨链技术', '去中心化金融', '数字身份', 'NFT技术', '隐私链'],
            'terms': ['哈希算法', '分布式账本', '节点共识', '链上治理', '侧链', '去中心化应用'],
            'apps': ['供应链溯源', '数字货币', '版权保护', '电子存证', '慈善透明', '跨境支付'],
        },
        {
            'name': '量子计算',
            'subs': ['量子比特', '量子纠错', '量子算法', '量子模拟', '量子密码', '拓扑量子计算'],
            'terms': ['叠加态', '量子纠缠', '退相干', '量子门', '量子优势', '量子退火'],
            'apps': ['药物研发', '材料模拟', '密码破译', '金融建模', '气候预测', '优化问题'],
        },
        {
            'name': '自动驾驶',
            'subs': ['感知系统', '路径规划', '决策控制', '高精地图', 'V2X通信', '仿真测试', '传感器融合'],
            'terms': ['激光雷达', '视觉感知', '定位导航', '行为预测', '安全冗余', '域控制器'],
            'apps': ['出租车', '物流配送', '矿区运输', '港口作业', '城市公交', '高速公路'],
        },
        {
            'name': '机器人技术',
            'subs': ['工业机器人', '服务机器人', '协作机器人', '软体机器人', '仿生机器人', '手术机器人'],
            'terms': ['运动控制', '力矩反馈', '路径规划', '人机交互', '自主导航', '抓取操作'],
            'apps': ['制造装配', '仓储物流', '家庭服务', '医疗护理', '深海探测', '太空探索'],
        },
        {
            'name': '新能源技术',
            'subs': ['太阳能', '风力发电', '氢能源', '储能技术', '核聚变', '地热能', '生物质能'],
            'terms': ['光伏效率', '电池技术', '并网发电', '能量密度', '碳中和', '清洁能源'],
            'apps': ['电动汽车', '智能电网', '建筑节能', '绿色数据中心', '碳交易', '可持续发展'],
        },
        {
            'name': '芯片与半导体',
            'subs': ['先进制程', '封装技术', 'AI芯片', 'RISC-V架构', '存算一体', 'EDA工具', '光芯片'],
            'terms': ['晶体管', '光刻技术', '良率优化', '功耗控制', '制造工艺', '设计自动化'],
            'apps': ['手机处理器', '数据中心', '自动驾驶', '物联网终端', '高性能计算', '航天电子'],
        },
        {
            'name': '虚拟现实与增强现实',
            'subs': ['VR头显', 'AR眼镜', '混合现实', '全息显示', '触觉反馈', '空间计算', '数字孪生'],
            'terms': ['渲染技术', '追踪定位', '交互设计', '显示分辨率', '延迟优化', '三维建模'],
            'apps': ['游戏娱乐', '教育培训', '工业设计', '远程协作', '房地产', '旅游体验'],
        },
        {
            'name': '生物信息学',
            'subs': ['基因组学', '蛋白质折叠', '药物发现', '精准医疗', '基因编辑', '单细胞测序'],
            'terms': ['序列比对', '结构预测', '分子动力学', '高通量筛选', '生物标志物', '基因调控'],
            'apps': ['癌症治疗', '遗传病诊断', '疫苗研发', '农作物改良', '微生物组', '衰老研究'],
        },
        {
            'name': '智能制造',
            'subs': ['工业互联网', '数字孪生', '预测性维护', '柔性制造', '质量检测', '供应链智能化'],
            'terms': ['PLC控制', 'SCADA系统', 'MES系统', '工艺优化', '产能调度', '设备互联'],
            'apps': ['汽车制造', '电子组装', '航空航天', '食品加工', '纺织服装', '石油化工'],
        },
        {
            'name': '数据库技术',
            'subs': ['分布式数据库', '图数据库', '时序数据库', '内存数据库', '向量数据库', 'NewSQL'],
            'terms': ['事务处理', '查询优化', '索引结构', '数据一致性', '高可用', '分区容错'],
            'apps': ['电商交易', '社交网络', '物联网存储', '金融系统', '日志分析', 'AI检索'],
        },
        {
            'name': '信息检索',
            'subs': ['搜索引擎', '推荐系统', '问答系统', '知识图谱检索', '多模态检索', '跨语言检索'],
            'terms': ['倒排索引', 'TF-IDF', 'BM25', '向量检索', '相关性排序', '查询扩展'],
            'apps': ['网页搜索', '学术搜索', '电商搜索', '企业知识库', '法律检索', '专利检索'],
        },
        {
            'name': '可持续发展与绿色科技',
            'subs': ['碳中和', '绿色计算', '循环经济', '环境监测', '节能减排', '生态保护', '可持续农业'],
            'terms': ['碳足迹', '能效优化', '绿色设计', '环保材料', '污染治理', '资源回收'],
            'apps': ['智慧环保', '绿色建筑', '清洁生产', '碳交易平台', '水资源管理', '生态修复'],
        },
    ]

    TEMPLATES = [
        (
            '{sub}：{name}领域的前沿方向',
            '近年来，{sub}作为{name}领域的重要研究方向，受到了学术界和产业界的广泛关注。'
            '研究人员通过改进{t1}和{t2}等关键技术，在{app1}、{app2}等应用场景中取得了显著突破。'
            '据统计，过去一年中相关论文发表量增长了{pct}%，多家知名企业加大了在该领域的研发投入。'
            '专家指出，{sub}技术的成熟将极大推动{name}的产业化进程，'
            '预计未来三到五年内，该技术将在{app3}等领域实现大规模商用。'
            '同时，需要关注技术发展带来的伦理、隐私和安全挑战。'
        ),
        (
            '{name}新突破：{sub}技术取得重大进展',
            '在最新发表的研究中，科学家团队在{sub}领域取得了重要突破。'
            '该研究利用{t1}结合{t2}的创新方法，将{app1}任务的性能提升了{pct}%。'
            '这一成果被认为是{name}发展历程中的里程碑事件。'
            '该团队负责人表示，新技术的核心在于对{t3}的优化，'
            '使系统能够更高效地处理复杂场景下的{app2}问题。'
            '业内人士预计，这一突破将加速{sub}技术在{app3}领域的落地应用。'
            '值得注意的是，该技术在节能方面也表现出色，符合绿色可持续发展的趋势。'
        ),
        (
            '{sub}产业报告：市场规模与发展趋势',
            '最新发布的行业报告显示，{name}领域中{sub}市场规模持续扩大。'
            '2025年全球{sub}市场规模达到{val}亿美元，同比增长{pct}%。'
            '报告指出，{t1}和{t2}是驱动市场增长的关键技术因素。'
            '在应用层面，{app1}和{app2}是当前最主要的商业化方向，'
            '预计到2028年，{app3}将成为新的增长点。'
            '中国在{sub}领域的专利申请量位居全球前列，展现出强劲的创新能力。'
            '然而，人才短缺和核心技术受制于人仍是行业面临的主要挑战。'
        ),
        (
            '深度解析：{sub}如何改变{app1}行业',
            '{sub}技术正在深刻改变{app1}行业的运作方式。'
            '通过引入{t1}和{t2}技术，企业能够显著提升运营效率和用户体验。'
            '以某头部企业为例，在部署{sub}解决方案后，其{app1}业务的处理效率提升了{pct}%，'
            '成本降低了约30%。{name}专家认为，{sub}与{app2}的结合'
            '将催生全新的商业模式。技术层面，{t3}的进步为这一变革提供了坚实基础。'
            '未来，随着技术不断成熟，{sub}将在{app3}等更多领域发挥变革性作用。'
        ),
        (
            '{name}前沿：{sub}研究综述与展望',
            '本文对{sub}领域近三年的研究进展进行了系统综述。'
            '{sub}是{name}中的核心课题之一，涉及{t1}、{t2}和{t3}等关键技术。'
            '在{app1}应用中，最新方法已经将准确率提升至{acc}%，相比传统方法提升显著。'
            '然而，当前研究仍面临数据质量不稳定、模型可解释性不足等挑战。'
            '展望未来，{sub}与{app2}的深度融合将是重要趋势，'
            '同时需要在技术发展中兼顾社会伦理和可持续发展。'
            '研究社区应加强跨学科合作，推动{sub}技术的健康发展。'
        ),
        (
            '国内外{sub}发展对比分析',
            '在{name}领域，{sub}技术的国内外发展呈现出不同特点。'
            '欧美国家在{t1}方面的基础研究较为领先，拥有更多的原创性成果。'
            '而中国在{t2}和应用落地方面展现出独特优势，特别是在{app1}和{app2}领域。'
            '数据显示，中国{sub}相关企业数量在过去三年增长了{pct}%。'
            '在技术标准方面，各国都在积极争夺{sub}领域的话语权。'
            '专家建议，应加强国际合作与交流，共同应对{sub}技术带来的全球性挑战，'
            '推动技术向着有利于环境保护和社会可持续发展的方向演进。'
        ),
        (
            '{sub}技术在{app1}中的创新应用',
            '随着{name}技术的快速发展，{sub}在{app1}领域展现出广阔的应用前景。'
            '最新的技术方案将{t1}与{t2}相结合，实现了{app1}流程的智能化改造。'
            '实践表明，采用{sub}技术后，{app1}系统的效率提升了{pct}%，'
            '同时大幅减少了人工干预的需求。在{app2}方面，{sub}同样展现出强大潜力。'
            '值得一提的是，这些技术方案在设计时充分考虑了节能环保因素，'
            '相比传统方案减少了约40%的能源消耗，体现了绿色发展理念。'
            '未来，{sub}技术有望在{app3}领域催生更多创新应用。'
        ),
    ]

    @classmethod
    def generate(cls, num_docs=150, save=True):
        """生成样本文档"""
        documents = []
        doc_id = 0
        random.seed(42)

        base_date = datetime(2025, 1, 1)

        while len(documents) < num_docs:
            for topic in cls.TOPICS:
                if len(documents) >= num_docs:
                    break
                for sub in topic['subs']:
                    if len(documents) >= num_docs:
                        break

                    tmpl_title, tmpl_content = random.choice(cls.TEMPLATES)
                    terms = random.sample(topic['terms'], min(3, len(topic['terms'])))
                    apps = random.sample(topic['apps'], min(3, len(topic['apps'])))

                    title = tmpl_title.format(
                        name=topic['name'], sub=sub,
                        app1=apps[0] if apps else '',
                    )
                    content = tmpl_content.format(
                        name=topic['name'], sub=sub,
                        t1=terms[0], t2=terms[1] if len(terms) > 1 else terms[0],
                        t3=terms[2] if len(terms) > 2 else terms[0],
                        app1=apps[0], app2=apps[1] if len(apps) > 1 else apps[0],
                        app3=apps[2] if len(apps) > 2 else apps[0],
                        pct=random.randint(15, 65),
                        val=random.randint(50, 800),
                        acc=random.randint(88, 99),
                    )

                    date = base_date + timedelta(days=random.randint(0, 400))
                    doc = {
                        'id': f'doc_{doc_id:04d}',
                        'title': title,
                        'content': content,
                        'url': f'https://tech.example.com/article/{doc_id:04d}',
                        'date': date.strftime('%Y-%m-%d'),
                        'images': cls._gen_images(topic['name'], sub, doc_id),
                        'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    documents.append(doc)
                    doc_id += 1

        if save:
            os.makedirs(DATA_DIR, exist_ok=True)
            for doc in documents:
                path = os.path.join(DATA_DIR, f"{doc['id']}.json")
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
            manifest = {
                'total': len(documents),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'sample_generator',
                'docs': [{'id': d['id'], 'title': d['title']} for d in documents],
            }
            with open(os.path.join(DATA_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print(f"[生成器] 已生成 {len(documents)} 篇样本文档到 {DATA_DIR}")

        return documents

    @classmethod
    def _gen_images(cls, topic, sub, doc_id):
        if random.random() < 0.4:
            return [
                {
                    'src': f'https://picsum.photos/seed/{doc_id}/400/300',
                    'alt': f'{topic}-{sub}示意图',
                }
            ]
        return []


def load_documents():
    """从本地存储加载所有文档"""
    documents = []
    if not os.path.exists(DATA_DIR):
        return documents
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.json') and fname != 'manifest.json':
            path = os.path.join(DATA_DIR, fname)
            with open(path, 'r', encoding='utf-8') as f:
                documents.append(json.load(f))
    documents.sort(key=lambda d: d.get('id', ''))
    return documents


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--crawl':
        urls = sys.argv[2:] if len(sys.argv) > 2 else None
        crawler = WebCrawler()
        crawler.crawl(seed_urls=urls)
    else:
        SampleDataGenerator.generate(150)
