"""
全局配置系统
所有配置集中管理，支持YAML文件和命令行覆盖
"""
import os
import yaml

# === 路径配置 ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DOC_DIR = os.path.join(DATA_DIR, 'documents')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
INDEX_DIR = os.path.join(DATA_DIR, 'index')
# FAISS doesn't handle Unicode paths on Windows; use a known-safe path
import tempfile
_tmp_base = tempfile.gettempdir()
VECTOR_DIR = os.path.join(_tmp_base, 'ir_system_vectors')
EVAL_DIR = os.path.join(DATA_DIR, 'evaluation')

# === 爬虫配置 ===
CRAWLER_CONFIG = {
    # 并发控制
    'max_concurrent': 20,          # 异步最大并发
    'request_delay': (0.5, 2.0),   # 请求间隔范围(秒)
    'request_timeout': 20,         # 单请求超时
    'max_retries': 3,              # 失败重试次数

    # User-Agent池 (15个)
    'user_agents': [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (iPad; CPU OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 OPR/114.0.0.0',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    ],

    # 新闻种子URL (14个源，5个类别)
    'seed_urls': [
        # 科技类
        'https://www.ithome.com/',
        'https://tech.sina.com.cn/',
        'https://tech.163.com/',
        'https://36kr.com/newsflashes',
        # 社会类
        'https://news.sina.com.cn/society/',
        'https://www.thepaper.cn/',
        'https://society.people.com.cn/',
        # 综合类
        'https://www.163.com/',
        'https://news.qq.com/',
        'https://www.sohu.com/',
        # 财经类
        'https://finance.sina.com.cn/',
        'https://money.163.com/',
        # 教育/健康
        'https://edu.sina.com.cn/',
        'https://health.sina.com.cn/',
    ],

    # 分类规则 (URL模式匹配)
    'category_rules': {
        '科技': ['tech', 'ithome', '36kr', 'keji'],
        '社会': ['society', 'shehui', 'thepaper', 'people.com.cn'],
        '财经': ['finance', 'money', 'caijing', 'stock'],
        '教育': ['edu', 'learning'],
        '健康': ['health', 'jiankang'],
        '体育': ['sports', 'tiyu'],
        '军事': ['mil', 'junshi'],
        '娱乐': ['ent', 'yule'],
    },

    # 内容质量过滤
    'min_content_length': 100,
    'min_title_length': 5,
    'max_title_length': 200,
}

# === NLP配置 ===
NLP_CONFIG = {
    'language': 'zh',
    'jieba_user_dict': None,
    'remove_stopwords': True,
    'min_token_length': 1,
    'enable_pos': True,
    'enable_ner': True,
}

# === 检索配置 ===
RETRIEVAL_CONFIG = {
    'default_algorithm': 'hybrid',   # bm25 | vsm | semantic | hybrid
    'top_k': 20,
    'bm25_k1': 1.5,
    'bm25_b': 0.75,
    'snippet_length': 250,
    'semantic_weight': 0.4,        # 语义检索在混合中的权重
    'bm25_weight': 0.6,            # BM25在混合中的权重
    'query_expansion_terms': 3,
}

# === 多模态配置 ===
MULTIMODAL_CONFIG = {
    'image_dir': IMAGE_DIR,
    'max_image_size': (800, 800),
    'clip_model': 'ViT-B-32',
    'ocr_lang': 'ch',
    'image_index_dim': 512,
    'max_images_per_doc': 10,
}

# === 评价配置 ===
EVAL_CONFIG = {
    'precision_recall_k': [5, 10, 20],
    'relevance_levels': {0: '不相关', 1: '部分相关', 2: '非常相关'},
}

# === Web配置 ===
WEB_CONFIG = {
    'host': '127.0.0.1',
    'port': 8090,
    'title': '信息检索与抽取平台',
}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
