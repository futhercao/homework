"""信息检索系统 - 全局配置"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'documents')
INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index')
EVAL_DIR = os.path.join(BASE_DIR, 'data', 'evaluation')

CRAWLER_CONFIG = {
    'seed_urls': [
        'https://news.sina.com.cn/technology/',
    ],
    'max_documents': 150,
    'request_delay': 1.0,
    'request_timeout': 15,
    'user_agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
}

PREPROCESS_CONFIG = {
    'language': 'zh',
    'min_term_length': 1,
    'remove_stopwords': True,
}

RETRIEVAL_CONFIG = {
    'default_algorithm': 'bm25',
    'top_k': 20,
    'bm25_k1': 1.5,
    'bm25_b': 0.75,
    'snippet_length': 200,
}

FLASK_CONFIG = {
    'host': '127.0.0.1',
    'port': 5000,
    'debug': True,
}
