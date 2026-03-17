"""多模态自然灾害事件信息抽取系统 - 全局配置"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'documents')
IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'images')
EVAL_DIR = os.path.join(BASE_DIR, 'data', 'evaluation')
KG_DIR = os.path.join(BASE_DIR, 'data', 'knowledge_graph')

CRAWLER_CONFIG = {
    'seed_urls': [
        'https://news.sina.com.cn/society/',
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

EXTRACTION_CONFIG = {
    'info_points': [
        'disaster_type',   # 灾害类型
        'event_time',      # 发生时间
        'event_location',  # 发生地点
        'casualties',      # 伤亡情况
        'economic_loss',   # 经济损失
        'response_level',  # 响应级别
        'rescue_org',      # 救援组织
    ],
    'info_point_names': {
        'disaster_type': '灾害类型',
        'event_time': '发生时间',
        'event_location': '发生地点',
        'casualties': '伤亡情况',
        'economic_loss': '经济损失',
        'response_level': '响应级别',
        'rescue_org': '救援组织',
    },
    'methods': ['regex', 'ner', 'dependency'],
    'default_method': 'ensemble',
}

IMAGE_CONFIG = {
    'ocr_backend': 'auto',      # auto / easyocr / pytesseract / fallback
    'generate_infographics': True,
    'image_width': 600,
    'image_height': 400,
}

FLASK_CONFIG = {
    'host': '127.0.0.1',
    'port': 5001,
    'debug': True,
}
