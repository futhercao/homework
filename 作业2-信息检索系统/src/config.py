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
# 向量索引随项目交付: 用 faiss.serialize_index + Python 文件IO 持久化,
# 不再依赖 faiss.write_index(其在 Windows 下无法处理中文路径)。
VECTOR_DIR = os.path.join(DATA_DIR, 'vectors')
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

    # 新闻种子URL — 权威采编源(国家通讯社/党媒/部委直属)的频道与列表页, BFS 从这些页发现正文链接。
    # 由门户聚合站(163/新浪/搜狐/腾讯)升级而来: 转载与快讯/广告少, 采编质量高。
    # 刻意纳入 社会/时政/滚动 频道, 以重新覆盖灾害(地震/台风/洪水)等现场报道与配图。
    'seed_urls': [
        # 新华网(国家通讯社)
        'http://www.news.cn/politics/',
        'http://www.news.cn/fortune/',
        'http://www.news.cn/tech/',
        'http://www.news.cn/local/',
        'http://www.news.cn/health/',
        # 人民网(党媒, 多频道)
        'http://society.people.com.cn/',
        'http://finance.people.com.cn/',
        'http://scitech.people.com.cn/',
        'http://edu.people.com.cn/',
        'http://health.people.com.cn/',
        # 央视网
        'https://news.cctv.com/',
        'https://news.cctv.com/china/',
        # 中国新闻网(滚动新闻链接海量, 题材广)
        'https://www.chinanews.com.cn/scroll-news/news1.html',
        'https://www.chinanews.com.cn/society/',
        'https://www.chinanews.com.cn/finance/',
        # 光明网
        'https://politics.gmw.cn/',
        'https://tech.gmw.cn/',
        'https://edu.gmw.cn/',
        # 中国经济网
        'http://www.ce.cn/xwzx/gnsz/',
        'http://finance.ce.cn/',
        # 科技日报
        'http://www.stdaily.com/',
        # 优质门户(深度报道; JS 较重, 预期低产, 作补充)
        'https://www.thepaper.cn/',
        'https://www.jiemian.com/lists/2.html',
    ],

    # 分类规则 (URL模式匹配) — 说明性配置; 实际分类由 src/crawler/parser.py 的 _classify 按内容关键词判定。
    'category_rules': {
        '科技': ['tech', 'scitech', 'stdaily', 'keji', 'kjrb'],
        '社会': ['society', 'shehui', 'local', 'people.com.cn', 'chinanews'],
        '财经': ['finance', 'fortune', 'money', 'jingji', 'ce.cn', 'caijing'],
        '教育': ['edu', 'learning'],
        '健康': ['health', 'jiankang'],
        '时政': ['politics', 'gnsz', 'shizheng'],
        '灾害': ['disaster', 'emergency', 'yingji'],
    },

    # 内容质量过滤(说明性: 实际正文长度闸在 src/crawler/parser.py 的 ArticleParser.MIN_LEN)
    'min_content_length': 250,
    'min_title_length': 8,
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
    # 中文原生 CLIP (Chinese-CLIP, 阿里达摩院, 大规模中文图文对训练): 中文"以文搜图/以图搜文"对齐更优。
    # 经 transformers 加载, projection_dim=512 与向量库一致。
    'clip_backend': 'chinese-clip',
    'clip_hf_id': 'OFA-Sys/chinese-clip-vit-base-patch16',
    # 回退后端: 多语言 open_clip (xlm-roberta 文本塔 + ViT-B-32 视觉塔), Chinese-CLIP 不可用时启用。
    'clip_model': 'xlm-roberta-base-ViT-B-32',
    'clip_pretrained': 'laion5b_s13b_b90k',
    'ocr_lang': 'ch',
    'image_index_dim': 512,
    'max_images_per_doc': 10,
    # 以文搜图相关性阈值: CLIP 永远返回"最相近"的 top-k, 但余弦低于此值的基本是
    # 背景相似度(无真正匹配)。补入真实灾害配图后实测: 有对应题材(地震/洪水/台风/芯片/
    # 新能源)真实匹配峰值约 0.38~0.45, 而无对应题材(篮球/熊猫/咖啡)噪声峰值仅 ≤0.377,
    # 故取 0.38 作为"强相关"下限(刚好压在噪声地板之上), 过滤弱匹配。前端可 0.34~0.45 滑动调节。
    'clip_text2img_min_score': 0.38,

    # --- VLM 图片质检 (可选): 用视觉大模型剔除无关图(logo/广告/二维码/纯文字/截图等) ---
    # 在"URL关键词+尺寸"廉价过滤之后再做一道语义级质检; 判定结果缓存到 vlm_verdicts.json。
    # 密钥仅从环境变量读取(绝不写入文件); 无密钥时应用已有缓存并跳过新图判定。
    'vlm_enabled': True,
    'vlm_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    # 视觉模型(看得见图): qwen3-vl-plus 实测可正确识别图片内容并输出严格 JSON。
    # (qwen3.7-plus 同样能读图, 改此一行即可切换; -vl- 版更快更省, 适合批量质检与逐查询评审。)
    'vlm_model': 'qwen3-vl-plus',
    'vlm_api_key_env': 'DASHSCOPE_API_KEY',
    'vlm_image_max_side': 512,
    'vlm_max_workers': 8,

    # --- 查询级 VLM 相关性评审 (以文搜图的"精排闸", 根治"搜地震出无关图") ---
    # 分工: CLIP 负责"召回"(快, 但模态gap使纯余弦阈值会漏判/误纳); VLM 负责"精排"——
    # 逐张审视 CLIP 召回的候选图是否真的与查询词相关, 只保留判定为相关者。
    # 判定按 (查询词, 文件名) 缓存到 vlm_judge_cache.json, 同词重复搜索零 API 开销。
    'vlm_judge_enabled': True,
    'vlm_judge_model': 'qwen3-vl-plus',
    'vlm_judge_candidates': 12,      # 送审的 CLIP 候选图上限(取余弦最高的前 N 张)
    'vlm_judge_recall_floor': 0.28,  # 候选召回下限: 低于此余弦的基本是噪声, 不送审(省钱省时)
    'vlm_judge_min_conf': 0.55,      # VLM 判为相关且置信度 >= 此值才保留
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
