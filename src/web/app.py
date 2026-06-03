"""
统一的Web平台 — FastAPI后端
整合: 搜索 / 信息抽取 / 多模态检索 / 人工评价 / 数据看板
"""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, Request, Form, File, UploadFile, Query as QueryParam
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional

from src.config import WEB_CONFIG, DOC_DIR, IMAGE_DIR

app = FastAPI(title='信息检索与抽取平台', version='2.0')

# 模板和静态文件
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# 延迟导入以避免循环依赖
_search_index = None
_retriever = None
_eval_store = None
_regex_extractor = None
_ner_enhancer = None
_event_builder = None
_kg_builder = None


def _ensure_search():
    global _search_index, _retriever, _eval_store
    if _search_index is None:
        from src.retrieval.indexer import SearchIndex
        from src.retrieval.hybrid import create_retriever
        from src.evaluation.metrics import EvaluationStore

        _search_index = SearchIndex()
        loaded = _search_index.load()
        if loaded:
            _retriever = create_retriever(_search_index, 'hybrid')
        _eval_store = EvaluationStore()
    return _search_index.total_docs > 0


def _ensure_extraction():
    global _regex_extractor, _ner_enhancer, _event_builder, _kg_builder
    if _regex_extractor is None:
        from src.extraction.regex_engine import RegexExtractor
        from src.extraction.ner_engine import NEREnhancer
        from src.extraction.event_builder import EventBuilder
        from src.extraction.kg_builder import KnowledgeGraphBuilder
        _regex_extractor = RegexExtractor()
        _ner_enhancer = NEREnhancer()
        _event_builder = EventBuilder()
        _kg_builder = KnowledgeGraphBuilder()
    return True


# ============================================================
# API 路由
# ============================================================

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    ready = _ensure_search()
    stats = _search_index.get_stats() if _search_index and _search_index.total_docs > 0 else {}
    eval_summary = _eval_store.get_search_summary() if _eval_store else {}
    return templates.TemplateResponse('index.html', {
        'request': request,
        'index_ready': ready,
        'stats': stats,
        'eval_summary': eval_summary,
        'title': '首页',
    })


@app.get('/extract', response_class=HTMLResponse)
async def extract_page(request: Request):
    return templates.TemplateResponse('extract.html', {'request': request, 'title': '抽取'})


@app.get('/evaluate', response_class=HTMLResponse)
async def evaluate_page(request: Request):
    _ensure_search()
    summary = _eval_store.get_search_summary() if _eval_store else {}
    return templates.TemplateResponse('evaluate.html', {'request': request, 'title': '评价', 'summary': summary})


@app.get('/dashboard', response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse('dashboard.html', {'request': request, 'title': '看板'})


@app.get('/multimodal', response_class=HTMLResponse)
async def multimodal_page(request: Request):
    return templates.TemplateResponse('multimodal.html', {'request': request, 'title': '多模态'})


@app.get('/api/search')
async def api_search(q: str = QueryParam(...),
                      algo: str = 'hybrid',
                      expand: bool = False,
                      page: int = 1,
                      per_page: int = 10):
    """搜索API"""
    if not _ensure_search():
        return JSONResponse({'error': '索引未构建'}, status_code=503)

    results = _retriever.search(q, top_k=50, algorithm=algo, use_expansion=expand)

    total = len(results)
    start = (page - 1) * per_page
    end = start + per_page
    page_results = results[start:end]
    total_pages = (total + per_page - 1) // per_page

    return {
        'query': q,
        'algorithm': algo,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages,
        'results': page_results,
        'expanded_terms': page_results[0].get('expanded_terms', []) if page_results else [],
    }


@app.post('/api/evaluate/search')
async def evaluate_search(data: dict):
    """提交检索评价"""
    from src.evaluation.metrics import evaluate_search
    query = data.get('query', '')
    algorithm = data.get('algorithm', '')
    results = data.get('results', [])
    judgments = data.get('judgments', {})

    metrics = evaluate_search(query, results, judgments)
    _eval_store.save_search_eval(query, algorithm, results, judgments, metrics)
    return {'success': True, 'metrics': metrics}


@app.get('/api/evaluate/summary')
async def eval_summary(type: str = 'search'):
    """评价汇总"""
    if type == 'search':
        return _eval_store.get_search_summary()
    return _eval_store.get_extraction_summary()


@app.post('/api/extract')
async def extract_document(doc_id: str = Form(...)):
    """信息抽取API"""
    _ensure_extraction()
    from src.crawler.engine import AsyncCrawlerEngine
    doc = None
    # 加载文档
    doc_path = os.path.join(DOC_DIR, f'{doc_id}.json')
    if not os.path.exists(doc_path):
        return JSONResponse({'error': '文档不存在'}, status_code=404)
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc = json.load(f)

    # 抽取
    regex_result = _regex_extractor.extract(doc)
    enhanced = _ner_enhancer.enhance(doc, regex_result)
    event = _event_builder.build(doc, enhanced)
    return {'success': True, 'event': event, 'extraction': enhanced}


@app.post('/api/extract/batch')
async def extract_batch(doc_ids: list):
    """批量抽取并构建知识图谱"""
    _ensure_extraction()
    events = []
    from src.crawler.engine import AsyncCrawlerEngine
    for did in doc_ids:
        doc_path = os.path.join(DOC_DIR, f'{did}.json')
        if not os.path.exists(doc_path):
            continue
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        regex_result = _regex_extractor.extract(doc)
        enhanced = _ner_enhancer.enhance(doc, regex_result)
        event = _event_builder.build(doc, enhanced)
        events.append(event)

    # 构建知识图谱
    kg = _kg_builder.build(events)
    _kg_builder.save()
    return {'success': True, 'num_events': len(events), 'kg': kg}


@app.post('/api/evaluate/extraction')
async def evaluate_extraction(data: dict):
    """提交抽取评价"""
    from src.evaluation.metrics import evaluate_extraction
    doc_id = data.get('doc_id', '')
    predictions = data.get('predictions', {})
    ground_truth = data.get('ground_truth', {})
    metrics = evaluate_extraction(predictions, ground_truth)
    _eval_store.save_extraction_eval(doc_id, predictions, ground_truth, metrics)
    return {'success': True, 'metrics': metrics}


@app.get('/api/multimodal/search-images')
async def search_images(q: str = QueryParam(...), top_k: int = 10):
    """多模态: 文字搜图片"""
    from src.multimodal.cross_modal import CrossModalSearch
    cs = CrossModalSearch()
    results = cs.search_images_by_text(q, top_k)
    return {'query': q, 'results': results}


@app.post('/api/multimodal/search-docs')
async def search_docs_by_image(image: UploadFile = File(...)):
    """多模态: 图片搜文档"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name
    from src.multimodal.cross_modal import CrossModalSearch
    cs = CrossModalSearch()
    results = cs.search_docs_by_image(tmp_path, None, top_k=10)
    os.unlink(tmp_path)
    return {'results': results}


# 静态文件: 提供本地图片访问
app.mount('/images', StaticFiles(directory=IMAGE_DIR), name='images')


@app.get('/api/stats')
async def stats():
    """系统统计"""
    _ensure_search()
    s = _search_index.get_stats() if _search_index else {}
    # 图片统计
    from src.multimodal.image_store import ImageStore
    img_store = ImageStore()
    s.update({'image_stats': img_store.stats()})
    s.update({'eval_stats': _eval_store.get_search_summary() if _eval_store else {}})
    return s


@app.get('/api/documents')
async def list_documents(page: int = 1, per_page: int = 20,
                          category: str = '', source: str = ''):
    """文档列表API"""
    from src.storage.doc_store import DocumentStore
    store = DocumentStore()
    docs = store.load_all(category=category or None, source=source or None,
                          limit=per_page, offset=(page - 1) * per_page)
    total = store.count(category=category, source=source)
    return {
        'total': total,
        'page': page,
        'docs': [{'id': d.get('id'), 'title': d.get('title', '')[:60],
                   'category': d.get('category', ''), 'source': d.get('source', ''),
                   'date': d.get('date', '')} for d in docs]
    }


# ============================================================
# 启动入口
# ============================================================

def run():
    import uvicorn
    print("=" * 60)
    print(f"  信息检索与抽取平台 v2.0")
    print(f"  访问: http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
    print("=" * 60)
    uvicorn.run(app, host=WEB_CONFIG['host'], port=WEB_CONFIG['port'])


if __name__ == '__main__':
    run()
