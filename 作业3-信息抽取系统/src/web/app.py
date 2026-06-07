"""
灾害信息抽取系统 (作业3) — Web 平台 FastAPI 后端

围绕"信息抽取"评分点组织:
  [抽取]       从灾害新闻抽取 7 要素(类型/时间/地点/伤亡/损失/响应级别/救援)
               采用创新的**集成抽取**(正则 + 触发词 + NER 加权投票)
  [知识图谱]   将事件关联为力导向图谱(事件↔地点↔类型↔机构)
  [OCR 多媒体] EasyOCR 识别灾情配图中的文字, 回灌进抽取(图片里的灾情也参与抽取)
  [人工评价]   每个字段旁打分按钮(正确/部分/错误)→ 逐字段准确率
  [可持续]     系统对环境与社会可持续发展的贡献

设计要点:
  - 抽取语料 (data/disaster_docs) 为真实灾害报道, 全部爬虫抓取, 无编造;
  - 命中 run_extraction 预计算结果 (extractions.json) 秒级返回, 否则在线集成抽取;
  - 图片一律按 basename 在本地 IMAGE_DIR 解析, 避免绝对路径不可移植 / Windows Unicode 路径问题。
"""
import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import WEB_CONFIG, IMAGE_DIR, DATA_DIR, VECTOR_DIR

app = FastAPI(title='灾害信息抽取系统', version='2.0')

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATE_DIR)

DISASTER_DIR = os.path.join(DATA_DIR, 'disaster_docs')
EXTRACTIONS_PATH = os.path.join(DATA_DIR, 'extractions.json')
KG_PATH = os.path.join(DATA_DIR, 'knowledge_graph.json')
SEVEN_FIELDS = ['disaster_type', 'event_time', 'event_location', 'casualties',
                'economic_loss', 'response_level', 'rescue_info']

# ====== 惰性单例 ======
_eval_store = None
_ensemble = None
_event_builder = None
_ocr_engine = None
_disaster_store = None
_extractions_cache = None


def _ensure_eval_store():
    global _eval_store
    if _eval_store is None:
        from src.evaluation.metrics import EvaluationStore
        _eval_store = EvaluationStore()
    return _eval_store


def _ensure_extraction():
    global _ensemble, _event_builder, _disaster_store, _extractions_cache
    if _ensemble is None:
        from src.extraction.ensemble import EnsembleExtractor
        from src.extraction.event_builder import EventBuilder
        from src.storage.doc_store import DocumentStore
        _ensemble = EnsembleExtractor()
        _event_builder = EventBuilder()
        _disaster_store = DocumentStore(doc_dir=DISASTER_DIR)
        _extractions_cache = {}
        if os.path.exists(EXTRACTIONS_PATH):
            try:
                _extractions_cache = json.load(open(EXTRACTIONS_PATH, 'r', encoding='utf-8'))
            except Exception:
                _extractions_cache = {}
    _ensure_eval_store()
    return True


def _load_disaster_doc(doc_id):
    path = os.path.join(DISASTER_DIR, f'{doc_id}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _resolve_image(local_path):
    """一律按 basename 在本地 IMAGE_DIR 解析, 保证文件夹可移植 + 规避 Windows Unicode 路径"""
    if not local_path:
        return ''
    return os.path.join(IMAGE_DIR, os.path.basename(local_path.replace('\\', '/')))


# ---- VLM 图片质检: 隐藏 logo / UI / 二维码 等无关配图 (裁决见 data/vectors/vlm_verdicts.json) ----
_img_verdicts = None


def _img_kept(name):
    """True=保留 (含无裁决 fail-open) / False=被 VLM 判为无关图。"""
    global _img_verdicts
    if _img_verdicts is None:
        _img_verdicts = {}
        vp = os.path.join(VECTOR_DIR, 'vlm_verdicts.json')
        if os.path.exists(vp):
            try:
                _img_verdicts = json.load(open(vp, 'r', encoding='utf-8'))
            except Exception:
                _img_verdicts = {}
    v = _img_verdicts.get(os.path.basename(str(name).replace('\\', '/')))
    return True if v is None else bool(v.get('keep'))


def _kept_local_images(doc):
    """文档中通过 VLM 质检的配图 (剔除 logo/UI/二维码 等无关图)。"""
    return [im for im in (doc.get('local_images') or [])
            if _img_kept(im.get('local_path') or '')]


def _extract_doc(doc, use_cache=True):
    """对单篇灾害文档抽取 (优先用预计算缓存, 否则在线集成抽取)"""
    did = doc.get('id', '')
    if use_cache and _extractions_cache and did in _extractions_cache:
        c = _extractions_cache[did]
        return {'event': c['event'], 'extraction': c['fields'],
                'ensemble': c.get('ensemble', {}), 'used_ocr': c.get('used_ocr', False),
                'ocr_images': doc.get('ocr_images', [])}
    aug = dict(doc)
    if doc.get('ocr_text'):
        aug['content'] = doc.get('content', '') + '\n【图片文字】' + doc['ocr_text']
    result = _ensemble.extract(aug)
    fields = {k: result[k] for k in SEVEN_FIELDS}
    event = _event_builder.build(doc, fields)
    return {'event': event, 'extraction': fields, 'ensemble': result.get('_ensemble', {}),
            'used_ocr': bool(doc.get('ocr_text')), 'ocr_images': doc.get('ocr_images', [])}


# ============================================================
# 页面
# ============================================================

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('extract.html', {'request': request, 'title': '抽取'})


@app.get('/kg', response_class=HTMLResponse)
async def kg_page(request: Request):
    return templates.TemplateResponse('kg.html', {'request': request, 'title': '图谱'})


@app.get('/ocr', response_class=HTMLResponse)
async def ocr_page(request: Request):
    return templates.TemplateResponse('ocr.html', {'request': request, 'title': 'OCR'})


@app.get('/evaluate-extraction', response_class=HTMLResponse)
async def evaluate_extraction_page(request: Request):
    _ensure_eval_store()
    summary = _eval_store.get_extraction_summary()
    return templates.TemplateResponse('evaluate_extraction.html',
                                      {'request': request, 'title': '抽取评价', 'summary': summary})


@app.get('/sustainability', response_class=HTMLResponse)
async def sustainability_page(request: Request):
    return templates.TemplateResponse('sustainability.html', {'request': request, 'title': '可持续'})


@app.get('/api/stats')
async def stats():
    """抽取系统总览统计"""
    _ensure_extraction()
    total = _disaster_store.count()
    cache = _extractions_cache or {}
    valid = sum(1 for c in cache.values() if c.get('event', {}).get('is_valid_event'))
    used_ocr = sum(1 for c in cache.values() if c.get('used_ocr'))
    kg = {}
    if os.path.exists(KG_PATH):
        try:
            kg = json.load(open(KG_PATH, 'r', encoding='utf-8')).get('stats', {})
        except Exception:
            kg = {}
    return {'total_docs': total, 'extracted': len(cache), 'valid_events': valid,
            'used_ocr': used_ocr, 'kg_stats': kg,
            'eval_stats': _eval_store.get_extraction_summary() if _eval_store else {}}


# ============================================================
# 抽取 API
# ============================================================

@app.get('/api/disaster/list')
async def disaster_list(page: int = 1, per_page: int = 300, dtype: str = ''):
    _ensure_extraction()
    docs = _disaster_store.load_all(limit=per_page, offset=(page - 1) * per_page)
    items = []
    for d in docs:
        if dtype and d.get('disaster_type', '') != dtype:
            continue
        items.append({'id': d['id'], 'title': d.get('title', '')[:70],
                      'source': d.get('source', ''), 'date': d.get('date', ''),
                      'disaster_type': d.get('disaster_type', ''),
                      'image_count': len(_kept_local_images(d)),
                      'has_ocr': bool(d.get('ocr_text'))})
    return {'total': _disaster_store.count(), 'docs': items}


@app.post('/api/extract')
async def extract_document(doc_id: str = Form(...)):
    _ensure_extraction()
    doc = _load_disaster_doc(doc_id)
    if not doc:
        return JSONResponse({'error': '文档不存在'}, status_code=404)
    out = _extract_doc(doc)
    return {'success': True, **out, 'title': doc.get('title', ''),
            'url': doc.get('url', ''), 'date': doc.get('date', ''),
            'source': doc.get('source', ''), 'content': doc.get('content', '')[:600]}


@app.post('/api/extract/batch')
async def extract_batch_all():
    """对全部灾害文档批量抽取并(重新)构建知识图谱"""
    _ensure_extraction()
    from src.extraction.kg_builder import KnowledgeGraphBuilder
    events = []
    for fp in sorted(glob.glob(os.path.join(DISASTER_DIR, 'dis_*.json'))):
        try:
            doc = json.load(open(fp, 'r', encoding='utf-8'))
        except Exception:
            continue
        events.append(_extract_doc(doc)['event'])
    kg_builder = KnowledgeGraphBuilder()    # 新实例, 避免跨请求累积
    kg = kg_builder.build(events)
    kg_builder.save()
    valid = sum(1 for e in events if e['is_valid_event'])
    return {'success': True, 'num_events': len(events), 'valid_events': valid, 'kg': kg}


@app.get('/api/kg')
async def get_kg():
    if os.path.exists(KG_PATH):
        with open(KG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'nodes': [], 'links': [], 'stats': {'total_nodes': 0, 'total_links': 0, 'total_events': 0}}


@app.get('/api/ocr/{doc_id}')
async def get_ocr(doc_id: str):
    """返回某文档配图的(预计算)OCR结果"""
    doc = _load_disaster_doc(doc_id)
    if not doc:
        return JSONResponse({'error': '文档不存在'}, status_code=404)
    return {'doc_id': doc_id, 'title': doc.get('title', ''),
            'images': _kept_local_images(doc),
            'ocr_images': [o for o in (doc.get('ocr_images') or []) if _img_kept(o.get('image', ''))]}


@app.post('/api/ocr/run/{doc_id}')
async def run_ocr_live(doc_id: str):
    """对某文档配图实时跑一次OCR (证明非预置; 命中缓存则秒回)"""
    global _ocr_engine
    doc = _load_disaster_doc(doc_id)
    if not doc:
        return JSONResponse({'error': '文档不存在'}, status_code=404)
    if _ocr_engine is None:
        from src.multimodal.ocr_engine import OCREngine
        _ocr_engine = OCREngine()
    out = []
    for img in _kept_local_images(doc):
        p = _resolve_image(img.get('local_path', ''))
        if p and os.path.exists(p):
            detail = _ocr_engine.extract_detail(p)
            out.append({'image': os.path.basename(p), 'text': detail['text'],
                        'lines': detail['lines'], 'backend': detail.get('backend', '')})
    _ocr_engine.save_cache()
    return {'doc_id': doc_id, 'ocr_images': out}


@app.post('/api/evaluate/extraction')
async def evaluate_extraction_ep(data: dict):
    """提交抽取人工评价 (逐字段 正确/部分/错误)"""
    from src.evaluation.metrics import evaluate_extraction_human
    _ensure_eval_store()
    doc_id = data.get('doc_id', '')
    predictions = data.get('predictions', {})
    judgments = data.get('judgments', {})
    metrics = evaluate_extraction_human(predictions, judgments)
    _eval_store.save_extraction_eval(doc_id, predictions, judgments, metrics)
    return {'success': True, 'metrics': metrics}


@app.get('/api/evaluate/extraction/summary')
async def extraction_eval_summary():
    _ensure_eval_store()
    return _eval_store.get_extraction_summary()


# 静态文件: 本地图片 (灾情配图)
app.mount('/images', StaticFiles(directory=IMAGE_DIR), name='images')


# ============================================================
def run():
    import uvicorn
    print("=" * 60)
    print("  灾害信息抽取系统 (作业3)")
    print(f"  访问: http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
    print("=" * 60)
    uvicorn.run(app, host=WEB_CONFIG['host'], port=WEB_CONFIG['port'])


if __name__ == '__main__':
    run()
