"""
信息检索系统 (作业2) — Web 平台 FastAPI 后端

围绕"信息检索"评分点组织:
  [首页/统计]  检索首页 + 语料统计 + 可持续发展
  [检索]       BM25 / VSM / 混合(BM25+VSM+语义 RRF) + PRF 查询扩展
  [人工评价]   检索结果旁打分按钮(相关/部分/不相关)→ P@k / MAP / nDCG
  [多模态]     中文 CLIP 以文搜图 · 以图搜文(跨媒体检索)
  [看板]       语料与来源分布可视化

设计要点:
  - 倒排索引(真实新闻语料)惰性加载、持久化复用;
  - CLIP 向量索引一次离线构建, 在线查询免重复编码;
  - 全流程纯 CPU 可跑(BM25/VSM 轻量, ViT-B/16 量级中文 CLIP)。
"""
import os
import sys
import json
import re
import html as _html

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, Request, File, UploadFile, Query as QueryParam
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import WEB_CONFIG, DOC_DIR, IMAGE_DIR, DATA_DIR, VECTOR_DIR

app = FastAPI(title='信息检索系统', version='2.0')

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ====== 惰性单例 ======
_search_index = None
_retriever = None
_eval_store = None


def _ensure_eval_store():
    global _eval_store
    if _eval_store is None:
        from src.evaluation.metrics import EvaluationStore
        _eval_store = EvaluationStore()
    return _eval_store


def _ensure_search():
    global _search_index, _retriever
    if _search_index is None:
        from src.retrieval.indexer import SearchIndex
        from src.retrieval.hybrid import create_retriever
        _search_index = SearchIndex()
        if _search_index.load():
            _retriever = create_retriever(_search_index, 'hybrid')
    _ensure_eval_store()
    return _search_index.total_docs > 0


# ====== 本地文章渲染辅助 (检索结果点击 → 本地展示全文, 不跳转外站) ======
def _highlight(text, q):
    """转义后高亮查询词 (单次正则替换, 避免标签嵌套)"""
    esc = _html.escape(text)
    terms = set()
    qs = (q or '').strip()
    if qs:
        terms.add(qs)
        try:
            import jieba
            terms.update(w for w in jieba.lcut(qs) if w.strip())
        except Exception:
            pass
    terms = [t for t in terms if t.strip()]
    if not terms:
        return esc
    pat = re.compile('(' + '|'.join(re.escape(_html.escape(t))
                                   for t in sorted(terms, key=len, reverse=True)) + ')')
    return pat.sub(r'<mark>\1</mark>', esc)


def _content_html(content, q):
    """正文按段落渲染 + 高亮 (服务端生成, 模板 | safe)"""
    paras = [p for p in (content or '').split('\n') if p.strip()]
    if not paras:
        return '<p style="color:var(--text-muted)">(无正文)</p>'
    return ''.join(f'<p>{_highlight(p, q)}</p>' for p in paras)


def _doc_images(doc):
    """详情页配图: 只展示内容图(复用 VLM 质检结论过滤 logo/广告/二维码/截图),
    一律按 basename 走 /images 挂载, 规避绝对路径不可移植。"""
    from src.content_filter import filter_content_images
    return filter_content_images(doc.get('local_images') or [])


# ============================================================
# 首页 / 统计 / 可持续发展
# ============================================================

def _clip_image_count():
    """首页统计用: CLIP 图像索引的真实规模(读取 build_multimodal 构建时写出的 ntotal)。"""
    try:
        p = os.path.join(VECTOR_DIR, 'image_index_stats.json')
        return json.load(open(p, 'r', encoding='utf-8')).get('total_images', 0)
    except Exception:
        return 0


@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    ready = _ensure_search()
    stats = _search_index.get_stats() if ready else {}
    eval_summary = _eval_store.get_search_summary() if _eval_store else {}
    return templates.TemplateResponse('index.html', {
        'request': request, 'index_ready': ready, 'stats': stats,
        'eval_summary': eval_summary, 'clip_images': _clip_image_count(), 'title': '首页',
    })


@app.get('/sustainability', response_class=HTMLResponse)
async def sustainability_page(request: Request):
    return templates.TemplateResponse('sustainability.html', {'request': request, 'title': '可持续'})


@app.get('/api/stats')
async def stats():
    _ensure_search()
    s = _search_index.get_stats() if _search_index else {}
    from src.multimodal.image_store import ImageStore
    s['image_stats'] = ImageStore().stats()
    s['eval_stats'] = _eval_store.get_search_summary() if _eval_store else {}
    return s


# ============================================================
# 检索 / 人工评价 / 多模态 / 看板
# ============================================================

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
async def api_search(q: str = QueryParam(...), algo: str = 'hybrid',
                     expand: bool = False, page: int = 1, per_page: int = 10):
    if not _ensure_search():
        return JSONResponse({'error': '索引未构建'}, status_code=503)
    results = _retriever.search(q, top_k=50, algorithm=algo, use_expansion=expand)
    # 相关度归一化: 各算法原始分量纲不同 (BM25 无上界 / VSM 0~1 / RRF ~0.05),
    # 统一映射为"相对最佳匹配"的 0~100 相关度, 排序不变, 原始分仍保留在 score 字段。
    top_score = results[0]['score'] if results else 0
    for r in results:
        r['relevance'] = round(100.0 * r.get('score', 0) / top_score, 1) if top_score > 0 else 0.0
    total = len(results)
    start = (page - 1) * per_page
    page_results = results[start:start + per_page]
    return {
        'query': q, 'algorithm': algo, 'total': total, 'page': page,
        'per_page': per_page, 'total_pages': (total + per_page - 1) // per_page,
        'results': page_results,
        'expanded_terms': page_results[0].get('expanded_terms', []) if page_results else [],
    }


@app.post('/api/evaluate/search')
async def evaluate_search(data: dict):
    from src.evaluation.metrics import evaluate_search
    _ensure_eval_store()
    query = data.get('query', '')
    algorithm = data.get('algorithm', '')
    results = data.get('results', [])
    judgments = data.get('judgments', {})
    metrics = evaluate_search(query, results, judgments)
    _eval_store.save_search_eval(query, algorithm, results, judgments, metrics)
    return {'success': True, 'metrics': metrics}


@app.get('/api/evaluate/summary')
async def eval_summary(type: str = 'search'):
    _ensure_eval_store()
    return _eval_store.get_search_summary()


@app.get('/api/multimodal/search-images')
async def search_images(q: str = QueryParam(...), top_k: int = 12,
                        min_score: float = None, judge: bool = True):
    """以文搜图: CLIP 召回 → (默认) Qwen-VL 相关性精排。

    精排开启时: CLIP 按"召回下限"宽松召回候选, 再由视觉大模型逐张判定是否真与
    查询词相关, 只回相关者 —— 根治中文 CLIP 模态gap 导致的"搜地震出无关图"。
    无密钥 / 显式 judge=false → 退化为纯余弦阈值(保留可对比的旧行为)。
    """
    from src.multimodal.cross_modal import CrossModalSearch
    from src.config import MULTIMODAL_CONFIG as M
    cs = CrossModalSearch()

    use_judge = bool(judge) and M.get('vlm_judge_enabled', True)
    jj = None
    if use_judge:
        from src.multimodal.vlm_judge import VLMRelevanceJudge
        jj = VLMRelevanceJudge()
        if not jj.ready:               # 无密钥 → 不空筛, 退化为余弦阈值
            use_judge = False

    if use_judge:
        n_cand = int(M.get('vlm_judge_candidates', 12))
        raw = cs.search_images_by_text(q, max(n_cand, top_k))
        max_score = round(raw[0]['score'], 3) if raw else 0.0
        floor = M.get('vlm_judge_recall_floor', 0.28) if min_score is None else float(min_score)
        cand = [r for r in raw if r['score'] >= floor][:n_cand]
        annotated, jstats = jj.judge(q, cand)
        kept = [r for r in annotated if r['keep']][:top_k]
        return {'query': q, 'mode': 'vlm_judge', 'results': kept,
                'recall_floor': round(floor, 3), 'max_score': max_score,
                'scanned': len(raw), 'candidates': len(cand),
                'filtered_out': jstats['dropped'],
                'judge': {'enabled': True, 'ready': True, 'model': jj.model, **jstats}}

    # —— 退化路径: 纯余弦阈值过滤 ——
    thr = M.get('clip_text2img_min_score', 0.0) if min_score is None else float(min_score)
    raw = cs.search_images_by_text(q, top_k)
    max_score = round(raw[0]['score'], 3) if raw else 0.0
    kept = [r for r in raw if r['score'] >= thr][:top_k]
    return {'query': q, 'mode': 'cosine', 'results': kept, 'threshold': round(thr, 3),
            'max_score': max_score, 'scanned': len(raw), 'filtered_out': len(raw) - len(kept),
            'judge': {'enabled': False, 'ready': bool(jj and jj.ready)}}


@app.post('/api/multimodal/search-docs')
async def search_docs_by_image(image: UploadFile = File(...)):
    import tempfile
    suffix = os.path.splitext(image.filename or '')[1] or '.jpg'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name
    try:
        from src.multimodal.cross_modal import CrossModalSearch
        cs = CrossModalSearch()
        results = cs.search_docs_by_image(tmp_path, None, top_k=10)
        for r in results:
            did = r.get('doc_id', '')
            if not did:
                continue
            p = os.path.join(DOC_DIR, f'{did}.json')
            if os.path.exists(p):
                d = json.load(open(p, 'r', encoding='utf-8'))
                r['title'] = d.get('title', '')
                r['url'] = d.get('url', '')
    finally:
        os.unlink(tmp_path)
    return {'results': results}


@app.get('/api/documents')
async def list_documents(page: int = 1, per_page: int = 20, category: str = '', source: str = ''):
    from src.storage.doc_store import DocumentStore
    store = DocumentStore()
    docs = store.load_all(category=category or None, source=source or None,
                          limit=per_page, offset=(page - 1) * per_page)
    total = store.count(category=category, source=source)
    return {'total': total, 'page': page,
            'docs': [{'id': d.get('id'), 'title': d.get('title', '')[:60],
                      'category': d.get('category', ''), 'source': d.get('source', ''),
                      'date': d.get('date', '')} for d in docs]}


@app.get('/document/{doc_id}', response_class=HTMLResponse)
async def document_detail(request: Request, doc_id: str, q: str = ''):
    """本地文章详情页 — 点击检索结果在本系统内展示完整正文(不跳转外站)"""
    path = os.path.join(DOC_DIR, f'{doc_id}.json')
    doc = None
    if os.path.exists(path):
        try:
            doc = json.load(open(path, 'r', encoding='utf-8'))
        except Exception:
            doc = None
    ctx = {'request': request, 'title': '文档', 'q': q, 'doc': doc}
    if doc:
        ctx['content_html'] = _content_html(doc.get('content', ''), q)
        ctx['images'] = _doc_images(doc)
    return templates.TemplateResponse('document.html', ctx)


# 静态文件: 本地图片
app.mount('/images', StaticFiles(directory=IMAGE_DIR), name='images')


# ============================================================
def run():
    import uvicorn
    print("=" * 60)
    print("  信息检索系统 (作业2)")
    print(f"  访问: http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
    print("=" * 60)
    uvicorn.run(app, host=WEB_CONFIG['host'], port=WEB_CONFIG['port'])


if __name__ == '__main__':
    run()
