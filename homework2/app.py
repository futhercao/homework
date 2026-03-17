"""
Flask Web应用 - 信息检索系统Web界面
提供搜索、结果展示、人工评价等功能
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import json

from config import FLASK_CONFIG, INDEX_DIR, EVAL_DIR
from indexer import InvertedIndex
from retrieval import create_retriever, QueryExpander
from evaluator import SearchEvaluator

app = Flask(__name__)
app.secret_key = 'ir_system_secret_key_2026'

index = InvertedIndex()
evaluator = SearchEvaluator()
_index_loaded = False


def ensure_index():
    global _index_loaded
    if not _index_loaded:
        if index.load():
            _index_loaded = True
        else:
            return False
    return True


@app.route('/')
def home():
    """搜索首页"""
    index_ready = ensure_index()
    stats = index.get_stats() if index_ready else {}
    return render_template('index.html', index_ready=index_ready, stats=stats)


@app.route('/search')
def search():
    """执行搜索并返回结果"""
    query = request.args.get('q', '').strip()
    algorithm = request.args.get('algo', 'bm25')
    use_expansion = request.args.get('expand', '') == 'on'
    page = int(request.args.get('page', 1))
    per_page = 10

    if not query:
        return redirect(url_for('home'))

    if not ensure_index():
        return render_template('results.html', query=query, results=[],
                               error='索引未构建，请先运行数据采集和索引构建。')

    retriever = create_retriever(index, algorithm)

    expanded_terms = []
    if use_expansion:
        expander = QueryExpander(index)
        expanded_terms = expander.expand(query)

    results = retriever.search(query, top_k=50)

    total = len(results)
    start = (page - 1) * per_page
    end = start + per_page
    page_results = results[start:end]
    total_pages = (total + per_page - 1) // per_page

    session['last_query'] = query
    session['last_algorithm'] = algorithm
    session['last_results'] = json.dumps(results[:20], ensure_ascii=False)

    return render_template(
        'results.html',
        query=query,
        algorithm=algorithm,
        results=page_results,
        total=total,
        page=page,
        total_pages=total_pages,
        expanded_terms=expanded_terms,
        use_expansion=use_expansion,
    )


@app.route('/evaluate', methods=['GET'])
def evaluate_page():
    """评价页面 - 展示最近的搜索结果供人工评价"""
    query = session.get('last_query', '')
    results_json = session.get('last_results', '[]')
    results = json.loads(results_json)
    algorithm = session.get('last_algorithm', 'bm25')

    history = evaluator.get_summary()

    return render_template(
        'evaluate.html',
        query=query,
        algorithm=algorithm,
        results=results,
        history=history,
    )


@app.route('/evaluate', methods=['POST'])
def submit_evaluation():
    """提交人工评价结果"""
    data = request.get_json()
    if not data:
        return jsonify({'error': '无效的请求数据'}), 400

    query = data.get('query', '')
    judgments = data.get('judgments', {})

    results_json = session.get('last_results', '[]')
    results = json.loads(results_json)

    if not results or not judgments:
        return jsonify({'error': '没有可评价的结果'}), 400

    metrics = evaluator.evaluate(query, results, judgments)
    metrics_rounded = {k: round(v, 4) for k, v in metrics.items()}

    return jsonify({
        'success': True,
        'metrics': metrics_rounded,
        'message': '评价已保存',
    })


@app.route('/evaluate/history')
def evaluation_history():
    """查看评价历史"""
    summary = evaluator.get_summary()
    return jsonify(summary)


@app.route('/stats')
def stats():
    """系统统计信息"""
    if not ensure_index():
        return jsonify({'error': '索引未加载'})
    return jsonify(index.get_stats())


@app.route('/about')
def about():
    """关于页面 - 包含可持续发展影响分析"""
    return render_template('about.html')


if __name__ == '__main__':
    print("=" * 60)
    print("  信息检索系统 Web 界面")
    print(f"  访问地址: http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    print("=" * 60)
    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug'],
    )
