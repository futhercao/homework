"""
Flask Web 应用 — 多模态灾害事件信息抽取系统
提供信息抽取、结果展示、知识图谱可视化、人工评价等功能
"""
import os
import json

from flask import Flask, render_template, request, jsonify, redirect, url_for, session

from config import FLASK_CONFIG, KG_DIR, EXTRACTION_CONFIG
from crawler import load_documents
from text_extractor import create_extractor
from image_extractor import ImageExtractor
from fusion import MultiModalFusion
from knowledge_graph import EventKnowledgeGraph
from evaluator import ExtractionEvaluator

app = Flask(__name__)
app.secret_key = 'ie_multimodal_2026_secret'

documents = []
extractions = {}
kg = EventKnowledgeGraph()
evaluator = ExtractionEvaluator()
text_extractor = create_extractor('ensemble')
image_extractor = ImageExtractor()
fusion_engine = MultiModalFusion()


def ensure_data():
    global documents
    if not documents:
        documents = load_documents()
    return bool(documents)


def run_extraction_for_doc(doc):
    """对单篇文档运行完整的多模态抽取流水线"""
    text_result = text_extractor.extract(doc.get('content', ''), doc.get('title', ''))

    image_results = []
    for img_info in doc.get('images', []):
        img_result = image_extractor.extract_from_image(img_info)
        image_results.append(img_result)

    fused = fusion_engine.fuse(text_result, image_results)

    return {
        'doc_id': doc['id'],
        'text_result': text_result,
        'image_results': image_results,
        'fused_result': fused,
        'event': fused.get('event', {}),
        'confidence': fused.get('confidence', {}),
        'fusion_log': fused.get('fusion_log', []),
        'modalities': fused.get('modalities_used', []),
    }


@app.route('/')
def home():
    data_ready = ensure_data()
    stats = {}
    if data_ready:
        type_counts = {}
        for doc in documents:
            t = doc.get('disaster_type', '未知')
            type_counts[t] = type_counts.get(t, 0) + 1
        stats = {
            'total_docs': len(documents),
            'disaster_types': len(type_counts),
            'type_distribution': type_counts,
            'docs_with_images': sum(1 for d in documents if d.get('images')),
            'total_extracted': len(extractions),
        }
    return render_template('index.html', data_ready=data_ready, stats=stats)


@app.route('/extract', methods=['GET', 'POST'])
def extract():
    """信息抽取页面"""
    if not ensure_data():
        return render_template('extract.html', error='数据未加载，请先生成数据。')

    if request.method == 'POST':
        doc_id = request.form.get('doc_id', '')
        action = request.form.get('action', 'single')

        if action == 'batch':
            count = 0
            for doc in documents:
                if doc['id'] not in extractions:
                    result = run_extraction_for_doc(doc)
                    extractions[doc['id']] = result
                    count += 1
            return redirect(url_for('results', message=f'批量抽取完成，共处理 {count} 篇文档'))

        doc = next((d for d in documents if d['id'] == doc_id), None)
        if not doc:
            return render_template('extract.html', documents=documents,
                                   error=f'文档 {doc_id} 未找到')

        result = run_extraction_for_doc(doc)
        extractions[doc['id']] = result
        return redirect(url_for('result_detail', doc_id=doc_id))

    return render_template('extract.html', documents=documents)


@app.route('/results')
def results():
    """抽取结果列表"""
    message = request.args.get('message', '')
    results_list = []
    for doc_id, ext in extractions.items():
        doc = next((d for d in documents if d['id'] == doc_id), {})
        results_list.append({
            'doc_id': doc_id,
            'title': doc.get('title', ''),
            'disaster_type': ext['event'].get('disaster_type', ''),
            'location': ext['event'].get('event_location', ''),
            'modalities': ext.get('modalities', []),
            'has_image': bool(ext.get('image_results')),
        })
    return render_template('results.html', results=results_list, message=message)


@app.route('/result/<doc_id>')
def result_detail(doc_id):
    """单篇文档的抽取结果详情"""
    ext = extractions.get(doc_id)
    if not ext:
        return redirect(url_for('extract'))

    doc = next((d for d in documents if d['id'] == doc_id), {})
    info_names = EXTRACTION_CONFIG['info_point_names']

    return render_template('detail.html',
                           doc=doc, ext=ext, info_names=info_names)


@app.route('/knowledge_graph')
def knowledge_graph_page():
    """知识图谱可视化"""
    if not extractions:
        return render_template('knowledge_graph.html',
                               kg_data=None, stats=None,
                               message='请先执行信息抽取')

    kg_local = EventKnowledgeGraph()
    for doc_id, ext in extractions.items():
        kg_local.add_event(doc_id, ext.get('fused_result', {}))

    kg_data = kg_local.to_vis_data()
    stats = kg_local.get_statistics()
    kg_local.save()

    return render_template('knowledge_graph.html',
                           kg_data=json.dumps(kg_data, ensure_ascii=False),
                           stats=stats)


@app.route('/evaluate', methods=['GET'])
def evaluate_page():
    """评价页面"""
    auto_result = None
    if extractions and documents:
        auto_result = evaluator.auto_evaluate_batch(documents, extractions)
    history = evaluator.get_summary()
    info_names = EXTRACTION_CONFIG['info_point_names']
    return render_template('evaluate.html',
                           auto_result=auto_result, history=history,
                           info_names=info_names,
                           extractions=extractions, documents=documents)


@app.route('/evaluate/human', methods=['POST'])
def submit_human_eval():
    """提交人工评价"""
    data = request.get_json()
    if not data:
        return jsonify({'error': '无效请求'}), 400

    doc_id = data.get('doc_id', '')
    judgments = data.get('judgments', {})
    ext = extractions.get(doc_id, {})

    result = evaluator.human_evaluate(doc_id, ext.get('event', {}), judgments)
    return jsonify({'success': True, 'result': result})


@app.route('/api/extract/<doc_id>')
def api_extract(doc_id):
    """API: 抽取单篇文档"""
    doc = next((d for d in documents if d['id'] == doc_id), None)
    if not doc:
        return jsonify({'error': '文档不存在'}), 404

    result = run_extraction_for_doc(doc)
    extractions[doc['id']] = result
    return jsonify({
        'doc_id': doc_id,
        'event': result['event'],
        'confidence': result['confidence'],
        'modalities': result['modalities'],
    })


@app.route('/api/batch_extract')
def api_batch_extract():
    """API: 批量抽取"""
    if not ensure_data():
        return jsonify({'error': '无数据'}), 400
    count = 0
    for doc in documents:
        if doc['id'] not in extractions:
            result = run_extraction_for_doc(doc)
            extractions[doc['id']] = result
            count += 1
    return jsonify({'success': True, 'extracted': count, 'total': len(extractions)})


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    print("=" * 60)
    print("  多模态灾害事件信息抽取系统")
    print(f"  访问: http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    print("=" * 60)
    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug'],
    )
