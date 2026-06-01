"""
使用 DeepSeek API 为真实新闻标注 ground truth（并发版）
"""
import json
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

API_KEY = "sk-de7e2cfc22ba47b4914f7780c65a13b0"
API_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"

PROMPT_TEMPLATE = """你是一个自然灾害事件标注专家。请分析以下新闻，判断它是否报道了一起自然灾害事件。

如果是灾害新闻，请提取以下7个信息点（中文文本，尽量精确匹配原文表述）：
1. disaster_type: 灾害类型（地震/台风/洪水/山体滑坡/泥石流/火灾/干旱/暴雪）
2. event_time: 发生时间（如"2026年5月18日"）
3. event_location: 发生地点（省+市，如"云南省昭通市"）
4. casualties: 伤亡情况（如"遇难3人，受伤5人"）
5. economic_loss: 经济损失（如"直接经济损失约2.3亿元"）
6. response_level: 响应级别（如"I级"/"II级"/"III级"/"IV级"）
7. rescue_org: 救援组织列表（如["中国红十字会", "国家消防救援局"]）

如果某个信息点在文中未提及，留空字符串或空列表。
如果不是灾害新闻，全部字段留空，is_disaster 填 false。

请严格按以下JSON格式返回，不要输出其他内容：
{{
  "is_disaster": true/false,
  "disaster_type": "...",
  "event_time": "...",
  "event_location": "...",
  "casualties": "...",
  "economic_loss": "...",
  "response_level": "...",
  "rescue_org": [...]
}}

新闻标题：{title}

新闻正文（前2000字）：
{content}
"""

print_lock = threading.Lock()
stats_lock = threading.Lock()


def load_real_docs(doc_dir, skip_annotated=True):
    docs = []
    for f in sorted(os.listdir(doc_dir)):
        if f.startswith('real_') and f.endswith('.json'):
            with open(os.path.join(doc_dir, f), 'r', encoding='utf-8') as fh:
                d = json.load(fh)
            if skip_annotated and 'ground_truth' in d:
                continue
            docs.append(d)
    return docs


def annotate_doc(doc, session, idx, total):
    """调用 API 标注单篇文档"""
    doc_id = doc['id']
    content = doc.get('content', '')[:2000]
    title = doc.get('title', '')
    prompt = PROMPT_TEMPLATE.replace('{title}', title).replace('{content}', content)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个精确的自然灾害事件信息标注专家。只返回JSON，不要解释。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 800,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            resp = session.post(
                f"{API_URL}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data['choices'][0]['message']['content'].strip()
                # 提取 JSON
                if '```' in text:
                    text = text.split('```')[1]
                    if text.startswith('json'):
                        text = text[4:]
                return json.loads(text)
            else:
                with print_lock:
                    print(f"  [{idx}/{total}] API错误 {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            with print_lock:
                print(f"  [{idx}/{total}] 重试 {attempt+1}: {e}")
        time.sleep(1)
    return None


def process_one(doc, doc_dir, idx, total, stats):
    """处理单篇文档的完整流程"""
    doc_id = doc['id']
    title = doc.get('title', '')[:60]

    session = requests.Session()
    annotation = annotate_doc(doc, session, idx, total)

    if annotation is None:
        with print_lock:
            print(f"  [{idx}/{total}] {title} [WARN] 标注失败，标记为非灾害")
        annotation = {"is_disaster": False}

    is_disaster = annotation.get('is_disaster', False)

    if is_disaster:
        gt = {
            "disaster_type": annotation.get('disaster_type', ''),
            "event_time": annotation.get('event_time', ''),
            "event_location": annotation.get('event_location', ''),
            "casualties": annotation.get('casualties', ''),
            "economic_loss": annotation.get('economic_loss', ''),
            "response_level": annotation.get('response_level', ''),
            "rescue_org": annotation.get('rescue_org', []),
        }
        with stats_lock:
            stats['disaster_count'] += 1
        with print_lock:
            print(f"  [{idx}/{total}] {title} [灾害] {gt['disaster_type']} @ {gt['event_location']}")
    else:
        gt = {
            "disaster_type": "",
            "event_time": "",
            "event_location": "",
            "casualties": "",
            "economic_loss": "",
            "response_level": "",
            "rescue_org": [],
        }
        with print_lock:
            print(f"  [{idx}/{total}] {title} [非灾害]")

    # Save back to file
    doc['ground_truth'] = gt
    doc['is_disaster'] = is_disaster
    filepath = os.path.join(doc_dir, f"{doc_id}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    return doc


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    doc_dir = os.path.join(base, 'homework3', 'data', 'documents')

    docs = load_real_docs(doc_dir)
    total = len(docs)
    print(f"加载了 {total} 篇真实文档")

    workers = 20  # 并发数，DeepSeek API 限制较宽松
    print(f"使用 {workers} 线程并发标注...")

    stats = {'disaster_count': 0}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, doc in enumerate(docs):
            fut = pool.submit(process_one, doc, doc_dir, i + 1, total, stats)
            futures[fut] = i

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  [ERROR] {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"标注完成: {stats['disaster_count']}/{total} 篇为灾害新闻")
    print(f"耗时: {elapsed:.0f}s ({elapsed/total:.1f}s/篇)")
    print(f"已保存 ground truth 到各文档的 JSON 文件中")


if __name__ == '__main__':
    main()
