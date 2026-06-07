"""
对灾害文档配图批量 OCR (EasyOCR) — 多媒体信息抽取的输入

- 遍历 data/disaster_docs 各文档的 local_images, 逐图 OCR (结果缓存到 data/ocr_cache.json);
- 把识别出的文字回灌进文档: 写入 doc['ocr_text'] (拼接文本) 与 doc['ocr_images']
  (逐图 {image, text, lines} 供 ocr.html 原图↔识别文本并排展示);
- run_extraction.py 会把 ocr_text 追加到正文后再抽取, 实现"图片里的灾情文字也参与抽取"。

CPU OCR 较慢(每图数秒), 但有缓存: 重跑只处理未缓存图片, 演示走缓存秒级。

用法: python scripts/run_ocr.py [--max N]   # --max 限制处理文档数, 便于快速试跑
"""
import os
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, IMAGE_DIR
from src.multimodal.ocr_engine import OCREngine
from src.multimodal.vlm_filter import VLMFilter

DISASTER_DIR = os.path.join(DATA_DIR, 'disaster_docs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max', type=int, default=0, help='最多处理多少篇文档 (0=全部)')
    args = ap.parse_args()

    eng = OCREngine()
    vlm = VLMFilter()   # VLM 图片质检: 仅对"内容图"做 OCR, 跳过 logo/UI/二维码等无关图
    files = sorted(glob.glob(os.path.join(DISASTER_DIR, 'dis_*.json')))
    if args.max:
        files = files[:args.max]

    docs_updated = 0
    imgs_seen = imgs_with_text = 0
    for i, fp in enumerate(files, 1):
        try:
            doc = json.load(open(fp, 'r', encoding='utf-8'))
        except Exception:
            continue

        ocr_images = []
        for img in (doc.get('local_images') or []):
            raw = img.get('local_path', '')
            # 一律按 basename 在 IMAGE_DIR 解析: 兼容相对/绝对存储, 规避不可移植与 cwd 依赖
            p = os.path.join(IMAGE_DIR, os.path.basename(raw.replace('\\', '/'))) if raw else ''
            if not p or not os.path.exists(p):
                continue
            if vlm.verdict(os.path.basename(p)) is False:   # VLM 判为无关图 -> 不 OCR、不参与抽取
                continue
            imgs_seen += 1
            detail = eng.extract_detail(p)      # 命中缓存则直接返回
            if detail.get('text'):
                imgs_with_text += 1
                ocr_images.append({
                    'image': os.path.basename(p),
                    'original_src': img.get('original_src', ''),
                    'text': detail['text'],
                    'lines': detail['lines'],
                    'backend': detail.get('backend', ''),
                })

        # 回灌
        doc['ocr_images'] = ocr_images
        doc['ocr_text'] = ' '.join(o['text'] for o in ocr_images)
        if ocr_images:
            docs_updated += 1
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        if i % 20 == 0:
            print(f'  [{i}/{len(files)}] 已识别含文字图 {imgs_with_text}/{imgs_seen}')
            eng.save_cache()

    eng.save_cache()
    print(f'\n[OCR] 完成: 处理 {len(files)} 篇文档, {imgs_seen} 张配图, '
          f'其中 {imgs_with_text} 张含文字, {docs_updated} 篇文档获得OCR文本')
    print(f'[OCR] 缓存: {os.path.join(DATA_DIR, "ocr_cache.json")}')


if __name__ == '__main__':
    main()
