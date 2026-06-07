"""
OCR引擎 — 提取新闻图片中的文字信息 (多媒体信息抽取的输入源)

- 主引擎: EasyOCR (easyocr.Reader(['ch_sim','en'], gpu=False))。
  纯CPU可跑, 中英文混排稳定, API 不随版本漂移。
- 回退:   PaddleOCR 3.x (.predict() 新接口)。旧的 use_angle_cls/show_log
  参数在 3.x 已移除, 故不再使用旧 .ocr() 路径。
- 识别结果按置信度过滤 (conf >= 0.3), 并以图片文件名为键缓存到
  data/ocr_cache.json, 避免重复识别 (CPU OCR 较慢, 缓存后演示秒开)。
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MULTIMODAL_CONFIG, DATA_DIR

OCR_CACHE_PATH = os.path.join(DATA_DIR, 'ocr_cache.json')
MIN_CONFIDENCE = 0.3


class OCREngine:
    """OCR文字提取引擎 (EasyOCR 主 / PaddleOCR 回退) + 磁盘缓存"""

    def __init__(self):
        self._reader = None
        self._backend = None        # 'easyocr' | 'paddle' | 'none'
        self.lang = MULTIMODAL_CONFIG.get('ocr_lang', 'ch')
        self._cache = self._load_cache()

    # ---------- 缓存 ----------
    def _load_cache(self):
        if os.path.exists(OCR_CACHE_PATH):
            try:
                with open(OCR_CACHE_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_cache(self):
        try:
            with open(OCR_CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f'[OCR] 缓存写入失败: {e}')

    # ---------- 引擎加载 (延迟) ----------
    def _load_engine(self):
        if self._backend is not None:
            return self._reader

        # 主引擎: EasyOCR
        try:
            import easyocr
            self._reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
            self._backend = 'easyocr'
            print('[OCR] EasyOCR 引擎已加载 (主)')
            return self._reader
        except Exception as e:
            print(f'[OCR] EasyOCR 不可用 ({e}); 尝试 PaddleOCR 回退')

        # 回退: PaddleOCR 3.x
        try:
            from paddleocr import PaddleOCR
            self._reader = PaddleOCR(lang=self.lang)
            self._backend = 'paddle'
            print('[OCR] PaddleOCR 引擎已加载 (回退)')
            return self._reader
        except Exception as e:
            print(f'[OCR] 无可用OCR引擎: {e}')
            self._backend = 'none'
            return None

    # ---------- 识别 ----------
    def extract_text(self, image_path, use_cache=True):
        """从图片提取文字, 返回拼接后的纯文本"""
        return self.extract_detail(image_path, use_cache=use_cache)['text']

    def extract_detail(self, image_path, use_cache=True):
        """返回 {text, lines:[{text,conf}], backend}

        lines 携带逐行置信度, 供 ocr.html 原图↔识别文本并排展示。
        """
        key = os.path.basename(image_path)
        if use_cache and key in self._cache:
            return self._cache[key]

        reader = self._load_engine()
        if reader is None:
            return {'text': '', 'lines': [], 'backend': 'none'}

        lines = []
        try:
            # 关键: EasyOCR/PaddleOCR 底层用 cv2.imread, 而 cv2 在 Windows 下无法读取
            # 含非ASCII(中文)的路径。用 PIL 读图(支持Unicode路径)转 numpy 数组再喂入,
            # 规避中文项目路径下 OCR 全部失败的问题。
            import numpy as np
            from PIL import Image
            arr = np.array(Image.open(image_path).convert('RGB'))
            if self._backend == 'easyocr':
                # readtext(ndarray) → [(bbox, text, conf), ...]
                for item in reader.readtext(arr):
                    text, conf = str(item[1]).strip(), float(item[2])
                    if conf >= MIN_CONFIDENCE and text:
                        lines.append({'text': text, 'conf': round(conf, 3)})
            elif self._backend == 'paddle':
                lines = self._paddle_predict(reader, arr)
        except Exception as e:
            print(f'[OCR] 识别失败 {image_path}: {e}')

        result = {
            'text': ' '.join(l['text'] for l in lines),
            'lines': lines,
            'backend': self._backend,
        }
        self._cache[key] = result
        return result

    @staticmethod
    def _paddle_predict(reader, image_path):
        """PaddleOCR 3.x .predict() 解析 (回退路径, 容错处理多种返回结构)"""
        lines = []
        out = reader.predict(image_path)
        for page in out:
            # 3.x 结果对象支持 dict 取值 (rec_texts / rec_scores)
            def _get(obj, k):
                if isinstance(obj, dict):
                    return obj.get(k, [])
                return getattr(obj, k, [])
            texts = _get(page, 'rec_texts')
            scores = _get(page, 'rec_scores')
            for text, conf in zip(texts, scores):
                text = str(text).strip()
                if float(conf) >= MIN_CONFIDENCE and text:
                    lines.append({'text': text, 'conf': round(float(conf), 3)})
        return lines

    def extract_batch(self, image_paths, save=True):
        """批量OCR, 期间分段落盘缓存; 返回 {path: text}"""
        results = {}
        total = len(image_paths)
        for i, path in enumerate(image_paths, 1):
            detail = self.extract_detail(path)
            if detail['text']:
                results[path] = detail['text']
            if i % 20 == 0:
                print(f'  [OCR] {i}/{total}')
                if save:
                    self.save_cache()
        if save:
            self.save_cache()
        return results
