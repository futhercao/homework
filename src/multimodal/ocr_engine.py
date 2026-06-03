"""
OCR引擎 — 提取新闻图片中的文字信息
使用PaddleOCR（支持中英文）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MULTIMODAL_CONFIG


class OCREngine:
    """OCR文字提取引擎"""

    def __init__(self):
        self._ocr = None
        self.lang = MULTIMODAL_CONFIG.get('ocr_lang', 'ch')

    def _load_ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    use_gpu=False,
                    show_log=False,
                )
                print('[OCR] PaddleOCR 引擎已加载')
            except ImportError:
                print('[OCR] PaddleOCR 未安装，使用 EasyOCR 回退')
                try:
                    import easyocr
                    self._ocr = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                    print('[OCR] EasyOCR 引擎已加载')
                except ImportError:
                    print('[OCR] 无OCR引擎可用')
                    self._ocr = False
        return self._ocr if self._ocr is not False else None

    def extract_text(self, image_path):
        """从图片中提取文字"""
        ocr = self._load_ocr()
        if not ocr:
            return ''

        try:
            # PaddleOCR
            if hasattr(ocr, 'ocr'):
                result = ocr.ocr(image_path, cls=True)
            else:
                # EasyOCR
                result = ocr.readtext(image_path)

            if not result or (isinstance(result, list) and len(result) == 0):
                return ''

            texts = []
            for line in (result[0] if isinstance(result[0], list) and result and isinstance(result[0][0], list) else result):
                if hasattr(line, '__iter__') and not isinstance(line, str):
                    text = line[1][0] if len(line) > 1 and isinstance(line[1], (list, tuple)) else str(line[1]) if len(line) > 1 else ''
                else:
                    text = str(line)
                if text and len(text.strip()) >= 1:
                    texts.append(text.strip())

            return ' '.join(texts)
        except Exception as e:
            print(f'[OCR] 识别失败: {e}')
            return ''

    def extract_batch(self, image_paths):
        """批量OCR"""
        results = {}
        for path in image_paths:
            text = self.extract_text(path)
            if text:
                results[path] = text
        return results
