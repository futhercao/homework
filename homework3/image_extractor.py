"""
多媒体信息抽取模块（创新点）
实现从图像中提取结构化灾害信息：
1. OCR 文字识别 — 从新闻配图 / 信息图中提取文字
2. 图像场景分析 — 通过颜色分布推断灾害类型
3. 图像元数据抽取 — 从 alt 文本和文件名中获取线索
支持 easyocr / pytesseract / 回退模式，确保无需 GPU 也能运行
"""
import os
import re
from collections import Counter

from config import IMAGE_DIR, IMAGE_CONFIG

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

_ocr_engine = None


def _get_ocr_engine():
    """懒加载 OCR 引擎，优先 easyocr → pytesseract → 回退"""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    backend = IMAGE_CONFIG.get('ocr_backend', 'auto')

    if backend in ('auto', 'easyocr'):
        try:
            import easyocr
            _ocr_engine = ('easyocr', easyocr.Reader(['ch_sim', 'en'], gpu=False))
            print("[OCR] 使用 EasyOCR 引擎")
            return _ocr_engine
        except ImportError:
            pass

    if backend in ('auto', 'pytesseract'):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            _ocr_engine = ('pytesseract', pytesseract)
            print("[OCR] 使用 Tesseract 引擎")
            return _ocr_engine
        except Exception:
            pass

    _ocr_engine = ('fallback', None)
    print("[OCR] 使用回退模式（利用图片元数据）")
    return _ocr_engine


class ImageExtractor:
    """从图像中抽取灾害信息"""

    DISASTER_COLORS = {
        '地震': {'dominant': [(150, 80, 60), (180, 60, 60)], 'name': '暖红/棕色'},
        '台风': {'dominant': [(50, 100, 160), (60, 120, 180)], 'name': '深蓝色'},
        '洪水': {'dominant': [(40, 90, 150), (70, 130, 180)], 'name': '蓝色'},
        '山体滑坡': {'dominant': [(120, 100, 60), (140, 110, 70)], 'name': '棕黄色'},
        '火灾': {'dominant': [(200, 80, 30), (220, 100, 40)], 'name': '橙红色'},
        '干旱': {'dominant': [(190, 160, 80), (200, 170, 90)], 'name': '黄色'},
        '暴雪': {'dominant': [(160, 180, 200), (180, 200, 220)], 'name': '浅蓝/灰白'},
    }

    def extract_from_image(self, image_info):
        """
        对单张图片进行多模态信息抽取

        Args:
            image_info: dict 包含 path, filename, alt, ocr_text(可选)

        Returns:
            dict: 抽取结果
        """
        result = {
            'source': 'image',
            'ocr_text': '',
            'ocr_extraction': {},
            'scene_analysis': {},
            'metadata_extraction': {},
        }

        ocr_text = self._do_ocr(image_info)
        result['ocr_text'] = ocr_text

        if ocr_text:
            result['ocr_extraction'] = self._extract_from_ocr_text(ocr_text)

        img_path = image_info.get('path', '')
        if img_path and os.path.exists(img_path) and HAS_PIL:
            result['scene_analysis'] = self._analyze_scene(img_path)

        result['metadata_extraction'] = self._extract_from_metadata(image_info)

        return result

    def _do_ocr(self, image_info):
        """执行 OCR 文字识别"""
        if image_info.get('ocr_text'):
            return image_info['ocr_text']

        img_path = image_info.get('path', '')
        if not img_path or not os.path.exists(img_path):
            return ''

        engine_type, engine = _get_ocr_engine()

        if engine_type == 'easyocr':
            try:
                results = engine.readtext(img_path, detail=0)
                return '\n'.join(results)
            except Exception as e:
                print(f"[OCR] EasyOCR 识别失败: {e}")
                return ''

        if engine_type == 'pytesseract':
            try:
                img = Image.open(img_path)
                text = engine.image_to_string(img, lang='chi_sim+eng')
                return text.strip()
            except Exception as e:
                print(f"[OCR] Tesseract 识别失败: {e}")
                return ''

        return image_info.get('ocr_text', '')

    def _extract_from_ocr_text(self, ocr_text):
        """从 OCR 识别出的文字中抽取信息点"""
        extracted = {}

        disaster_types = ['地震', '台风', '洪水', '山体滑坡', '火灾', '干旱', '暴雪']
        for dt in disaster_types:
            if dt in ocr_text:
                extracted['disaster_type'] = dt
                break

        time_m = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日)', ocr_text)
        if time_m:
            extracted['event_time'] = time_m.group(1)

        loc_m = re.search(
            r'([\u4e00-\u9fa5]{2,6}(?:省|自治区|市)[\u4e00-\u9fa5]{0,6}(?:市|州|区|县)?)',
            ocr_text
        )
        if loc_m:
            extracted['event_location'] = loc_m.group(1)

        cas_parts = []
        dead_m = re.search(r'遇难(\d+)人', ocr_text)
        inj_m = re.search(r'受伤(\d+)人', ocr_text)
        if dead_m:
            cas_parts.append(f'遇难{dead_m.group(1)}人')
        if inj_m:
            cas_parts.append(f'受伤{inj_m.group(1)}人')
        if cas_parts:
            extracted['casualties'] = '，'.join(cas_parts)

        loss_m = re.search(r'(\d+(?:\.\d+)?)亿元', ocr_text)
        if loss_m:
            extracted['economic_loss'] = f'{loss_m.group(1)}亿元'

        level_m = re.search(r'(I{1,4}级|IV级|III级|II级)', ocr_text)
        if level_m:
            extracted['response_level'] = level_m.group(1)

        return extracted

    def _analyze_scene(self, img_path):
        """
        通过图像颜色分布分析灾害场景类型。
        创新点：利用主色调推断灾害类型。
        """
        analysis = {'dominant_colors': [], 'inferred_type': '', 'confidence': 0.0}

        try:
            img = Image.open(img_path).convert('RGB')
            img_small = img.resize((100, 100))
            pixels = list(img_small.getdata())

            color_bins = Counter()
            for r, g, b in pixels:
                r_bin = (r // 32) * 32
                g_bin = (g // 32) * 32
                b_bin = (b // 32) * 32
                color_bins[(r_bin, g_bin, b_bin)] += 1

            top_colors = color_bins.most_common(5)
            analysis['dominant_colors'] = [
                {'rgb': list(c), 'ratio': round(n / len(pixels), 2)}
                for c, n in top_colors
            ]

            best_type = ''
            best_score = 0
            for dtype, color_info in self.DISASTER_COLORS.items():
                score = 0
                for ref_color in color_info['dominant']:
                    for (c, n) in top_colors[:3]:
                        dist = sum((a - b) ** 2 for a, b in zip(c, ref_color)) ** 0.5
                        if dist < 80:
                            score += n / len(pixels) * (1 - dist / 80)
                if score > best_score:
                    best_score = score
                    best_type = dtype

            analysis['inferred_type'] = best_type
            analysis['confidence'] = round(min(best_score, 1.0), 2)

        except Exception as e:
            analysis['error'] = str(e)

        return analysis

    def _extract_from_metadata(self, image_info):
        """从图片元数据 (alt / filename) 中抽取信息"""
        extracted = {}
        alt = image_info.get('alt', '')
        filename = image_info.get('filename', '')

        combined = f'{alt} {filename}'
        disaster_types = ['地震', '台风', '洪水', '山体滑坡', '火灾', '干旱', '暴雪']
        for dt in disaster_types:
            if dt in combined:
                extracted['disaster_type'] = dt
                break

        return extracted

    def extract_batch(self, documents):
        """批量处理文档中的图片"""
        results = {}
        for doc in documents:
            doc_id = doc.get('id', '')
            images = doc.get('images', [])
            if images:
                doc_results = []
                for img_info in images:
                    r = self.extract_from_image(img_info)
                    doc_results.append(r)
                results[doc_id] = doc_results
        return results


class AudioExtractor:
    """
    音频信息抽取（扩展创新）
    从灾害新闻音频/视频中提取语音内容并抽取信息。
    支持 whisper 模型或回退到文本模式。
    """

    def __init__(self):
        self._whisper_model = None

    def extract_from_audio(self, audio_path):
        """从音频文件抽取文本信息"""
        transcript = self._transcribe(audio_path)
        if not transcript:
            return {'source': 'audio', 'transcript': '', 'extraction': {}}

        from text_extractor import RegexExtractor
        extractor = RegexExtractor()
        extraction = extractor.extract(transcript)

        return {
            'source': 'audio',
            'transcript': transcript,
            'extraction': extraction,
        }

    def _transcribe(self, audio_path):
        """语音转文字"""
        try:
            import whisper
            if self._whisper_model is None:
                self._whisper_model = whisper.load_model("base")
            result = self._whisper_model.transcribe(audio_path, language='zh')
            return result.get('text', '')
        except ImportError:
            print("[音频] whisper 未安装，跳过音频抽取")
            return ''
        except Exception as e:
            print(f"[音频] 转录失败: {e}")
            return ''
