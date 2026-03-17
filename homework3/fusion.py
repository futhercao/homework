"""
跨模态信息融合引擎（核心创新点）
将文本抽取结果与图像 OCR/场景分析结果进行融合，
采用置信度加权投票机制，实现多模态互补与验证。

融合策略:
1. 信息互补 — 某模态缺失的信息点由另一模态补充
2. 交叉验证 — 多模态一致的信息点获得更高置信度
3. 冲突消解 — 不一致时按模态权重和置信度决策
"""


class MultiModalFusion:
    """多模态信息融合器"""

    MODALITY_WEIGHTS = {
        'text': 0.6,
        'ocr': 0.25,
        'scene': 0.1,
        'metadata': 0.05,
    }

    def fuse(self, text_result, image_results):
        """
        融合文本和图像的抽取结果

        Args:
            text_result: EnsembleExtractor 输出
            image_results: list[ImageExtractor.extract_from_image 输出]

        Returns:
            dict: 融合后的结构化事件信息
        """
        text_info = text_result.get('fused', {})
        text_conf = text_result.get('confidence', {})

        image_info = self._merge_image_results(image_results)

        fused = {}
        confidence = {}
        fusion_log = []

        info_points = [
            'disaster_type', 'event_time', 'event_location',
            'casualties', 'economic_loss', 'response_level', 'rescue_org',
        ]

        for point in info_points:
            text_val = text_info.get(point, '')
            text_c = text_conf.get(point, 0.0)
            img_val = image_info.get(point, {}).get('value', '')
            img_c = image_info.get(point, {}).get('confidence', 0.0)

            if point == 'rescue_org':
                text_orgs = text_val if isinstance(text_val, list) else []
                img_orgs = img_val if isinstance(img_val, list) else []
                merged = list(set(text_orgs + img_orgs))[:5]
                fused[point] = merged
                confidence[point] = max(text_c, img_c)
                fusion_log.append({
                    'point': point,
                    'strategy': 'union',
                    'text_val': text_orgs,
                    'image_val': img_orgs,
                    'final': merged,
                })
                continue

            if text_val and img_val:
                if self._is_consistent(point, text_val, img_val):
                    fused[point] = text_val
                    confidence[point] = min(1.0, text_c * 0.6 + img_c * 0.4 + 0.15)
                    strategy = 'cross_validated'
                else:
                    if text_c >= img_c:
                        fused[point] = text_val
                        confidence[point] = text_c * 0.8
                    else:
                        fused[point] = img_val
                        confidence[point] = img_c * 0.8
                    strategy = 'conflict_resolved'
            elif text_val:
                fused[point] = text_val
                confidence[point] = text_c
                strategy = 'text_only'
            elif img_val:
                fused[point] = img_val
                confidence[point] = img_c * 0.9
                strategy = 'image_complement'
            else:
                fused[point] = ''
                confidence[point] = 0.0
                strategy = 'missing'

            fusion_log.append({
                'point': point,
                'strategy': strategy,
                'text_val': text_val,
                'text_conf': text_c,
                'image_val': img_val,
                'image_conf': img_c,
                'final_val': fused[point],
                'final_conf': confidence[point],
            })

        return {
            'event': fused,
            'confidence': confidence,
            'fusion_log': fusion_log,
            'modalities_used': self._get_modalities(text_result, image_results),
        }

    def _merge_image_results(self, image_results):
        """合并多张图片的抽取结果"""
        if not image_results:
            return {}

        merged = {}
        for img_result in image_results:
            ocr_ext = img_result.get('ocr_extraction', {})
            for k, v in ocr_ext.items():
                if v and k not in merged:
                    merged[k] = {'value': v, 'confidence': 0.7, 'source': 'ocr'}

            scene = img_result.get('scene_analysis', {})
            inferred = scene.get('inferred_type', '')
            scene_conf = scene.get('confidence', 0.0)
            if inferred and 'disaster_type' not in merged:
                merged['disaster_type'] = {
                    'value': inferred, 'confidence': scene_conf, 'source': 'scene'
                }

            meta_ext = img_result.get('metadata_extraction', {})
            for k, v in meta_ext.items():
                if v and k not in merged:
                    merged[k] = {'value': v, 'confidence': 0.3, 'source': 'metadata'}

        return merged

    def _is_consistent(self, point, val1, val2):
        """判断两个值是否一致"""
        if isinstance(val1, str) and isinstance(val2, str):
            if val1 == val2:
                return True
            if val1 in val2 or val2 in val1:
                return True
            if point in ('casualties', 'economic_loss'):
                import re
                nums1 = set(re.findall(r'\d+', val1))
                nums2 = set(re.findall(r'\d+', val2))
                return bool(nums1 & nums2)
        return False

    def _get_modalities(self, text_result, image_results):
        """统计使用了哪些模态"""
        modalities = ['text']
        if image_results:
            for r in image_results:
                if r.get('ocr_text'):
                    modalities.append('ocr')
                if r.get('scene_analysis', {}).get('inferred_type'):
                    modalities.append('scene')
                if r.get('metadata_extraction'):
                    modalities.append('metadata')
        return list(set(modalities))
