"""
NER增强信息抽取
利用LAC命名实体识别补充正则匹配的不足
- 地名实体识别（补充不匹配正则的地名）
- 机构名实体识别（补充救援组织）
- 人名实体识别（关键人物）
"""
import re

from src.nlp.pipeline import nlp


class NEREnhancer:
    """NER增强抽取 — 补充正则引擎的盲区"""

    def enhance(self, doc, regex_result):
        """用NER结果补充正则抽取结果"""
        title = doc.get('title', '')
        content = doc.get('content', '')
        full_text = f"{title}\n{content}"

        entities = nlp.ner(full_text[:5000])

        # 补充地点（如果正则没抽到）
        if not regex_result.get('event_location'):
            locs = entities.get('LOC', [])
            # 过滤太短或太泛的地点
            valid_locs = [l for l in locs if len(l) >= 3 and not l.endswith(('省',))]
            if valid_locs:
                # 选最具体的（最长的）
                regex_result['event_location'] = max(valid_locs, key=len)

        # 补充救援组织
        orgs = entities.get('ORG', [])
        rescue_current = regex_result.get('rescue_info', [])
        for org in orgs:
            if any(kw in org for kw in ['消防', '救援', '应急', '救灾', '红十字',
                                         '武警', '公安', '部队', '支队', '总队',
                                         '管理局', '应急局']):
                if org not in rescue_current:
                    rescue_current.append(org)
        regex_result['rescue_info'] = rescue_current[:10]

        return regex_result
