"""
文本信息抽取模块
实现三种抽取策略：
1. RegexExtractor   — 基于正则表达式的模式匹配（课程基本要求）
2. NERExtractor     — 基于命名实体识别的抽取（jieba + 规则）
3. DependencyExtractor — 基于依存句法分析的抽取
4. EnsembleExtractor — 多策略融合的集成抽取器

抽取的 7 个信息点构成一个"自然灾害事件"：
  灾害类型 / 发生时间 / 发生地点 / 伤亡情况 / 经济损失 / 响应级别 / 救援组织
"""
import re
from collections import defaultdict

import jieba
import jieba.posseg as pseg

from config import EXTRACTION_CONFIG


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 正则表达式抽取器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RegexExtractor:
    """基于正则表达式的信息抽取"""

    DISASTER_TYPES = ['地震', '台风', '洪水', '暴雨', '山体滑坡', '泥石流',
                      '火灾', '森林火灾', '草原火灾', '山林火灾', '干旱',
                      '暴雪', '飓风', '龙卷风', '冰雹', '海啸']

    PATTERNS = {
        'disaster_type': [
            re.compile(r'发生(?:了)?(?:里氏)?[\d.]+级(地震)'),
            re.compile(r'(台风|飓风)(?:"|「)?[\u4e00-\u9fa5]+(?:"|」)?'),
            re.compile(r'(洪水|洪涝|特大洪水)'),
            re.compile(r'(山体滑坡|泥石流)'),
            re.compile(r'(森林火灾|草原火灾|山林火灾|火灾)'),
            re.compile(r'(?:严重|特大)?(干旱)'),
            re.compile(r'(暴雪|暴风雪)'),
        ],

        'event_time': [
            re.compile(r'(\d{4}年\d{1,2}月\d{1,2}日)'),
            re.compile(r'(\d{4}-\d{1,2}-\d{1,2})'),
            re.compile(r'(\d{4}/\d{1,2}/\d{1,2})'),
            re.compile(r'(今年\d{1,2}月\d{1,2}日)'),
            re.compile(r'(昨[天日]|前[天日]|今[天日])'),
        ],

        'event_location': [
            re.compile(
                r'((?:[\u4e00-\u9fa5]{2,6}(?:省|自治区|市))'
                r'(?:[\u4e00-\u9fa5]{2,6}(?:市|州|区|县|镇))?)'
            ),
            re.compile(r'在([\u4e00-\u9fa5]{2,8}(?:市|县|区|镇|乡))'),
        ],

        'casualties': [
            re.compile(r'(?:造成|导致)(\d+)人(?:遇难|死亡)'),
            re.compile(r'(\d+)人(?:受伤|负伤)'),
            re.compile(r'(?:造成|导致)(\d+)人(?:遇难|死亡)[、，](\d+)人(?:受伤|负伤)'),
            re.compile(r'遇难(\d+)人[，、]受伤(\d+)人'),
            re.compile(r'(\d+)人(?:失踪|失联)'),
            re.compile(r'(\d+)人被埋'),
            re.compile(r'受灾人口[达约]?(\d+(?:\.\d+)?)万人'),
        ],

        'economic_loss': [
            re.compile(r'(?:直接)?经济损失[约达]?(\d+(?:\.\d+)?)[万亿]元'),
            re.compile(r'损失[约达]?(\d+(?:\.\d+)?)[万亿]元'),
            re.compile(r'经济损失(\d+(?:\.\d+)?)亿'),
        ],

        'response_level': [
            re.compile(r'启动(I{1,4}|Ⅰ{1,4}|一|二|三|四)级(?:应急)?响应'),
            re.compile(r'(I{1,4}级|Ⅰ级|Ⅱ级|Ⅲ级|Ⅳ级)(?:应急)?响应'),
            re.compile(r'启动(I级|II级|III级|IV级)'),
        ],

        'rescue_org': [
            re.compile(
                r'(国家应急管理部|中国红十字会|国家消防救援局|'
                r'解放军某部|武警部队|蓝天救援队|中国地震局|'
                r'国家气象局|民政部|当地消防支队|省应急管理厅|'
                r'国家防汛抗旱总指挥部|中国气象局|中国国际救援队|'
                r'公益救援联盟|绿舟救援队|国家减灾委员会|国家防总|'
                r'气象部门|水利专家|国务院)'
            ),
            re.compile(r'([\u4e00-\u9fa5]{2,6}(?:救援队|消防[队局]|部队))'),
        ],
    }

    def extract(self, text, title=''):
        """从文本中抽取所有信息点"""
        full_text = f'{title} {text}'
        results = {}
        for point, patterns in self.PATTERNS.items():
            matches = []
            for pat in patterns:
                for m in pat.finditer(full_text):
                    value = m.group(1) if m.lastindex else m.group()
                    if value and value not in matches:
                        matches.append(value)
            results[point] = self._post_process(point, matches, full_text)
        return results

    def _post_process(self, point, matches, text):
        """对抽取结果后处理"""
        if point == 'disaster_type':
            for dt in self.DISASTER_TYPES:
                if dt in text and dt not in matches:
                    matches.insert(0, dt)
            return matches[0] if matches else ''

        if point == 'casualties':
            return self._merge_casualties(matches, text)

        if point == 'economic_loss':
            return matches[0] + '亿元' if matches else ''

        if point == 'event_location':
            return matches[0] if matches else ''

        if point == 'rescue_org':
            return list(set(matches))[:5]

        if point == 'response_level':
            return matches[0] if matches else ''

        return matches[0] if matches else ''

    def _merge_casualties(self, matches, text):
        """合并伤亡信息"""
        parts = []
        dead_m = re.search(r'(\d+)人(?:遇难|死亡)', text)
        injured_m = re.search(r'(\d+)人(?:受伤|负伤)', text)
        missing_m = re.search(r'(\d+)人(?:失踪|失联)', text)

        if dead_m:
            parts.append(f'遇难{dead_m.group(1)}人')
        if injured_m:
            parts.append(f'受伤{injured_m.group(1)}人')
        if missing_m:
            parts.append(f'失踪{missing_m.group(1)}人')

        if not parts:
            affected_m = re.search(r'受灾人口[达约]?(\d+(?:\.\d+)?)万人', text)
            if affected_m:
                parts.append(f'受灾{affected_m.group(1)}万人')

        return '，'.join(parts) if parts else ''


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 基于命名实体识别的抽取器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NERExtractor:
    """基于 jieba 词性标注的命名实体识别抽取"""

    LOCATION_POS = {'ns', 'nt'}  # 地名、机构名
    TIME_POS = {'t'}             # 时间词
    ORG_POS = {'nt'}             # 机构名

    DISASTER_KEYWORDS = {
        '地震': '地震', '台风': '台风', '洪水': '洪水', '洪涝': '洪水',
        '滑坡': '山体滑坡', '泥石流': '山体滑坡', '火灾': '火灾',
        '干旱': '干旱', '暴雪': '暴雪', '暴风雪': '暴雪',
        '飓风': '台风', '龙卷风': '台风', '海啸': '海啸',
    }

    def extract(self, text, title=''):
        """使用 NER 进行信息抽取"""
        full_text = f'{title} {text}'
        words = pseg.cut(full_text)
        word_list = [(w.word, w.flag) for w in words]

        results = {
            'disaster_type': self._extract_disaster_type(word_list, full_text),
            'event_time': self._extract_time(word_list, full_text),
            'event_location': self._extract_location(word_list),
            'casualties': self._extract_casualties_ner(full_text),
            'economic_loss': self._extract_loss_ner(full_text),
            'response_level': self._extract_response_ner(full_text),
            'rescue_org': self._extract_org(word_list),
        }
        return results

    def _extract_disaster_type(self, word_list, text):
        for word, _ in word_list:
            if word in self.DISASTER_KEYWORDS:
                return self.DISASTER_KEYWORDS[word]
        return ''

    def _extract_time(self, word_list, text):
        time_pat = re.compile(r'\d{4}年\d{1,2}月\d{1,2}日')
        m = time_pat.search(text)
        if m:
            return m.group()
        for word, flag in word_list:
            if flag == 't' and len(word) >= 4:
                return word
        return ''

    def _extract_location(self, word_list):
        locations = []
        for word, flag in word_list:
            if flag == 'ns' and len(word) >= 2:
                locations.append(word)
        province = ''
        city = ''
        for loc in locations:
            if any(loc.endswith(s) for s in ['省', '自治区']):
                province = loc
            elif any(loc.endswith(s) for s in ['市', '州', '区', '县']):
                if not city:
                    city = loc
        return province + city if province else (city or (locations[0] if locations else ''))

    def _extract_org(self, word_list):
        orgs = []
        org_keywords = ['部', '局', '会', '队', '委', '厅', '院']
        for word, flag in word_list:
            if flag == 'nt' and len(word) >= 3:
                orgs.append(word)
            elif any(word.endswith(k) for k in org_keywords) and len(word) >= 4:
                orgs.append(word)
        return list(set(orgs))[:5]

    def _extract_casualties_ner(self, text):
        parts = []
        for pat, label in [
            (r'(\d+)\s*人\s*(?:遇难|死亡)', '遇难'),
            (r'(\d+)\s*人\s*(?:受伤|负伤)', '受伤'),
            (r'(\d+)\s*人\s*(?:失踪|失联)', '失踪'),
        ]:
            m = re.search(pat, text)
            if m:
                parts.append(f'{label}{m.group(1)}人')
        return '，'.join(parts) if parts else ''

    def _extract_loss_ner(self, text):
        m = re.search(r'经济损失[约达]?(\d+(?:\.\d+)?)\s*亿元', text)
        if m:
            return f'{m.group(1)}亿元'
        m = re.search(r'损失[约达]?(\d+(?:\.\d+)?)\s*[万亿]元', text)
        return f'{m.group(1)}亿元' if m else ''

    def _extract_response_ner(self, text):
        m = re.search(r'(I{1,4}级|Ⅰ级|Ⅱ级|Ⅲ级|Ⅳ级|IV级|III级|II级)', text)
        return m.group(1) if m else ''


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 基于依存句法分析的抽取器（简化版）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DependencyExtractor:
    """
    基于句法结构的信息抽取。
    利用动词-宾语、主语-谓语等句法关系，
    从灾害描述中精确定位信息点。
    """

    TRIGGER_VERBS = {
        'disaster': ['发生', '遭遇', '遭受', '袭击', '侵袭', '爆发', '暴发'],
        'casualty': ['造成', '导致', '致使', '已致', '共造成'],
        'response': ['启动', '宣布', '发布', '开启'],
        'rescue': ['派出', '调派', '出动', '调集', '组织', '调拨'],
        'loss': ['损失', '毁损', '摧毁', '倒塌'],
    }

    def extract(self, text, title=''):
        """基于句法结构的信息抽取"""
        full_text = f'{title} {text}'
        sentences = self._split_sentences(full_text)
        results = defaultdict(str)
        orgs = []

        for sent in sentences:
            pairs = [(w.word, w.flag) for w in pseg.cut(sent)]
            for i, (word, flag) in enumerate(pairs):
                context_after = ''.join(w for w, _ in pairs[i:i+10])

                if word in self.TRIGGER_VERBS['disaster']:
                    results['disaster_type'] = results['disaster_type'] or \
                        self._find_disaster_in_context(context_after, sent)

                if word in self.TRIGGER_VERBS['casualty']:
                    cas = self._extract_casualty_from_trigger(context_after)
                    if cas and len(cas) > len(results.get('casualties', '')):
                        results['casualties'] = cas

                if word in self.TRIGGER_VERBS['response']:
                    level = self._find_response_level(context_after)
                    if level:
                        results['response_level'] = level

                if word in self.TRIGGER_VERBS['rescue']:
                    org = self._find_org_before_trigger(pairs, i)
                    if org:
                        orgs.append(org)

            time_m = re.search(r'\d{4}年\d{1,2}月\d{1,2}日', sent)
            if time_m and not results['event_time']:
                results['event_time'] = time_m.group()

            loc_m = re.search(
                r'([\u4e00-\u9fa5]{2,6}(?:省|自治区)[\u4e00-\u9fa5]{2,6}(?:市|州|区|县))', sent
            )
            if loc_m and not results['event_location']:
                results['event_location'] = loc_m.group(1)

            loss_m = re.search(r'经济损失[约达]?(\d+(?:\.\d+)?)亿元', sent)
            if loss_m and not results['economic_loss']:
                results['economic_loss'] = f'{loss_m.group(1)}亿元'

        results['rescue_org'] = list(set(orgs))[:5]
        return dict(results)

    def _split_sentences(self, text):
        return re.split(r'[。！？；\n]', text)

    def _find_disaster_in_context(self, context, sent):
        types = ['地震', '台风', '洪水', '山体滑坡', '泥石流', '火灾',
                 '干旱', '暴雪', '暴风雪', '飓风']
        for t in types:
            if t in context or t in sent:
                return t
        return ''

    def _extract_casualty_from_trigger(self, context):
        parts = []
        for pat, label in [
            (r'(\d+)\s*人\s*(?:遇难|死亡)', '遇难'),
            (r'(\d+)\s*人\s*(?:受伤|负伤)', '受伤'),
            (r'(\d+)\s*人\s*(?:失踪|失联)', '失踪'),
        ]:
            m = re.search(pat, context)
            if m:
                parts.append(f'{label}{m.group(1)}人')
        return '，'.join(parts) if parts else ''

    def _find_response_level(self, context):
        m = re.search(r'(I{1,4}级|Ⅰ级|Ⅱ级|Ⅲ级|Ⅳ级|IV级|III级|II级)', context)
        return m.group(1) if m else ''

    def _find_org_before_trigger(self, words, trigger_idx):
        for j in range(max(0, trigger_idx - 5), trigger_idx):
            w, f = words[j]
            if f == 'nt' and len(w) >= 3:
                return w
        window = ''.join(w for w, _ in words[max(0, trigger_idx-5):trigger_idx])
        org_pat = re.compile(
            r'([\u4e00-\u9fa5]{2,10}(?:部|局|会|队|委|厅|总指挥部))'
        )
        m = org_pat.search(window)
        return m.group(1) if m else ''


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 集成抽取器（多策略投票融合）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EnsembleExtractor:
    """
    多策略集成抽取器：将 Regex / NER / Dependency 三种方法的
    抽取结果进行加权投票融合，输出最终结果并附带置信度。
    """

    WEIGHTS = {'regex': 0.4, 'ner': 0.35, 'dependency': 0.25}

    def __init__(self):
        self.regex_ext = RegexExtractor()
        self.ner_ext = NERExtractor()
        self.dep_ext = DependencyExtractor()

    def extract(self, text, title=''):
        """集成抽取"""
        regex_res = self.regex_ext.extract(text, title)
        ner_res = self.ner_ext.extract(text, title)
        dep_res = self.dep_ext.extract(text, title)

        all_results = {
            'regex': regex_res,
            'ner': ner_res,
            'dependency': dep_res,
        }

        fused = {}
        confidence = {}
        info_points = EXTRACTION_CONFIG['info_points']

        for point in info_points:
            candidates = {}
            for method, res in all_results.items():
                val = res.get(point, '')
                if isinstance(val, list):
                    val = '|'.join(val) if val else ''
                if val:
                    key = val.strip()
                    candidates[key] = candidates.get(key, 0) + self.WEIGHTS[method]

            if candidates:
                best = max(candidates, key=candidates.get)
                conf = candidates[best]
                if point == 'rescue_org':
                    all_orgs = set()
                    for method, res in all_results.items():
                        orgs = res.get('rescue_org', [])
                        if isinstance(orgs, list):
                            all_orgs.update(orgs)
                        elif orgs:
                            all_orgs.update(orgs.split('|'))
                    fused[point] = list(all_orgs)[:5]
                else:
                    fused[point] = best
                confidence[point] = round(min(conf / 1.0, 1.0), 2)
            else:
                fused[point] = '' if point != 'rescue_org' else []
                confidence[point] = 0.0

        return {
            'fused': fused,
            'confidence': confidence,
            'details': all_results,
        }


def create_extractor(method='ensemble'):
    """工厂函数：创建指定类型的抽取器"""
    extractors = {
        'regex': RegexExtractor,
        'ner': NERExtractor,
        'dependency': DependencyExtractor,
        'ensemble': EnsembleExtractor,
    }
    cls = extractors.get(method, EnsembleExtractor)
    return cls()
