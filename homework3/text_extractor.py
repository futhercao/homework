"""
文本信息抽取模块
实现三种抽取策略：
1. RegexExtractor      — 基于正则表达式的模式匹配（课程基本要求）
2. NERExtractor        — 基于命名实体识别的抽取（jieba posseg）
3. TriggerBasedExtractor — 基于触发词-上下文窗口的规则抽取
4. EnsembleExtractor   — 多策略融合的集成抽取器

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

    DISASTER_TYPES = [
        '地震', '台风', '洪水', '暴雨', '洪涝', '山体滑坡', '泥石流',
        '火灾', '森林火灾', '草原火灾', '山林火灾', '干旱',
        '暴雪', '飓风', '龙卷风', '冰雹', '海啸', '崩塌', '塌方',
        '低温雨雪冰冻', '风雹', '沙尘暴', '雪灾',
    ]

    PATTERNS = {
        'disaster_type': [
            re.compile(r'发生(?:了)?(?:里氏)?[\d.]+级(地震)'),
            re.compile(r'(台风|飓风)(?:"|「)?[一-龥]+(?:"|」)?'),
            re.compile(r'(洪水|洪涝|特大洪水|山洪)'),
            re.compile(r'(山体滑坡|泥石流|崩塌|塌方)'),
            re.compile(r'(森林火灾|森林大火|草原火灾|山林火灾|山林大火|火灾)'),
            re.compile(r'(?:严重|特大)?(干旱|旱灾)'),
            re.compile(r'(暴雪|暴风雪|雪灾|低温雨雪冰冻)'),
            re.compile(r'(冰雹|风雹|沙尘暴|龙卷风|海啸)'),
        ],

        'event_time': [
            # 完整日期+可选时分
            re.compile(r'(\d{4}年\d{1,2}月\d{1,2}日(?:\d{1,2}时\d{1,2}分)?)'),
            # 月-日（无年份），带可选时间修饰词
            re.compile(r'(\d{1,2}月\d{1,2}日(?:凌晨|早上|上午|中午|下午|傍晚|晚上|深夜|午夜|夜间|\d{1,2}时\d{1,2}分)?)'),
            # ISO 格式
            re.compile(r'(\d{4}-\d{1,2}-\d{1,2})'),
            # 斜杠格式
            re.compile(r'(\d{4}/\d{1,2}/\d{1,2})'),
            # "今年"前缀
            re.compile(r'(今年\d{1,2}月\d{1,2}日)'),
            # 相对时间（昨天/前天/今天）
            re.compile(r'(昨[天日]|前[天日]|今[天日])'),
        ],

        'event_location': [
            # 省/自治区/直辖市 + 市/自治州/地区/盟/县/区（完整层级）
            re.compile(
                r'([一-龥]{2,10}'
                r'(?:省|自治区|壮族自治区|回族自治区|维吾尔自治区|特别行政区)'
                r'[一-龥]{2,15}'
                r'(?:市|自治州|州|地区|盟|县|区|镇))'
            ),
            # 只到省级
            re.compile(
                r'([一-龥]{2,12}'
                r'(?:省|自治区|壮族自治区|回族自治区|维吾尔自治区|特别行政区|市))'
            ),
            # 在/位于 X市/X县/X区（常见报道句式）
            re.compile(r'(?:在|位于|地处)([一-龥]{2,8}(?:市|自治州|州|县|区|镇|乡|村))'),
            # 外国地名（常见灾害多发国家）
            re.compile(r'(菲律宾|日本|智利|印度尼西亚|缅甸|越南|泰国|尼泊尔|'
                      r'美国|墨西哥|土耳其|伊朗|巴基斯坦|阿富汗|'
                      r'印度|孟加拉国|马来西亚|韩国|朝鲜|俄罗斯)'),
        ],

        'casualties': [
            re.compile(r'(?:造成|导致|已致|共造成|致使)(\d+)人(?:遇难|死亡)'),
            re.compile(r'(?:造成|导致|已致|共造成|致使)(\d+)人(?:受伤|负伤)'),
            re.compile(r'(?:造成|导致|已致)(\d+)人(?:遇难|死亡)[、，,](\d+)人(?:受伤|负伤)'),
            re.compile(r'遇难(\d+)人[，、,]受伤(\d+)人'),
            re.compile(r'(\d+)人(?:遇难|死亡)'),
            re.compile(r'(\d+)人(?:受伤|负伤)'),
            re.compile(r'(\d+)人(?:失踪|失联)'),
            re.compile(r'(\d+)人被埋'),
            re.compile(r'(\d+)人被困'),
            re.compile(r'受灾人口[达约]?(\d+(?:\.\d+)?)万?人'),
            # 转移安置
            re.compile(r'(?:紧急)?转移(?:安置)?(?:群众|人员|受灾群众)?[约达]?(\d+(?:\.\d+)?)万?人'),
        ],

        'economic_loss': [
            re.compile(r'(?:直接)?经济损失[约达]?(\d+(?:\.\d+)?)[万亿]元'),
            re.compile(r'损失[约达]?(\d+(?:\.\d+)?)[万亿]元'),
            re.compile(r'经济损失(?:约|达)?(\d+(?:\.\d+)?)亿'),
        ],

        'response_level': [
            re.compile(r'启动\s*(I{1,4}级|Ⅰ{1,4}级|Ⅳ级|Ⅲ级|Ⅱ级|Ⅰ级|IV级|III级|II级)\s*(?:应急)?响应'),
            re.compile(r'启动\s*(四|三|二|一)级\s*(?:应急)?响应'),
            re.compile(r'国家(?:地震|防汛|防台风|救灾|自然灾害)(?:灾害)?(I{1,4}级|Ⅰ级|Ⅳ级|Ⅲ级|Ⅱ级|四|三|二|一)'),
            re.compile(r'(?<!\w)(I{1,3}级|Ⅰ级|Ⅱ级|Ⅲ级|Ⅳ级)(?:\s*应急)?响应'),
        ],

        'rescue_org': [
            re.compile(
                r'(国家应急管理部|应急管理部|中国红十字会|国家消防救援局|'
                r'国家防汛抗旱总指挥部|国家防总|中国地震局|中国气象局|'
                r'民政部|国家减灾委员会|国家减灾委|国务院抗震救灾指挥部|'
                r'自然资源部|水利部|中国国际救援队|'
                r'蓝天救援队|绿舟救援队|公益救援联盟|蓝豹救援队|'
                r'公羊救援队|曙光救援队|山地救援队)'
            ),
            re.compile(
                r'([一-龥]{2,6}'
                r'(?:省|市|区|自治区)'
                r'(?:消防救援总队|消防救援支队|消防救援大队|'
                r'应急管理厅|应急管理局|消防总队|消防支队|'
                r'消防救援局|公安消防总队|森林消防总队|'
                r'地震局|气象局|红十字会))'
            ),
            re.compile(r'([一-龥]{2,8}(?:救援队|消防[队局支队]|应急队|搜救队))'),
            re.compile(
                r'(解放军[一-龥]{2,8}|武警[一-龥]{2,8}|'
                r'[一-龥]{2,6}部队|[一-龥]{2,6}民兵)'
            ),
        ],
    }

    # 常见的地名提取前缀（用于清理）
    LOCATION_PREFIX_PATTERNS = [
        re.compile(r'^(?:记者来到|记者又来到了|记者在|记者赴|来到了|来到|位于'
                   r'|赶赴|抵达|进入|前往|到达|在)'),
    ]

    def extract(self, text, title=''):
        """从文本中抽取所有信息点"""
        full_text = f'{title} {text}'
        results = {}
        for point, patterns in self.PATTERNS.items():
            matches = []
            for pat in patterns:
                for m in pat.finditer(full_text):
                    if m.lastindex and m.lastindex >= 2:
                        parts = [g for g in m.groups() if g]
                        value = ''.join(parts)
                    elif m.lastindex:
                        value = m.group(1)
                    else:
                        value = m.group()
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

        if point == 'event_time':
            if not matches:
                return ''
            best = matches[0]
            # 如果有多个日期，优先选择完整的"年月日"格式
            full_dates = [m for m in matches if re.match(r'\d{4}年', m)]
            if full_dates:
                best = full_dates[0]
            # 清洗：去掉时间修饰词，只保留日期
            best = re.sub(r'(凌晨|早上|上午|中午|下午|傍晚|晚上|深夜|午夜|夜间).*$', '', best)
            best = re.sub(r'\d{1,2}时\d{1,2}分?', '', best).strip()
            return best

        if point == 'event_location':
            if not matches:
                return ''
            best = matches[0]
            # 清理常见前缀（如"记者来到"）
            for pat in self.LOCATION_PREFIX_PATTERNS:
                best = pat.sub('', best)
            return best

        if point == 'casualties':
            return self._merge_casualties(matches, text)

        if point == 'economic_loss':
            if matches:
                val = matches[0]
                if not val.endswith('元'):
                    val += '亿元' if '亿' in text and '万' not in val else '元'
                return val
            return ''

        if point == 'rescue_org':
            return list(set(matches))[:5]

        if point == 'response_level':
            if matches:
                val = matches[0]
                cn_to_roman = {'一': 'I级', '二': 'II级', '三': 'III级', '四': 'IV级'}
                if val in cn_to_roman:
                    val = cn_to_roman[val]
                return val
            return ''

        return matches[0] if matches else ''

    def _merge_casualties(self, matches, text):
        """合并伤亡信息"""
        parts = []
        found_labels = set()

        queries = [
            (r'(\d+)\s*人\s*(?:遇难|死亡)', '遇难'),
            (r'(\d+)\s*人\s*(?:受伤|负伤)', '受伤'),
            (r'(\d+)\s*人\s*(?:失踪|失联)', '失踪'),
            (r'(\d+)\s*人\s*被埋', '被埋'),
            (r'(\d+)\s*人\s*被困', '被困'),
            (r'(?:紧急)?转移(?:安置)?(?:群众|人员|受灾群众)?[约达]?(\d+(?:\.\d+)?)万?人', '转移'),
            (r'受灾人口[达约]?(\d+(?:\.\d+)?)万?人', '受灾'),
        ]

        for pat, label in queries:
            if label in found_labels:
                continue
            m = re.search(pat, text)
            if m:
                num = m.group(1)
                if '万' in text[m.start():m.end()]:
                    num += '万'
                parts.append(f'{label}{num}人')
                found_labels.add(label)

        return '，'.join(parts) if parts else ''


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 基于命名实体识别的抽取器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NERExtractor:
    """基于 jieba 词性标注的命名实体识别抽取"""

    LOCATION_POS = {'ns', 'nt'}  # 地名、机构名
    TIME_POS = {'t'}             # 时间词
    ORG_POS = {'nt'}             # 机构名

    # 修复：泥沙流和山体滑坡是不同的灾害类型
    DISASTER_KEYWORDS = {
        '地震': '地震', '台风': '台风', '洪水': '洪水', '洪涝': '洪水',
        '山洪': '洪水', '暴雨': '暴雨',
        '泥石流': '泥石流',       # 保持原类型，不映射到山体滑坡
        '山体滑坡': '山体滑坡',
        '滑坡': '山体滑坡',       # 通用"滑坡"可归入山体滑坡
        '崩塌': '山体滑坡',
        '火灾': '火灾', '森林火灾': '火灾',
        '干旱': '干旱', '旱灾': '干旱',
        '暴雪': '暴雪', '暴风雪': '暴雪', '雪灾': '暴雪',
        '飓风': '台风', '龙卷风': '台风', '海啸': '海啸',
        '冰雹': '冰雹',
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
        priority_order = ['泥石流', '山体滑坡', '森林火灾', '洪水', '暴雨',
                         '地震', '台风', '干旱', '暴雪', '海啸', '冰雹']
        found = None
        for word, _ in word_list:
            if word in self.DISASTER_KEYWORDS:
                mapped = self.DISASTER_KEYWORDS[word]
                if found is None or (
                    found in priority_order and mapped in priority_order and
                    priority_order.index(mapped) < priority_order.index(found)
                ):
                    found = mapped
        if not found:
            for dt in priority_order:
                if dt in text:
                    found = dt
                    break
        return found if found else ''

    def _extract_time(self, word_list, text):
        # 优先完整日期
        time_pat = re.compile(r'\d{4}年\d{1,2}月\d{1,2}日')
        m = time_pat.search(text)
        if m:
            return m.group()
        # 次选月-日
        md_pat = re.compile(r'(\d{1,2}月\d{1,2}日)')
        m = md_pat.search(text)
        if m:
            return m.group(1)
        # 词性标注回退
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
        m = re.search(r'(?:直接)?经济损失[约达]?(\d+(?:\.\d+)?)\s*亿?元?', text)
        if m:
            val = m.group(1)
            if '亿' in text[m.start():m.end()]:
                return f'{val}亿元'
            if '万' in text[m.start():m.end()]:
                return f'{val}万元'
            return f'{val}元'
        return ''

    def _extract_response_ner(self, text):
        m = re.search(
            r'(?:启动\s*)?'
            r'(I{1,4}级|Ⅰ{1,4}级|Ⅳ级|Ⅲ级|Ⅱ级|Ⅰ级|IV级|III级|II级|'
            r'四级|三级|二级|一级)',
            text
        )
        if m:
            val = m.group(1)
            cn_to_roman = {'一级': 'I级', '二级': 'II级', '三级': 'III级', '四级': 'IV级'}
            return cn_to_roman.get(val, val)
        return ''


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 基于触发词-上下文窗口的规则抽取器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TriggerBasedExtractor:
    """
    基于触发词-上下文窗口的信息抽取。

    核心方法：定义灾害事件相关的触发动词（如"发生""造成""启动""派出"等），
    在句子中定位触发词后，利用上下文窗口（前5词/后10词）结合正则表达式
    抽取对应的信息点。

    注意：本方法使用触发词 + 上下文窗口 + 正则表达式的组合策略，
    未使用完整的依存句法分析器。
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

            if not results['event_time']:
                time_m = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日)', sent)
                if not time_m:
                    time_m = re.search(r'(\d{1,2}月\d{1,2}日)', sent)
                if time_m:
                    results['event_time'] = time_m.group(1)

            if not results['event_location']:
                loc_m = re.search(
                    r'([一-龥]{2,10}'
                    r'(?:省|自治区|壮族自治区|回族自治区|维吾尔自治区)'
                    r'[一-龥]{2,10}'
                    r'(?:市|州|地区|县|区))',
                    sent
                )
                if loc_m:
                    results['event_location'] = loc_m.group(1)

            if not results['economic_loss']:
                loss_m = re.search(r'经济损失[约达]?(\d+(?:\.\d+)?)亿元', sent)
                if loss_m:
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
        m = re.search(
            r'(I{1,4}级|Ⅰ级|Ⅱ级|Ⅲ级|Ⅳ级|IV级|III级|II级|四级|三级|二级|一级)',
            context
        )
        return m.group(1) if m else ''

    def _find_org_before_trigger(self, words, trigger_idx):
        for j in range(max(0, trigger_idx - 5), trigger_idx):
            w, f = words[j]
            if f == 'nt' and len(w) >= 3:
                return w
        window = ''.join(w for w, _ in words[max(0, trigger_idx-5):trigger_idx])
        org_pat = re.compile(
            r'([一-龥]{2,10}(?:部|局|会|队|委|厅|总指挥部))'
        )
        m = org_pat.search(window)
        return m.group(1) if m else ''


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 集成抽取器（多策略投票融合）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EnsembleExtractor:
    """
    多策略集成抽取器：将 Regex / NER / TriggerBased 三种方法的
    抽取结果进行加权投票融合，输出最终结果并附带置信度。
    包含非灾害新闻过滤：无灾害触发词时返回空结果以避免误抽取。
    """

    WEIGHTS = {'regex': 0.4, 'ner': 0.35, 'trigger': 0.25}

    DISASTER_TRIGGERS = [
        '地震', '台风', '飓风', '洪水', '洪涝', '暴雨', '泥石流',
        '山体滑坡', '滑坡', '崩塌', '火灾', '森林大火', '山林大火',
        '干旱', '暴雪', '冰雹', '海啸', '火山', '龙卷风',
        '遇难', '受灾', '灾情', '救灾', '抢险', '防汛', '防灾',
        '应急响应', '启动.*级.*应急', '救援队', '灾后',
        '塌方', '溃坝', '决堤', '堰塞湖',
    ]

    def __init__(self):
        self.regex_ext = RegexExtractor()
        self.ner_ext = NERExtractor()
        self.trigger_ext = TriggerBasedExtractor()
        self._disaster_pattern = re.compile('|'.join(self.DISASTER_TRIGGERS))

    def _is_disaster_article(self, text, title=''):
        """检查文档是否涉及自然灾害事件"""
        combined = (title + ' ' + text[:3000]) if title else text[:3000]
        return bool(self._disaster_pattern.search(combined))

    def extract(self, text, title=''):
        """集成抽取，非灾害新闻返回空结果。text 可以是字符串或文档 dict。"""
        if isinstance(text, dict):
            title = text.get('title', title)
            text = text.get('content', '')

        if not self._is_disaster_article(text, title):
            empty = {p: '' if p != 'rescue_org' else [] for p in EXTRACTION_CONFIG['info_points']}
            return {
                'fused': dict(empty),
                'confidence': {p: 0.0 for p in EXTRACTION_CONFIG['info_points']},
                'details': {'regex': dict(empty), 'ner': dict(empty), 'trigger': dict(empty)},
            }

        regex_res = self.regex_ext.extract(text, title)
        ner_res = self.ner_ext.extract(text, title)
        trigger_res = self.trigger_ext.extract(text, title)

        all_results = {
            'regex': regex_res,
            'ner': ner_res,
            'trigger': trigger_res,
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
        'trigger': TriggerBasedExtractor,
        'dependency': TriggerBasedExtractor,
        'ensemble': EnsembleExtractor,
    }
    cls = extractors.get(method, EnsembleExtractor)
    return cls()
