"""
集成抽取器 (创新点: 多抽取器加权投票 + 置信度)

动机:
    单一正则在结构化字段上精度高、但召回有盲区; 单一 NER 在地名/机构上召回好、
    但对"伤亡/损失/响应级别"等数值化字段无能为力。本模块在 RegexExtractor 之外
    新增一个 TriggerBasedExtractor (基于触发词 + 上下文窗口的抽取器), 与正则形成
    互补的"第二意见", 并融合 NER, 通过字段级加权投票得到更稳健的结果。

方法:
    candidates = { 'regex': ..., 'trigger': ..., 'ner': ... }   # 三路各给出 7 字段
    标量字段 (类型/时间/地点/伤亡/损失/响应):
        - 各路候选按字段级权重投票; 取值相同或互为子串者归为一组、权重累加;
        - 选累计权重最高的组为最终值; 置信度 = 该组权重 / 在场方法权重之和;
        - 多路一致 → 置信度趋近 1.0; 仅单路 → 置信度等于该方法权重 (标记低置信)。
    列表字段 (救援信息): 三路取并集去重 (召回优先)。

输出:
    与 RegexExtractor.extract 完全兼容的 7 字段 dict (可直接喂给 EventBuilder),
    另在 '_ensemble' 键下附每个字段的 {confidence, sources, candidates},
    供前端展示置信度、报告分析消融实验。
"""
import re
from src.extraction.regex_engine import RegexExtractor, clean_location
from src.nlp.pipeline import nlp


class TriggerBasedExtractor:
    """基于触发词 + 上下文窗口的抽取器 (与正则互补, 提升召回)"""

    DISASTER_TYPES = RegexExtractor.DISASTER_TYPES

    TIME_RE = re.compile(
        r'\d{4}年\d{1,2}月\d{1,2}日(?:\d{1,2}时(?:\d{1,2}分)?)?'
        r'|\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}(?::\d{2})?)?'
        r'|\d{1,2}月\d{1,2}日(?:\d{1,2}时)?')
    NUM_RE = re.compile(r'\d{1,5}')
    LEVEL_RE = re.compile(r'[ⅠⅡⅢⅣIV一二三四]{1,3}级')
    LEVEL_MAP = {'一': 'Ⅰ', '二': 'Ⅱ', '三': 'Ⅲ', '四': 'Ⅳ',
                 'I': 'Ⅰ', 'II': 'Ⅱ', 'III': 'Ⅲ', 'IV': 'Ⅳ'}
    MONEY_RE = re.compile(r'[\d,.，．]+(?:亿|万|千|百)?元')
    LOC_RE = re.compile(
        r'[一-鿿]{2,5}省[一-鿿]{2,8}(?:市|州|地区)(?:[一-鿿]{2,8}(?:县|区|市))?'
        r'|(?:北京|上海|天津|重庆)市[一-鿿]{2,8}(?:区|县)'
        r'|[一-鿿]{2,8}(?:市|州)[一-鿿]{2,8}(?:县|区)')

    CASUALTY_TRIGGERS = ['遇难', '死亡', '罹难', '丧生', '受伤', '失踪', '失联', '下落不明']
    LOSS_TRIGGERS = ['经济损失', '直接损失', '损失']
    RESPONSE_TRIGGERS = ['应急响应', '响应', '启动']
    LOC_TRIGGERS = ['位于', '震中', '发生在', '登陆', '发生']
    RESCUE_TRIGGERS = ['出动', '派出', '调派', '投入', '集结']
    RESCUE_ORG_RE = re.compile(
        r'(国家消防救援局|中国红十字会|应急管理部|国家防汛抗旱总指挥部)'
        r'|([一-鿿]{2,6}(?:消防救援总队|消防救援支队|应急管理局|应急管理厅|武警支队))')

    def _positions(self, text, words):
        pos = []
        for w in words:
            start = 0
            while True:
                i = text.find(w, start)
                if i < 0:
                    break
                pos.append(i)
                start = i + 1
        return pos

    def _windows(self, text, words, left=20, right=26):
        for p in self._positions(text, words):
            yield text[max(0, p - left): p + right]

    def extract(self, doc):
        title = doc.get('title', '')
        content = doc.get('content', '')
        text = f"{title}\n{content}"[:8000]
        return {
            'disaster_type': self._type(text),
            'event_time': self._time(text),
            'event_location': self._location(text),
            'casualties': self._casualties(text),
            'economic_loss': self._loss(text),
            'response_level': self._response(text),
            'rescue_info': self._rescue(text),
        }

    def _type(self, text):
        """全文词频投票 (区别于正则只看前1000字、首次命中)"""
        t = text[:2500]
        scores = {dt: sum(t.count(kw) for kw in kws) for dt, kws in self.DISASTER_TYPES.items()}
        scores = {k: v for k, v in scores.items() if v > 0}
        return max(scores, key=scores.get) if scores else ''

    def _time(self, text):
        """选离灾害关键词最近的时间表达 (邻近原则, 区别于正则的优先级原则)"""
        times = [(m.start(), m.group()) for m in self.TIME_RE.finditer(text)]
        if not times:
            return ''
        dis_pos = self._positions(text, [kw for kws in self.DISASTER_TYPES.values() for kw in kws])
        if dis_pos:
            return min(times, key=lambda tp: min(abs(tp[0] - d) for d in dis_pos))[1]
        return times[0][1]

    def _location(self, text):
        for win in self._windows(text, self.LOC_TRIGGERS, 4, 18):
            m = self.LOC_RE.search(win)
            if m and len(m.group()) >= 4:
                c = clean_location(m.group())
                if c:
                    return c
        m = self.LOC_RE.search(text)
        return clean_location(m.group()) if m else ''

    def _casualties(self, text):
        dead = injured = missing = 0
        for win in self._windows(text, self.CASUALTY_TRIGGERS, 12, 5):
            nums = [int(n) for n in self.NUM_RE.findall(win)]
            if not nums:
                continue
            n = max(nums)
            if any(k in win for k in ('遇难', '死亡', '罹难', '丧生')):
                dead = max(dead, n)
            elif '受伤' in win:
                injured = max(injured, n)
            elif any(k in win for k in ('失踪', '失联', '下落不明')):
                missing = max(missing, n)
        parts = []
        if dead:
            parts.append(f'遇难{dead}人')
        if injured:
            parts.append(f'受伤{injured}人')
        if missing:
            parts.append(f'失踪{missing}人')
        return '，'.join(parts)

    def _loss(self, text):
        for win in self._windows(text, self.LOSS_TRIGGERS, 2, 16):
            m = self.MONEY_RE.search(win)
            if m:
                return m.group()
        return ''

    def _response(self, text):
        for win in self._windows(text, self.RESPONSE_TRIGGERS, 12, 8):
            m = self.LEVEL_RE.search(win)
            if m:
                roman = self.LEVEL_MAP.get(m.group()[:-1], m.group()[:-1])  # 去"级"后归一化
                return f'{roman}级响应'
        return ''

    def _rescue(self, text):
        orgs = []
        for m in self.RESCUE_ORG_RE.finditer(text):
            org = m.group(0)
            if org and org not in orgs:
                orgs.append(org)
        # 救援人数
        m = re.search(r'(?:出动|派出|投入)(?:了)?([\d,，]+)(?:名|人|人次)', text)
        if m:
            orgs.append(f'投入{m.group(1)}人')
        return orgs[:8]


class EnsembleExtractor:
    """加权投票集成: regex + trigger + ner"""

    SCALAR_FIELDS = ['disaster_type', 'event_time', 'event_location',
                     'casualties', 'economic_loss', 'response_level']

    # 字段级方法权重 (regex 精度高基准 0.6; trigger 召回补充; ner 仅地名/机构)
    WEIGHTS = {
        'disaster_type':  {'regex': 0.6, 'trigger': 0.4},
        'event_time':     {'regex': 0.5, 'trigger': 0.5},
        'event_location': {'regex': 0.45, 'trigger': 0.30, 'ner': 0.40},
        'casualties':     {'regex': 0.6, 'trigger': 0.4},
        'economic_loss':  {'regex': 0.6, 'trigger': 0.4},
        'response_level': {'regex': 0.6, 'trigger': 0.4},
    }

    def __init__(self):
        self.regex = RegexExtractor()
        self.trigger = TriggerBasedExtractor()

    NOISE_PREFIX = re.compile(r'^(当日|当天|今日|昨日|日前|日从|今天|昨天|近日|目前|连日来|随后|该|此|其|从|向|在|于)')

    def _clean_org(self, s):
        """剥离 NER 边界误纳的时间/连接词前缀 (如 '当日应急管理部'→'应急管理部')"""
        s = str(s).strip()
        prev = None
        while prev != s:
            prev = s
            s = self.NOISE_PREFIX.sub('', s).strip()
        return s

    def extract(self, doc):
        cand = {
            'regex': self.regex.extract(doc),
            'trigger': self.trigger.extract(doc),
            'ner': self._ner_fields(doc),
        }
        final, detail = {}, {}
        for field in self.SCALAR_FIELDS:
            value, conf, sources = self._vote(field, cand)
            final[field] = value
            detail[field] = {
                'confidence': round(conf, 3),
                'sources': sources,
                'candidates': {m: cand[m].get(field, '') for m in cand if cand[m].get(field)},
            }
        # 列表字段(救援信息): 清洗NER噪声前缀 + 并集 + 包含去重
        merged = []
        for m in ('regex', 'trigger', 'ner'):
            for v in (cand[m].get('rescue_info') or []):
                v = self._clean_org(v)
                if not v or len(v) < 3:
                    continue
                dup = False
                for i, kept in enumerate(merged):
                    if v in kept or kept in v:
                        if len(v) < len(kept):   # 保留更短的规范名, 舍弃含噪长串
                            merged[i] = v
                        dup = True
                        break
                if not dup:
                    merged.append(v)
        final['rescue_info'] = merged[:10]
        detail['rescue_info'] = {
            'confidence': 1.0 if len(merged) else 0.0,
            'sources': [m for m in ('regex', 'trigger', 'ner') if cand[m].get('rescue_info')],
            'candidates': {m: cand[m].get('rescue_info', []) for m in cand if cand[m].get('rescue_info')},
        }
        final['_ensemble'] = detail
        final['event_location'] = clean_location(final.get('event_location', ''))
        return final

    def _ner_fields(self, doc):
        """NER 仅补地名 + 救援机构"""
        text = f"{doc.get('title', '')}\n{doc.get('content', '')}"[:5000]
        try:
            ents = nlp.ner(text)
        except Exception:
            ents = {}
        locs = [l for l in ents.get('LOC', []) if len(l) >= 3 and not l.endswith('省')]
        loc = clean_location(max(locs, key=len) if locs else '')
        orgs = [o for o in ents.get('ORG', [])
                if any(k in o for k in ['消防', '救援', '应急', '救灾', '红十字',
                                        '武警', '公安', '部队', '支队', '总队', '管理局'])]
        return {'event_location': loc, 'rescue_info': orgs}

    def _norm(self, field, value):
        s = str(value).strip()
        if field == 'response_level':
            roman = re.sub(r'[^ⅠⅡⅢⅣIV]', '', s)
            return roman.replace('III', 'Ⅲ').replace('IV', 'Ⅳ').replace('II', 'Ⅱ').replace('I', 'Ⅰ')
        return s

    def _vote(self, field, cand):
        weights = self.WEIGHTS[field]
        present = {m: cand[m].get(field, '') for m in weights if cand[m].get(field, '')}
        if not present:
            return '', 0.0, []

        # 分组: 取值相同或互为子串者合并 (捕捉部分一致)
        groups = []   # [{'key','weight','methods','value'}]
        for m, val in present.items():
            nv = self._norm(field, val)
            placed = False
            for g in groups:
                if nv and g['key'] and (nv in g['key'] or g['key'] in nv):
                    g['weight'] += weights[m]
                    g['methods'].append(m)
                    if len(str(val)) > len(str(g['value'])):
                        g['value'], g['key'] = val, nv
                    placed = True
                    break
            if not placed:
                groups.append({'key': nv, 'weight': weights[m], 'methods': [m], 'value': val})

        best = max(groups, key=lambda g: g['weight'])
        total = sum(weights[m] for m in present)
        conf = best['weight'] / total if total else 0.0
        return best['value'], conf, best['methods']
