"""
正则表达式信息抽取引擎
实现 ≥6 个自然灾害事件信息点的高精度正则匹配

信息点:
1. disaster_type   - 灾害类型
2. event_time      - 发生时间
3. event_location  - 发生地点
4. casualties      - 伤亡情况
5. economic_loss   - 经济损失
6. response_level  - 应急响应级别
7. rescue_info     - 救援信息
"""
import re
from datetime import datetime


# ============================================================
# 地点规范化 (提升 event_location 精度)
#   动机: NER/正则常把"家住XX""在XX""造成XX"等前缀, "XX县县"重复后缀,
#         或"部署重点地区/南方地区"等泛化区域表述误纳入地点。本函数:
#   1) 省份名作为可信锚点, 在任意位置抽取最长行政区划链;
#   2) 无省份时: 剥离引导词(在/家住/造成…) 后, 从串首锚定抽取行政区划链;
#   3) 剔除"等部分地区/灾区/危险区/南方地区"等泛化噪声尾巴;
#   4) 末校验: 结果必须完整匹配严格行政区划模式, 否则置空 (宁缺毋滥)。
# ============================================================
_LC_CJK = r'[一-龥]'
_LC_PROV = (r'(?:北京|天津|上海|重庆|黑龙江|内蒙古|河北|山西|辽宁|吉林|江苏|浙江|安徽|福建|江西|'
            r'山东|河南|湖北|湖南|广东|广西|海南|四川|贵州|云南|西藏|陕西|甘肃|青海|宁夏|新疆|台湾|香港|澳门)')
_LC_PSUF = r'(?:省|壮族自治区|回族自治区|维吾尔自治区|特别行政区|自治区|市)?'
_LC_ST = r'(?:' + _LC_CJK + r'{1,4}(?:自治州|市)|' + _LC_CJK + r'{1,3}州(?![县区]))'
_LC_CO = r'(?:' + _LC_CJK + r'{1,4}(?:县|区))'
_LC_TW = r'(?:' + _LC_CJK + r'{1,3}(?:镇|乡))'
_LC_VL = r'(?:' + _LC_CJK + r'{1,3}村)'
_LC_PROV_CHAIN = re.compile(_LC_PROV + _LC_PSUF + _LC_ST + r'?' + _LC_CO + r'?' + _LC_TW + r'?' + _LC_VL + r'?')
_LC_HEAD = re.compile(r'^(?:' + _LC_CJK + r'{2,3}(?:市|州)' + _LC_CO + r'?' + _LC_TW + r'?' + _LC_VL + r'?|'
                      + _LC_CJK + r'{2,4}(?:县|区)' + _LC_TW + r'?)')
_LC_ACCEPT = re.compile(
    r'^(?:' + _LC_PROV + _LC_PSUF + _LC_ST + r'?' + _LC_CO + r'?' + _LC_TW + r'?' + _LC_VL + r'?'
    r'|' + _LC_CJK + r'{2,4}(?:市|州)' + _LC_CO + r'?' + _LC_TW + r'?' + _LC_VL + r'?'
    r'|' + _LC_CJK + r'{2,5}(?:县|区)' + _LC_TW + r'?)$')
_LC_LEAD = re.compile(
    r'^(?:家住|位于|地处|坐落于?|发生在|出现在|分别在|主要在|经连线|连线|前往|赶赴|奔赴|来到|抵达|进入|'
    r'途经|途径|针对|关于|对于|当地|目前|近日|我国|我省|我市|造成|导致|涉及|波及|席卷|横扫|来袭|最大为|'
    r'自东向西|自西向东|尤其是|日夜间|院工作组赴|前期|心位于|经|在|于|从|的|了|日|时|中心)')
_LC_DOUBLE = re.compile(r'([省市州县区镇乡村])\1')
_LC_TAIL = re.compile(
    r'(?:(?:地震)?灾区|震区|落区|安全区|危险区?|等地?|(?:大?部分|相关|重点|有关|周边|附近)?(?:地区|区域|一带|流域|地带)'
    r'|(?:中)?[东西南北]+部|[东西南北]方|沿海|内陆)$')
_LC_BAD = ('部分', '大部', '重点地', '相关', '有关', '周边', '一带', '流域', '省份', '各地', '多地',
           '工作组', '中小河流', '灌区', '掌握', '督促', '指导', '调度', '部署', '转移', '安置', '做好',
           '应对', '抢险', '救援', '派出', '设区', '点对点', '会商', '主要城', '街镇', '景区', '城市')


def _lc_dangle(s):
    m = re.match(r'^(.*[省市州县区镇乡村])[^省市州县区镇乡村]{1,3}$', s)
    return m.group(1) if m else s


def _lc_normalize(s):
    prev = None
    while prev != s:
        prev = s; s = _LC_LEAD.sub('', s).strip()
    prev = None
    while prev != s:
        prev = s; s = _LC_TAIL.sub('', s).strip()
    return _lc_dangle(_LC_DOUBLE.sub(r'\1', s))


def _lc_valid(s):
    return bool(s) and not any(b in s for b in _LC_BAD) and bool(_LC_ACCEPT.fullmatch(s))


def clean_location(loc):
    """规范化地点串: 返回干净的行政区划, 无法识别为具体地点时返回 ''。"""
    if not loc:
        return ''
    s0 = _LC_DOUBLE.sub(r'\1', str(loc).strip())
    # A) 省份锚定 (省名无歧义, 可全串搜索取最长链)
    best = ''
    for m in _LC_PROV_CHAIN.finditer(s0):
        if len(m.group(0)) > len(best):
            best = m.group(0)
    if len(best) >= 3:
        c = _lc_normalize(best)
        if _lc_valid(c):
            return c
    # B) 无省份: 剥引导词后, 整串校验, 否则从串首锚定抽取
    t = s0
    prev = None
    while prev != t:
        prev = t; t = _LC_LEAD.sub('', t).strip()
    c = _lc_normalize(t)
    if _lc_valid(c):
        return c
    m = _LC_HEAD.match(t)
    if m:
        c = _lc_normalize(m.group(0))
        if _lc_valid(c):
            return c
    return ''


class RegexExtractor:
    """基于正则表达式的信息点抽取器"""

    # 灾害类型关键词词典
    DISASTER_TYPES = {
        '地震': ['地震', '震级', '余震', '震源', '震中', '里氏'],
        '台风': ['台风', '飓风', '热带风暴', '强热带风暴', '登陆'],
        '洪水': ['洪水', '洪涝', '洪灾', '山洪', '内涝', '溃堤', '决口'],
        '暴雨': ['暴雨', '强降雨', '特大暴雨', '大暴雨', '降水', '雷暴'],
        '山体滑坡': ['山体滑坡', '滑坡', '山体崩塌', '岩崩'],
        '泥石流': ['泥石流', '泥流'],
        '火灾': ['火灾', '森林大火', '森林火灾', '草原火灾', '过火面积'],
        '干旱': ['干旱', '旱灾', '旱情', '饮水困难'],
        '暴雪': ['暴雪', '雪灾', '低温雨雪', '冰冻'],
        '冰雹': ['冰雹', '风雹'],
        '海啸': ['海啸'],
        '龙卷风': ['龙卷风'],
    }

    # 时间正则
    TIME_PATTERNS = [
        (r'(\d{4}年\d{1,2}月\d{1,2}日\d{1,2}时\d{1,2}分)', 'full'),
        (r'(\d{4}年\d{1,2}月\d{1,2}日\d{1,2}时)', 'full'),
        (r'(\d{4}年\d{1,2}月\d{1,2}日)', 'date'),
        (r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', 'full'),
        (r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})', 'full'),
        (r'(\d{4}-\d{2}-\d{2})', 'date'),
        (r'(\d{1,2}月\d{1,2}日\d{1,2}时\d{1,2}分)', 'partial'),
        (r'(\d{1,2}月\d{1,2}日\d{1,2}时)', 'partial'),
        (r'(\d{1,2}月\d{1,2}日)', 'partial'),
        (r'(截至\d{1,2}月\d{1,2}日\d{1,2}时)', 'deadline'),
        (r'(\d{4}年\d{1,2}月)', 'month'),
    ]

    # 地点正则
    LOCATION_PATTERNS = [
        # 省+市+县
        r'([一-鿿]{2,5}省[一-鿿]{2,8}(?:市|州|地区)[一-鿿]{2,8}(?:县|区|市))',
        # 省+市
        r'([一-鿿]{2,5}省[一-鿿]{2,8}(?:市|州|地区))',
        # 直辖市+区
        r'((?:北京|上海|天津|重庆)市[一-鿿]{2,8}(?:区|县))',
        # 市+县
        r'([一-鿿]{2,8}(?:市|州)[一-鿿]{2,8}(?:县|区|市))',
        # 单独市
        r'([一-鿿]{2,8}(?:市|县|区|镇|乡|村))',
    ]

    # 伤亡正则
    CASUALTY_PATTERNS = [
        # 死亡
        r'(?:造成|已造成|导致|已导致|致|已有|目前已有)?(\d{1,5})人(?:死亡|遇难|罹难|丧生|不幸遇难)',
        r'(?:死亡|遇难|罹难|丧生)(?:人数|人员)?(?:达|为|约)?(\d{1,5})人',
        # 受伤
        r'(\d{1,5})人(?:受伤|不同程度受伤|重伤|轻伤)',
        r'(?:受伤|不同程度受伤)(?:人数|人员)?(?:达|为|约)?(\d{1,5})人',
        # 失踪
        r'(\d{1,5})人(?:失踪|失联|下落不明)',
        r'(?:失踪|失联|下落不明)(?:人数|人员)?(?:达|为|约)?(\d{1,5})人',
        # 综合伤亡
        r'(?:共|累计)?(?:造成|导致)?(\d{1,5})人(?:死亡|遇难)[,，、\s]*(\d{1,5})人(?:受伤)',
        r'(\d{1,3})死(\d{1,3})伤',
    ]

    # 经济损失正则
    ECONOMIC_PATTERNS = [
        r'(?:直接|间接)?经济损失(?:约|达|共计|约为|高达)?[\d,.，．]+(?:亿|万|千|百)?元',
        r'(?:造成|导致|造成直接|直接)(?:经济)?损失(?:约|达|共计|约为|高达)?[\d,.，．]+(?:亿|万|千|百)?元',
        r'[\d,.，．]+(?:亿|万)元(?:人民币)?(?:的)?(?:直接|间接)?(?:经济)?损失',
    ]

    # 响应级别: 锚定"响应"(可带 应急/救灾/防汛/防台风/抗旱/地质灾害 等前缀), 在其前后窗口内
    # 寻找"级别token"(中文一二三四 / 罗马ⅠⅡⅢⅣ / 拉丁IV 数字 + 级), 兼容
    # "防汛四级应急响应""国家四级救灾应急响应""防汛应急响应提升至三级""防台风一级响应"等表述。
    RESP_ANCHOR = re.compile(r'(?:应急|救灾|防汛|防台风?|抗旱|地质灾害|防汛抗旱)?响应')
    RESP_LEVEL_TOKEN = re.compile(r'([ⅠⅡⅢⅣIV一二三四]{1,3})级')
    RESP_LEVEL_MAP = {'一': 'Ⅰ', '二': 'Ⅱ', '三': 'Ⅲ', '四': 'Ⅳ',
                      'I': 'Ⅰ', 'II': 'Ⅱ', 'III': 'Ⅲ', 'IV': 'Ⅳ',
                      'Ⅰ': 'Ⅰ', 'Ⅱ': 'Ⅱ', 'Ⅲ': 'Ⅲ', 'Ⅳ': 'Ⅳ'}

    # 救援正则 — 精确匹配，避免噪声
    RESCUE_PATTERNS = [
        # 出动/派出/调派 X人
        r'(?:出动|派出|调派|投入|集结)(?:了|的)?[\d,，]+(?:名|人|人次|余)(?:消防|救援|应急|官兵|指战员|民兵|武警)',
        # 紧急调拨/运送物资
        r'(?:紧急)?(?:调拨|下拨|发放|运送|空运|抢运)(?:了)?[一-鿿\d,，]+(?:顶|件|套|吨|批)?(?:帐篷|棉被|食品|饮用水|救灾|救援|应急)(?:物资|装备)?',
        # 明确救援组织名 (全名优先)
        r'(国家消防救援局|中国红十字会|应急管理部|国家防灾减灾救灾委员会|国家防汛抗旱总指挥部|国务院抗震救灾指挥部)',
        r'([一-鿿]{2,6}(?:省|市|县))(?:消防救援总队|消防救援支队|消防救援大队|应急管理局|应急管理厅)',
        r'([一-鿿]{2,6}(?:消防|救援|救灾|应急|武警)(?:队|支队|总队|中心|大队))',
        # 启动响应（常与救援信息同时出现）
        r'启动(?:了)?(?:国家|省|市|县)?[一-鿿]+级(?:应急)?响应',
    ]

    # 非灾害信号（过滤误匹配）
    NON_DISASTER_SIGNALS = [
        '养生', '保健', '美容', '护肤', '减肥', '食疗', '中医',
        '股票', '基金', 'A股', '理财', '涨停', '跌停', '分红',
        '游戏', '手游', '电竞', '直播', '综艺', '明星',
    ]

    # 强灾害信号（必须命中一个才算真灾害）
    STRONG_DISASTER_SIGNALS = [
        '遇难', '伤亡', '失踪', '失联', '倒塌', '受损', '受灾',
        '应急响应', '救灾', '抢险', '搜救', '转移群众', '灾民',
        '震级', '震源', '震中', '余震',
        '台风', '登陆', '风力',
        '暴雨', '强降雨', '降水量', '水位',
        '经济损失', '直接损失', '受灾面积',
        '应急管理部', '消防救援', '国家防灾减灾',
    ]

    def extract(self, doc):
        """从文档中提取所有信息点"""
        title = doc.get('title', '')
        content = doc.get('content', '')
        full_text = f"{title}\n{content}"
        text_8000 = full_text[:8000]

        # 先验证是否是真灾害新闻
        dtype = self._extract_disaster_type(full_text)
        if dtype and not self._is_real_disaster(full_text, dtype):
            dtype = ''  # 过滤误匹配

        result = {
            'disaster_type': dtype,
            'event_time': self._extract_time(text_8000),
            'event_location': self._extract_location(text_8000),
            'casualties': self._extract_casualties(text_8000),
            'economic_loss': self._extract_economic_loss(text_8000),
            'response_level': self._extract_response_level(text_8000),
            'rescue_info': self._extract_rescue_info(text_8000),
        }
        return result

    def _is_real_disaster(self, text, dtype):
        """验证是否为真实灾害新闻（而非隐喻/列表提及）"""
        text_2000 = text[:2000]
        # 1. 命中非灾害信号 → 不是真灾害
        for kw in self.NON_DISASTER_SIGNALS:
            if kw in text_2000[:500]:  # 前500字出现非灾害词则可疑
                # 但如果有多个强信号，仍然可能是灾害
                strong = sum(1 for s in self.STRONG_DISASTER_SIGNALS if s in text_2000)
                if strong < 2:
                    return False
        # 2. 必须命中至少1个强灾害信号
        strong_hits = sum(1 for s in self.STRONG_DISASTER_SIGNALS if s in text_2000)
        if strong_hits == 0:
            return False
        return True

    def _extract_disaster_type(self, text):
        """识别灾害类型"""
        scores = {}
        text_1000 = text[:1000]
        for dtype, keywords in self.DISASTER_TYPES.items():
            score = sum(1 for kw in keywords if kw in text_1000)
            if score > 0:
                scores[dtype] = score
        if scores:
            return max(scores, key=scores.get)
        return ''

    def _extract_time(self, text):
        """提取事件发生时间"""
        times = []
        for pat, style in self.TIME_PATTERNS:
            matches = re.findall(pat, text)
            for m in matches:
                t = m if isinstance(m, str) else m[0]
                times.append((t, style))

        if not times:
            return ''

        # 优先返回完整时间
        for t, style in times:
            if style in ('full', 'date'):
                return t
        return times[0][0] if times else ''

    def _extract_location(self, text):
        """提取事件发生地点"""
        for pat in self.LOCATION_PATTERNS:
            matches = re.findall(pat, text)
            if matches:
                # 排除非灾害地的干扰
                for loc in matches:
                    loc_clean = loc if isinstance(loc, str) else loc[0]
                    # 过滤太泛的地点（如"人民日报"不是地点）
                    if any(kw in loc_clean for kw in ['日报', '通讯', '电视台', '记者']):
                        continue
                    if len(loc_clean) >= 4:
                        cleaned = clean_location(loc_clean)
                        if cleaned:
                            return cleaned
        return ''

    def _extract_casualties(self, text):
        """提取伤亡情况"""
        dead = injured = missing = 0

        # 提取死亡人数
        dead_patterns = [
            r'(?:造成|已造成|导致|已导致|致|已有)?(\d{1,5})人(?:死亡|遇难|罹难|丧生)',
            r'(?:死亡|遇难|罹难)(?:人数|人员)?(?:达|为|约)?(\d{1,5})人',
            r'(?:共|累计)?(?:造成|导致)?(\d{1,5})人(?:死亡|遇难)',
        ]
        for pat in dead_patterns:
            m = re.search(pat, text)
            if m:
                dead = max(dead, int(m.group(1)))
                break

        # 提取受伤人数
        inj_patterns = [
            r'(\d{1,5})人(?:受伤|不同程度受伤)',
            r'(?:受伤|不同程度受伤)(?:人数|人员)?(?:达|为|约)?(\d{1,5})人',
        ]
        for pat in inj_patterns:
            m = re.search(pat, text)
            if m:
                injured = max(injured, int(m.group(1)))
                break

        # 提取失踪人数
        miss_patterns = [
            r'(\d{1,5})人(?:失踪|失联|下落不明)',
            r'(?:失踪|失联)(?:人数|人员)?(?:达|为|约)?(\d{1,5})人',
        ]
        for pat in miss_patterns:
            m = re.search(pat, text)
            if m:
                missing = max(missing, int(m.group(1)))
                break

        if dead == 0 and injured == 0 and missing == 0:
            return ''

        parts = []
        if dead > 0:
            parts.append(f'遇难{dead}人')
        if injured > 0:
            parts.append(f'受伤{injured}人')
        if missing > 0:
            parts.append(f'失踪{missing}人')
        return '，'.join(parts)

    def _extract_economic_loss(self, text):
        """提取经济损失"""
        for pat in self.ECONOMIC_PATTERNS:
            m = re.search(pat, text)
            if m:
                return m.group(0)
        return ''

    def _extract_response_level(self, text):
        """提取应急响应级别 (窗口式: 锚定"响应" + 邻近级别token, 支持中文/罗马/拉丁数字)"""
        for m in self.RESP_ANCHOR.finditer(text):
            win = text[max(0, m.start() - 14): m.end() + 10]
            lm = self.RESP_LEVEL_TOKEN.search(win)
            if lm:
                roman = self.RESP_LEVEL_MAP.get(lm.group(1))
                if roman:
                    return f'{roman}级响应'
        return ''

    def _extract_rescue_info(self, text):
        """提取救援信息"""
        orgs = set()
        for pat in self.RESCUE_PATTERNS:
            matches = re.findall(pat, text)
            for m in matches:
                org = m if isinstance(m, str) else m[0]
                if len(org) >= 3:
                    orgs.add(org)

        # 提取救援人数
        personnel = ''
        m = re.search(r'(?:出动|派出|投入)(?:了)?([\d,，]+)(?:名|人|人次)', text)
        if m:
            personnel = f'投入{m.group(1)}人'

        result = list(orgs)[:8]
        if personnel:
            result.append(personnel)
        return result
