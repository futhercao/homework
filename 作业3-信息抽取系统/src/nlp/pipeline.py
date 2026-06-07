"""
NLP基础管道
- 中文分词 (jieba)
- 停用词过滤
- 词性标注
- 命名实体识别 (jieba 词性标注实体; LAC 可选增强)
- 分句
"""
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import NLP_CONFIG

import jieba

# === 停用词 ===
_STOPWORDS_ZH = set("""
的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你
会 着 没有 看 好 自己 这 他 她 它 们 那 些 什么 与 吗 但 被 此
这个 之 将 已 于 由 对 等 其 中 或 个 把 为 以 及 从 而 因
可以 能 所 使 同 如 又 并 才 还 比 怎么 让 多 时候 如果 只
其他 这些 两 三 四 五 六 七 八 九 十 百 千 万 亿
更 最 很 非常 十分 已经 正在 可能 应该 不是 没有 因为 所以
但是 虽然 如果 那么 通过 进行 可以 需要 已经 目前 其中 以及
关于 对于 根据 按照 作为 由于 然而 因此 总之 此外 另外 同时
不仅 而且 无论 尽管 只要 只有 即使 除了 包括 之间 之后 之前
各种 各类 相关 主要 重要 基本 一定 全部 所有 其他 每个 任何
这样 那样 非常 特别 一般 具有 属于 存在 发展 工作 开始 完成
提供 表示 认为 研究 分析 实现 利用 采用 建立 形成 产生 影响
提高 加强 促进 推动 支持 保证 满足 发挥 取得 保持 做到 具有
啊 哦 呢 吧 呀 嗯 噢 哈 嘿 喂 唉 哇 嘛
的话 来说 而言 看来 为了 以来 以后 以前 至今 从而
""".split())

# LAC 延迟加载
_lac = None


def get_lac():
    global _lac
    if _lac is None:
        try:
            from LAC import LAC
            _lac = LAC(mode='lac')
        except Exception:
            _lac = False
    return _lac if _lac is not False else None


class NLPPipeline:
    """统一NLP处理管道"""

    def __init__(self, config=None):
        self.config = config or NLP_CONFIG
        self.language = self.config.get('language', 'zh')

    def tokenize(self, text, remove_stopwords=True):
        """分词 — 使用搜索引擎模式提高召回"""
        text = self._normalize(text)
        tokens = list(jieba.cut_for_search(text))
        tokens = [t.strip() for t in tokens if self._valid(t)]
        if remove_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS_ZH]
        return tokens

    def pos_tag(self, tokens):
        """词性标注"""
        import jieba.posseg as pseg
        words = pseg.cut(' '.join(tokens))
        return [(w.word, w.flag) for w in words]

    def ner(self, text):
        """命名实体识别 (人名 PER / 地名 LOC / 机构 ORG / 时间 TIME)。

        开源 NER:
          - 若安装了 LAC(可选增强), 用百度 LAC 词法分析;
          - 否则用 jieba 词性标注 (jieba.posseg) 做实体识别:
            nr/nrt/nrfg→人名, ns→地名, nt→机构, t→时间。
        再用领域规则增强 (jieba 对机构名、完整行政区划、日期召回较弱, 与 NER 候选合并)。
        """
        entities = {'PER': [], 'LOC': [], 'ORG': [], 'TIME': []}

        # 领域规则 (高精度: 完整行政区划 / 机构 / 日期) 先入, 保证不被 [:20] 截断
        entities['TIME'] += re.findall(
            r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{2}-\d{2}|\d{1,2}月\d{1,2}日', text)
        entities['LOC'] += re.findall(r'[一-鿿]{2,}(?:省|市|县|区|镇|村)', text)
        entities['ORG'] += re.findall(r'[一-鿿]{2,}(?:局|部|委|中心|公司|基金会|协会|委员会)', text)

        # 开源 NER 追加候选 (LAC 若安装则用 LAC, 否则用 jieba 词性标注实体)
        lac = get_lac()
        if lac:
            words, tags = lac.run(text)
            for w, t in zip(words, tags):
                if t in entities:
                    entities[t].append(w)
        else:
            import jieba.posseg as pseg
            jmap = {'nr': 'PER', 'nrt': 'PER', 'nrfg': 'PER',
                    'ns': 'LOC', 'nt': 'ORG', 't': 'TIME'}
            for w in pseg.cut(text):
                cat = jmap.get(w.flag)
                if cat and len(w.word.strip()) >= 2:
                    entities[cat].append(w.word.strip())

        # 去重保序 + 限长
        for k in entities:
            entities[k] = list(dict.fromkeys(entities[k]))[:20]
        return entities

    def segment_sentences(self, text):
        """分句"""
        text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'[。！？!?\n]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def extract_keywords(self, text, top_k=10):
        """TF-IDF关键词提取"""
        from collections import Counter
        tokens = self.tokenize(text)
        counter = Counter(tokens)
        # 简单TF排序
        total = len(tokens) or 1
        keywords = [(w, c / total) for w, c in counter.most_common(top_k * 3) if len(w) >= 2]
        return keywords[:top_k]

    def get_snippet(self, text, query_tokens, max_length=250):
        """生成包含查询词的摘要片段"""
        text_lower = text.lower()
        best_pos = 0
        best_score = 0
        for i in range(0, max(1, len(text) - 50), 20):
            window = text_lower[i:i + max_length]
            score = sum(1 for t in query_tokens if t.lower() in window)
            if score > best_score:
                best_score = score
                best_pos = i
        start = max(0, best_pos)
        snippet = text[start:start + max_length]
        if start > 0:
            snippet = '...' + snippet
        if start + max_length < len(text):
            snippet = snippet + '...'
        return snippet

    def highlight(self, text, query_tokens):
        """HTML高亮查询词"""
        for token in sorted(query_tokens, key=len, reverse=True):
            if token and len(token) >= 1:
                pattern = re.compile(re.escape(token), re.IGNORECASE)
                text = pattern.sub(f'<mark>{token}</mark>', text)
        return text

    def _normalize(self, text):
        text = text.lower()
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'[^\w一-鿿\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _valid(self, token):
        token = token.strip()
        if not token or len(token) < self.config.get('min_token_length', 1):
            return False
        if re.match(r'^\d+$', token):
            return False
        return True


nlp = NLPPipeline()
