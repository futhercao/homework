"""
文本预处理模块
- 中文分词（jieba）
- 停用词过滤
- 文本归一化
"""
import re

import jieba

from config import PREPROCESS_CONFIG

STOPWORDS_ZH = set("""
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

STOPWORDS_EN = set("""
a an the and or but in on at to for of with by from as is was were be been
being have has had do does did will would shall should can could may might
must need not no nor so if then than too very just about also back been
before both each few here how its let more most much now only other our
out over own same she some such tell that their them there these they this
those through under up us want we what when where which while who whom why
all any are between both during further into more most no such through
until we after again against am because being both did does doing down
during few for having he her hers herself him himself his itself me
mine my myself off once only ours ourselves over own re same should
then theirs themselves these through under until us very wasn we were
what when where which while who whom why will with you your yours
yourself yourselves
""".split())


class TextPreprocessor:
    """文本预处理器，支持中文和英文"""

    def __init__(self, config=None):
        self.config = config or PREPROCESS_CONFIG
        self.language = self.config.get('language', 'zh')
        self.stopwords = STOPWORDS_ZH if self.language == 'zh' else STOPWORDS_EN

    def tokenize(self, text):
        """分词：中文用jieba，英文按空格"""
        text = self._normalize(text)
        if self.language == 'zh':
            tokens = list(jieba.cut(text))
        else:
            tokens = text.split()
        return [t for t in tokens if self._is_valid_token(t)]

    def tokenize_for_index(self, text):
        """为索引构建分词，同时去除停用词"""
        tokens = self.tokenize(text)
        if self.config.get('remove_stopwords', True):
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def tokenize_query(self, query):
        """对查询字符串分词"""
        return self.tokenize_for_index(query)

    def _normalize(self, text):
        """文本归一化：统一标点、去除特殊字符、转小写"""
        text = text.lower()
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'[^\w\u4e00-\u9fff\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _is_valid_token(self, token):
        """检查token有效性"""
        token = token.strip()
        if not token:
            return False
        min_len = self.config.get('min_term_length', 1)
        if len(token) < min_len:
            return False
        if re.match(r'^\d+$', token):
            return False
        if re.match(r'^\s+$', token):
            return False
        return True

    def get_snippet(self, text, query_tokens, max_length=200):
        """
        生成包含查询词的文本摘要片段，高亮匹配词。
        用于搜索结果展示。
        """
        text_lower = text.lower()
        best_pos = 0
        best_score = 0

        for i in range(0, len(text) - 50, 20):
            window = text_lower[i:i + max_length]
            score = sum(1 for t in query_tokens if t in window)
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
        """在文本中高亮显示查询词（用HTML标签包裹）"""
        for token in query_tokens:
            if token:
                pattern = re.compile(re.escape(token), re.IGNORECASE)
                text = pattern.sub(f'<mark>{token}</mark>', text)
        return text
