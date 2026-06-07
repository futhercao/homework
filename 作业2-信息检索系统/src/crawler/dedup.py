"""
SimHash 内容去重
防止同一篇文章从不同来源重复爬取
"""
import re
from simhash import Simhash


class ContentDeduplicator:
    """基于SimHash的内容去重器"""

    def __init__(self, threshold=3):
        self.threshold = threshold    # 汉明距离阈值
        self.fingerprints = []        # [(doc_id, simhash_value), ...]

    def _get_features(self, text):
        """提取文本特征用于SimHash"""
        # 清理后取2-gram
        text = re.sub(r'\s+', '', text)
        features = []
        for i in range(len(text) - 1):
            features.append(text[i:i + 2])
        return features

    def is_duplicate(self, text, doc_id=''):
        """判断文本是否重复，返回 (is_dup, similar_doc_id)"""
        if len(text) < 100:
            return False, ''

        features = self._get_features(text[:3000])
        fingerprint = Simhash(features)

        for existing_id, existing_fp in self.fingerprints:
            distance = fingerprint.distance(existing_fp)
            if distance <= self.threshold:
                return True, existing_id

        self.fingerprints.append((doc_id, fingerprint))
        return False, ''

    def add(self, text, doc_id):
        """添加文本指纹"""
        if len(text) < 100:
            return
        features = self._get_features(text[:3000])
        self.fingerprints.append((doc_id, Simhash(features)))

    def __len__(self):
        return len(self.fingerprints)
