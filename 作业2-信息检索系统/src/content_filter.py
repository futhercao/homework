"""内容图过滤 —— 复用 build_multimodal 阶段 VLM 视觉质检的结论(data/vectors/vlm_verdicts.json),
在「文档详情页配图展示」与「检索结果配图计数」中统一隐藏 VLM 判定的无关图
(logo / 广告横幅 / 二维码 / 界面截图 / 纯文字快讯等)。

采用**黑名单(per-image fail-open)语义**:只剔除被 VLM 明确判为「非内容图」(keep=false)者,
未判定的图一律保留。相比白名单(只保留 keep 集合内者),黑名单对**部分覆盖 / 过期**的
verdicts 稳健 —— 换语料后旧缓存未覆盖新图时,不会把新图误判为噪声而全部隐藏;
verdicts 文件缺失时同样回退为不过滤(全部保留)。口径与 CLIP 以文搜图索引一致。
"""
import os
import json

from src.config import VECTOR_DIR

_DROP = None


def _drop_set():
    """加载并缓存 VLM 明确判为「非内容图」(keep=false / 打不开)的图片 basename 集合(仅读一次)。"""
    global _DROP
    if _DROP is None:
        _DROP = set()
        try:
            path = os.path.join(VECTOR_DIR, 'vlm_verdicts.json')
            verdicts = json.load(open(path, 'r', encoding='utf-8'))
            for name, v in verdicts.items():
                keep = v.get('keep', True) if isinstance(v, dict) else bool(v)
                if not keep:
                    _DROP.add(name)
        except Exception:
            _DROP = set()
    return _DROP


def is_content_image(basename):
    """是否为内容图:仅当被 VLM 明确判为非内容图时返回 False;其余(含未判定)返回 True。"""
    return basename not in _drop_set()


def filter_content_images(local_images):
    """local_images(含 local_path 的 dict 列表) → 内容图的 basename 列表(已剔除 VLM 判否的无关图)。"""
    out = []
    for im in (local_images or []):
        raw = im.get('local_path') if isinstance(im, dict) else im
        lp = str(raw or '').replace('\\', '/')
        if lp and is_content_image(os.path.basename(lp)):
            out.append(os.path.basename(lp))
    return out
