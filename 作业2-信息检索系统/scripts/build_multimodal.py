"""
构建多模态(CLIP)索引 — 作业2"以文搜图 / 以图搜文"的离线预构建

1. 扫描 data/images 全部图片 → (廉价URL/尺寸过滤 → VLM语义质检剔除无关图) →
   中文CLIP视觉编码 → 持久化 image_features FAISS索引;
   同时从 data/documents + data/disaster_docs 的 local_images 建立 图片→doc_id 关联,
   使"以图搜文/图片归属"可用。
2. 用CLIP文本塔编码每篇文档(标题+摘要) → 持久化 clip_text_features 索引, 供"以图搜文"
   (上传图片 → 在文档CLIP文本向量中找最相近者)。

首次运行会自动下载中文CLIP权重 (Chinese-CLIP, OFA-Sys/chinese-clip-vit-base-patch16, ~600MB) 到 HuggingFace 缓存。
索引经 faiss.serialize_index 持久化到 data/vectors/, 随交付文件夹分发, 演示时秒级加载。

用法: python scripts/build_multimodal.py
"""
import os
import sys
import glob
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import IMAGE_DIR, DOC_DIR, DATA_DIR, VECTOR_DIR, MULTIMODAL_CONFIG
from src.multimodal.cross_modal import CrossModalSearch

DISASTER_DIR = os.path.join(DATA_DIR, 'disaster_docs')

# 非内容图过滤: 新闻页含大量图标/logo/导航/二维码/头像等, 进图片检索库纯属噪声。
# 按 (1) 来源URL垃圾关键词 (2) 最小边像素 两道闸过滤, 显著提升"以文搜图"相关性。
JUNK_SRC_KW = ('logo', 'icon', 'nav', 'topapp', 'qrcode', 'qr_', 'avatar', '/ad/', '/ads/',
               'sprite', 'banner', 'common_nav', 'static.ws', 'share', 'weibo', 'button', 'blank')
MIN_SIDE = 200


def _iter_doc_files():
    yield from glob.glob(os.path.join(DOC_DIR, '*.json'))
    yield from glob.glob(os.path.join(DISASTER_DIR, 'dis_*.json'))


def build_src_map():
    """图片 basename → 原始来源URL (小写), 供垃圾图过滤"""
    m = {}
    for fp in _iter_doc_files():
        try:
            doc = json.load(open(fp, 'r', encoding='utf-8'))
        except Exception:
            continue
        for img in (doc.get('local_images') or []):
            lp = (img.get('local_path') or '').replace('\\', '/')
            if lp:
                m[os.path.basename(lp)] = (img.get('original_src') or '').lower()
    return m


def filter_content_images(images, src_map):
    """只保留'内容图': 来源URL不含垃圾关键词, 且最小边 >= MIN_SIDE 像素"""
    from PIL import Image
    kept, dropped = [], 0
    for p in images:
        src = src_map.get(os.path.basename(p), '')
        if any(k in src for k in JUNK_SRC_KW):
            dropped += 1
            continue
        try:
            w, h = Image.open(p).size
        except Exception:
            dropped += 1
            continue
        if min(w, h) < MIN_SIDE:
            dropped += 1
            continue
        kept.append(p)
    return kept, dropped


def _record_cheap_drops(dropped_basenames):
    """把廉价过滤(URL垃圾关键词 / 最小边<MIN_SIDE)剔除的图记为 keep=false, 并入 vlm_verdicts.json。
    使详情页/角标用的 content_filter(黑名单)与本图像索引口径完全一致 —— logo/广告/二维码/小图
    既不进 CLIP 索引, 也不在详情页展示。在 VLMFilter 之前写入, 不会被其覆盖(VLMFilter 会原样载回再保存)。"""
    if not dropped_basenames:
        return
    path = os.path.join(VECTOR_DIR, 'vlm_verdicts.json')
    try:
        cache = json.load(open(path, 'r', encoding='utf-8')) if os.path.exists(path) else {}
    except Exception:
        cache = {}
    for b in dropped_basenames:
        cache.setdefault(b, {'keep': False, 'type': 'cheap-filter',
                             'reason': 'URL垃圾关键词或最小边<%dpx' % MIN_SIDE})
    os.makedirs(VECTOR_DIR, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    print(f'[multimodal] 廉价过滤剔除 {len(dropped_basenames)} 张已记入 vlm_verdicts(详情页同口径隐藏)')


def build_path_to_doc():
    """图片文件名(basename) → doc_id 映射 (来自各文档的 local_images)"""
    mapping = {}
    for fp in _iter_doc_files():
        try:
            doc = json.load(open(fp, 'r', encoding='utf-8'))
        except Exception:
            continue
        did = doc.get('id', '')
        for img in (doc.get('local_images') or []):
            p = img.get('local_path', '')
            if p:
                mapping[os.path.basename(p)] = did
    return mapping


def collect_doc_texts():
    """doc_id → 文本(标题+摘要), 供CLIP文本编码(以图搜文)"""
    texts = {}
    for fp in _iter_doc_files():
        try:
            doc = json.load(open(fp, 'r', encoding='utf-8'))
        except Exception:
            continue
        did = doc.get('id', '')
        if not did:
            continue
        text = (doc.get('title', '') + ' ' + doc.get('content', '')[:60]).strip()
        if text:
            texts[did] = text
    return texts


def main():
    cs = CrossModalSearch()
    if not cs.encoder.ready:
        print('[multimodal] CLIP 不可用, 退出')
        return

    # 1) 图像向量索引 (先过滤掉非内容图)
    all_imgs = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.jpg')) +
                      glob.glob(os.path.join(IMAGE_DIR, '*.png')))
    src_map = build_src_map()
    images, dropped = filter_content_images(all_imgs, src_map)
    print(f'[multimodal] 过滤非内容图(图标/logo/二维码/<{MIN_SIDE}px): 丢弃 {dropped}, 保留 {len(images)} 张内容图')
    # 廉价过滤剔除的图也记入 verdicts, 使详情页/角标(content_filter 黑名单)与本索引同口径(在 VLM 判定前写)
    _record_cheap_drops({os.path.basename(p) for p in all_imgs} - {os.path.basename(p) for p in images})

    # 1.5) VLM 语义质检: 剔除廉价过滤漏掉的无关图(纯文字快讯/广告横幅/界面截图/装饰素材等)
    if MULTIMODAL_CONFIG.get('vlm_enabled'):
        from src.multimodal.vlm_filter import VLMFilter
        vf = VLMFilter()
        before = len(images)
        images, vstats = vf.filter_images(images)
        print(f'[multimodal] VLM质检({vf.model}): {before} → 保留 {len(images)}, 剔除 {vstats["dropped"]}')
        if vstats['by_type']:
            print(f'[multimodal] 剔除类别分布: {vstats["by_type"]}')

    base2doc = build_path_to_doc()
    doc_ids = [base2doc.get(os.path.basename(p), '') for p in images]
    linked = sum(1 for x in doc_ids if x)
    print(f'[multimodal] 内容图 {len(images)} 张, 其中 {linked} 张关联到文档')
    cs.index_images(images, doc_ids)

    # 2) 文档CLIP文本向量索引(以图搜文)
    texts = collect_doc_texts()
    print(f'[multimodal] 文档文本 {len(texts)} 条 → CLIP文本编码 (以图搜文用)')
    cs.index_document_texts(texts)

    # 3) 写出真实索引规模 → 首页统计的单一可信来源(避免硬编码数字与实际不符)
    stats_path = os.path.join(VECTOR_DIR, 'image_index_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({'total_images': cs.image_index.count,
                   'text_vectors': cs.text_index.count,
                   'dim': MULTIMODAL_CONFIG['image_index_dim']}, f, ensure_ascii=False)
    print(f'[multimodal] 索引规模: 图像 {cs.image_index.count} / 文本 {cs.text_index.count} → {stats_path}')

    print(f'[multimodal] 完成. 向量索引已持久化到 {VECTOR_DIR}')


if __name__ == '__main__':
    main()
