"""
对灾害语料的全部配图做 VLM 质检, 剔除 logo / UI / 广告 / 二维码 / 纯文字 / 截图等无关图。

用法:
    python scripts/filter_images.py            # 判定所有未缓存图 (需 DASHSCOPE_API_KEY)
    python scripts/filter_images.py --report   # 仅按已有缓存汇总, 不调用 API

判定结果缓存到 data/vectors/vlm_verdicts.json; 之后 Web 端按此缓存隐藏无关配图。
"""
import os
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import IMAGE_DIR, DATA_DIR
from src.multimodal.vlm_filter import VLMFilter

DISASTER_DIR = os.path.join(DATA_DIR, 'disaster_docs')
EXTS = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.gif')


def all_image_paths():
    paths = set()
    for ext in EXTS:
        paths.update(glob.glob(os.path.join(IMAGE_DIR, ext)))
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true', help='仅按已有缓存汇总, 不调用 API')
    args = ap.parse_args()

    paths = all_image_paths()
    print(f'图库共 {len(paths)} 张待质检')

    flt = VLMFilter()
    if args.report:
        flt.ready = False  # 强制只用缓存
    kept, stats = flt.filter_images(paths)

    print('\n===== 质检结果 =====')
    print(f"总计 {stats['total']} / 保留 {stats['kept']} / 剔除 {stats['dropped']}")
    print('剔除分类:')
    for t, c in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
        print(f'  {t:18} {c}')

    # 展示几个被剔除的样例 (文件名 + 理由), 便于核对
    dropped = [(os.path.basename(p), flt.cache.get(os.path.basename(p), {}))
               for p in paths if not (flt.cache.get(os.path.basename(p)) is None
                                      or flt.cache.get(os.path.basename(p), {}).get('keep'))]
    print('\n剔除样例 (前 15):')
    for name, v in dropped[:15]:
        print(f"  {name}  [{v.get('type','')}] {v.get('reason','')}")

    # 统计有多少文档会因此丢图 / 变空
    kept_set = {os.path.basename(p) for p in kept}
    docs_total = docs_with_img = docs_all_dropped = 0
    for fp in glob.glob(os.path.join(DISASTER_DIR, 'dis_*.json')):
        try:
            doc = json.load(open(fp, 'r', encoding='utf-8'))
        except Exception:
            continue
        imgs = [os.path.basename((im.get('local_path') or '').replace('\\', '/'))
                for im in (doc.get('local_images') or [])]
        imgs = [b for b in imgs if b]
        if not imgs:
            continue
        docs_total += 1
        kept_imgs = [b for b in imgs if b in kept_set]
        if kept_imgs:
            docs_with_img += 1
        else:
            docs_all_dropped += 1
    print(f'\n含配图文档 {docs_total}: 过滤后仍有内容图 {docs_with_img}, '
          f'图片全被剔除(变为无配图) {docs_all_dropped}')


if __name__ == '__main__':
    main()
