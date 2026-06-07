"""
扩充以文搜图图库 — 从百度图片(flip 接口)按主题词批量下载真实配图 (作业2 多模态)

动机: CLIP 以文搜图的"召回"取决于库里有没有该题材的图。原库灾害/科技题材偏少,
下载一批主题相关图片显著提升召回; 下游两道闸保证质量:
  (1) build_multimodal.py 的廉价过滤 + VLM 质检(剔除 logo/广告/截图);
  (2) 查询时 Qwen-VL 相关性精排(只回真正相关者)。
因此下载图即使有少量噪声也不会污染检索结果。

用法:
  python scripts/fetch_web_images.py [每词数量=25] ["单个查询词"(仅测试)]
存入 data/images/web_<hash>.jpg, 按内容 md5 去重; 不写入任何文档, 仅扩充图库。
"""
import os
import sys
import re
import io
import time
import hashlib
import urllib.request
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import IMAGE_DIR
from PIL import Image

HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'),
    'Referer': 'https://image.baidu.com/',
}
TARGET_SIDE = 224   # 上采样短边到此值, 既喂饱 CLIP(224) 又越过 build 的 MIN_SIDE=200 闸

QUERIES = [
    ('earthquake', '地震 废墟 救援 现场'),
    ('flood',      '洪水 淹没 街道 救援'),
    ('typhoon',    '台风 登陆 狂风 暴雨'),
    ('fire',       '火灾 大火 火焰 现场'),
    ('rainstorm',  '暴雨 城市 内涝 积水'),
    ('landslide',  '山体滑坡 泥石流 灾害'),
    ('blizzard',   '暴雪 道路 积雪 救援'),
    ('drought',    '干旱 土地 龟裂'),
    ('ev',         '新能源汽车 充电桩'),
    ('chip',       '芯片 半导体 晶圆'),
    ('ai',         '人工智能 机器人'),
    ('space',      '火箭 发射 航天'),
    ('hsr',        '高铁 动车组 列车'),
]


def baidu_items(query, pages=6):
    """flip 接口抓取 (objURL 原图, thumbURL 缩略) 对, 按出现序去重。"""
    items, seen = [], set()
    for pi in range(pages):
        api = ('https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=%s&pn=%d&rn=30'
               % (urllib.parse.quote(query), pi * 30))
        try:
            raw = urllib.request.urlopen(urllib.request.Request(api, headers=HEADERS),
                                         timeout=20).read().decode('utf-8', 'ignore')
        except Exception:
            break
        objs = [u.replace('\\/', '/') for u in re.findall(r'"objURL":"([^"]+)"', raw)]
        thumbs = [u.replace('\\/', '/') for u in re.findall(r'"thumbURL":"(https?:[^"]+)"', raw)]
        page = []
        for i in range(max(len(objs), len(thumbs))):
            o = objs[i] if i < len(objs) else ''
            t = thumbs[i] if i < len(thumbs) else ''
            k = t or o
            if k and k not in seen:
                seen.add(k)
                page.append((o, t))
        if not page:
            break
        items += page
        time.sleep(0.4)
    return items


def fetch_image(obj, thumb):
    """优先原图 objURL, 失败回退 baidu CDN thumbURL(几乎必成); 短边上采样到 TARGET_SIDE。"""
    for u in (obj, thumb):
        if not u or not u.startswith('http'):
            continue
        try:
            data = urllib.request.urlopen(urllib.request.Request(u, headers=HEADERS),
                                          timeout=20).read()
            if len(data) < 3000:
                continue
            im = Image.open(io.BytesIO(data)).convert('RGB')
        except Exception:
            continue
        w, h = im.size
        m = min(w, h)
        if m < TARGET_SIDE:
            s = TARGET_SIDE / float(m)
            im = im.resize((max(TARGET_SIDE, int(w * s)), max(TARGET_SIDE, int(h * s))), Image.LANCZOS)
        return im
    return None


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    queries = [('single', sys.argv[2])] if len(sys.argv) > 2 else QUERIES
    os.makedirs(IMAGE_DIR, exist_ok=True)
    existing = set(os.listdir(IMAGE_DIR))
    total = 0
    for slug, q in queries:
        items = baidu_items(q, pages=6)
        saved = 0
        for obj, thumb in items:
            if saved >= per:
                break
            im = fetch_image(obj, thumb)
            if im is None:
                continue
            buf = io.BytesIO()
            im.save(buf, 'JPEG', quality=88)
            b = buf.getvalue()
            name = 'web_%s.jpg' % hashlib.md5(b).hexdigest()[:12]
            if name in existing:
                continue
            with open(os.path.join(IMAGE_DIR, name), 'wb') as f:
                f.write(b)
            existing.add(name)
            saved += 1
            total += 1
        print('[fetch] %-11s saved=%d (items=%d)' % (slug, saved, len(items)))
    print('[fetch] DONE total_new=%d  image_dir_now=%d' % (total, len(os.listdir(IMAGE_DIR))))


if __name__ == '__main__':
    main()
