"""
VLM 图片质检过滤器 — 用视觉大模型(Qwen-VL)剔除非内容图

新闻页抓取的配图里混有大量与新闻无关的噪声(站标logo、UI图标/按钮、广告横幅、
二维码、纯文字快讯图/海报、网页截图、纯装饰图等)。仅靠"来源URL关键词 + 最小边像素"
的廉价过滤无法识别这些"看起来是图、其实是噪声"的内容。

本模块在廉价过滤之后增加一道**语义级质检**: 把图片(降采样到 512px)送入视觉大模型,
判断其是否为"有意义的新闻内容图", 只保留通过者再进入 CLIP 以文搜图索引。

设计要点:
- **缓存**: 判定结果按 文件名→verdict 持久化到 data/vectors/vlm_verdicts.json,
  重复运行零 API 调用; 缓存随交付分发, 即使无 API Key 也能复现"已清洗"的索引。
- **并发 + 重试**: 线程池并发请求, 失败指数退避; 多次失败则 fail-open(默认保留),
  且不写缓存(留待下次重判), 避免一次网络抖动永久丢图。
- **密钥**: 仅从环境变量读取(默认 DASHSCOPE_API_KEY), **绝不写入任何文件**。
  无密钥时只应用已有缓存、跳过新图判定(优雅降级)。
"""
import os
import io
import json
import time
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from src.config import MULTIMODAL_CONFIG, VECTOR_DIR

# 质检提示词: 与平台"新闻配图检索"目标对齐 —— 保留真实新闻照片, 剔除各类噪声/UI/广告。
_PROMPT = (
    "你是新闻图片库的质检员。判断这张图片是否为『有意义的新闻内容图』——"
    "即真实拍摄的新闻照片(人物、事件、现场、场景、实物、图表数据等),适合作为新闻配图被检索。"
    "若属于以下任一类无关/噪声,判为不保留:网站logo/品牌标识、图标/按钮/导航等UI元素、"
    "广告或促销横幅、二维码、纯文字图/海报、网页或App界面截图、纯装饰图(纯色/渐变/花纹)、"
    "与新闻无关的通用素材图、水印占主导、表情包。"
    "只输出一个JSON对象,不要多余文字: "
    '{"keep": true/false, "type": "<类别>", "reason": "<简短中文理由>"}'
)


class VLMFilter:
    """用视觉大模型对图片做"内容图 vs 噪声"二分类, 带磁盘缓存与并发。"""

    def __init__(self):
        cfg = MULTIMODAL_CONFIG
        self.model = cfg.get('vlm_model', 'qwen3.6-plus')
        self.base_url = cfg.get('vlm_base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.api_key = os.environ.get(cfg.get('vlm_api_key_env', 'DASHSCOPE_API_KEY'), '').strip()
        self.max_side = int(cfg.get('vlm_image_max_side', 512))
        self.workers = int(cfg.get('vlm_max_workers', 5))
        self.cache_path = os.path.join(VECTOR_DIR, 'vlm_verdicts.json')
        self.cache = self._load_cache()
        self._lock = threading.Lock()
        self._client = None
        self.ready = bool(self.api_key)   # 有无密钥(能否判定新图); 缓存应用与之无关

    # ---- 缓存 ----
    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        os.makedirs(VECTOR_DIR, exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=1)

    # ---- 单图判定 ----
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=60.0, max_retries=0)
        return self._client

    def _to_data_uri(self, path):
        """读图 → 降采样到 max_side → JPEG → base64 data URI (PIL 处理中文路径无碍)。"""
        im = Image.open(path).convert('RGB')
        im.thumbnail((self.max_side, self.max_side))
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=85)
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

    def _judge_one(self, path):
        """返回 verdict dict; None 表示 API 多次失败(交由调用方 fail-open 且不缓存)。"""
        try:
            uri = self._to_data_uri(path)
        except Exception as e:
            # 图本身打不开 → 对检索无用, 直接判弃(可缓存)
            return {'keep': False, 'type': 'unreadable', 'reason': f'打开失败: {str(e)[:60]}'}

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': [
                        {'type': 'text', 'text': _PROMPT},
                        {'type': 'image_url', 'image_url': {'url': uri}},
                    ]}],
                    temperature=0,
                    max_tokens=200,
                )
                raw = resp.choices[0].message.content or ''
                j = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
                return {
                    'keep': bool(j.get('keep')),
                    'type': str(j.get('type', ''))[:40],
                    'reason': str(j.get('reason', ''))[:200],
                }
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        return None  # 判定失败

    # ---- 批量过滤 ----
    def filter_images(self, paths):
        """对 paths 应用质检, 返回 (kept_paths, stats)。

        已缓存者直接用缓存裁决; 未缓存者在有密钥时并发判定并写缓存,
        无密钥时默认保留(优雅降级)。API 失败的图 fail-open 保留且不缓存。
        """
        uncached = [p for p in paths if os.path.basename(p) not in self.cache]
        cached_n = len(paths) - len(uncached)
        print(f'[VLM] 共 {len(paths)} 张, 已缓存 {cached_n}, 待判定 {len(uncached)}')

        if uncached and self.ready:
            done = 0
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._judge_one, p): p for p in uncached}
                for fut in as_completed(futs):
                    p = futs[fut]
                    v = fut.result()
                    if v is not None:
                        with self._lock:
                            self.cache[os.path.basename(p)] = v
                    done += 1
                    if done % 25 == 0:
                        self._save_cache()
                        print(f'[VLM] 判定进度 {done}/{len(uncached)}')
            self._save_cache()
        elif uncached and not self.ready:
            print(f'[VLM] 未设置 API Key(env), {len(uncached)} 张无缓存图默认保留(跳过判定)')

        kept, dropped_by_type = [], {}
        for p in paths:
            v = self.cache.get(os.path.basename(p))
            if v is None or v.get('keep'):       # 无裁决(失败/无密钥) → fail-open 保留
                kept.append(p)
            else:
                t = v.get('type', '其他') or '其他'
                dropped_by_type[t] = dropped_by_type.get(t, 0) + 1
        stats = {'total': len(paths), 'kept': len(kept),
                 'dropped': len(paths) - len(kept), 'by_type': dropped_by_type}
        return kept, stats
