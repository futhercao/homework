"""
查询级 VLM 相关性评审 (以文搜图的"精排闸")

问题背景: CLIP 以文搜图永远返回"余弦最相近"的 top-k, 但中文 CLIP 的模态gap
会把图文余弦压进很窄的区间 —— 单靠余弦阈值要么漏掉真匹配、要么放进无关图
(典型现象: 搜"地震"却混入无关风景/人物图)。

本模块用视觉大模型(Qwen-VL)做"精排": 对 CLIP 召回的候选图逐张审视, 判断它
在内容上是否真的与查询词相关, 只保留判为相关者。分工清晰:
  - CLIP   = 召回 (recall): 快, 把可能相关的图都捞上来 (低余弦下限);
  - VLM    = 精排 (precision): 慢但准, 逐张裁定是否真相关, 滤掉无关图。

设计要点 (与 vlm_filter.py 一致):
- **缓存**: 判定按 (查询词, 文件名) 持久化到 data/vectors/vlm_judge_cache.json,
  同一查询词重复搜索零 API 调用 (演示秒回)。
- **并发 + 重试**: 线程池并发, 失败指数退避; 多次失败则 fail-open(保留该图但标记
  judged=false 且不写缓存), 避免一次网络抖动把结果清空。
- **密钥**: 仅从环境变量读取(默认 DASHSCOPE_API_KEY), 绝不写入任何文件;
  无密钥时优雅降级(直接返回 CLIP 候选, 不做精排)。
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

# 评审提示词: 站在"用此词搜图的用户"视角判断这张图是否就是他想要的结果。
_PROMPT_TMPL = (
    "你是图文检索系统的相关性评审员。用户的检索词是「{q}」。"
    "请判断这张图片在内容上是否与该检索词相关——即一个用「{q}」搜索图片的用户,"
    "会不会认为这张图正是他想要的结果。注意要看图片真实内容, 不要被无关元素干扰。"
    "严格只输出一个 JSON 对象, 不要任何多余文字、解释或代码块标记: "
    '{{"match": true或false, "conf": 0到1之间的小数(判定为相关的把握), '
    '"desc": "图片真实内容的简短中文描述(不超过12字)"}}'
)


class VLMRelevanceJudge:
    """对 (查询词, 候选图) 做"相关 vs 无关"二分类, 带 (query,file) 级磁盘缓存与并发。"""

    def __init__(self):
        cfg = MULTIMODAL_CONFIG
        self.model = cfg.get('vlm_judge_model', 'qwen3-vl-plus')
        self.base_url = cfg.get('vlm_base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.api_key = os.environ.get(cfg.get('vlm_api_key_env', 'DASHSCOPE_API_KEY'), '').strip()
        self.max_side = int(cfg.get('vlm_image_max_side', 512))
        self.workers = int(cfg.get('vlm_max_workers', 8))
        self.min_conf = float(cfg.get('vlm_judge_min_conf', 0.55))
        self.cache_path = os.path.join(VECTOR_DIR, 'vlm_judge_cache.json')
        self.cache = self._load_cache()
        self._lock = threading.Lock()
        self._client = None
        self.ready = bool(self.api_key)   # 有无密钥(能否评审新图); 缓存命中与之无关

    # ---- 缓存 ((query||basename) -> verdict) ----
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

    @staticmethod
    def _key(query, path):
        return (query or '').strip() + '||' + os.path.basename(path)

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

    def _judge_one(self, query, path):
        """返回 verdict dict {match,conf,desc}; None 表示 API 多次失败(交由调用方 fail-open)。"""
        try:
            uri = self._to_data_uri(path)
        except Exception as e:
            return {'match': False, 'conf': 0.0, 'desc': '打不开(%s)' % str(e)[:20]}
        prompt = _PROMPT_TMPL.format(q=query)
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': uri}},
                    ]}],
                    temperature=0,
                    max_tokens=120,
                )
                raw = resp.choices[0].message.content or ''
                j = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
                conf = j.get('conf', 0)
                try:
                    conf = float(conf)
                except Exception:
                    conf = 0.0
                return {'match': bool(j.get('match')),
                        'conf': max(0.0, min(1.0, conf)),
                        'desc': str(j.get('desc', ''))[:40]}
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        return None  # 判定失败

    # ---- 批量精排 ----
    def judge(self, query, candidates):
        """对 CLIP 候选图做相关性精排。

        参数: candidates = [{'image_path','score'(余弦), 'metadata'...}, ...] (已按余弦降序)
        返回: (annotated, stats)
          annotated: 与输入等长, 每项追加 vlm_match/vlm_conf/vlm_desc/judged/keep;
                     列表按 (keep, vlm_conf, 余弦) 重排, 相关者在前。
          stats: {'judged': n, 'kept': n, 'dropped': n, 'no_key': bool, 'cache_hits': n}
        无密钥时: 不评审, 原样返回并标记 judged=False/keep=True(降级为纯 CLIP)。
        """
        if not candidates:
            return [], {'judged': 0, 'kept': 0, 'dropped': 0, 'no_key': not self.ready, 'cache_hits': 0}

        # 1) 先吃缓存, 找出真正要打 API 的图
        verdicts = {}        # idx -> verdict dict
        to_call = []         # [(idx, path)]
        cache_hits = 0
        for i, c in enumerate(candidates):
            path = c['image_path']
            k = self._key(query, path)
            if k in self.cache:
                verdicts[i] = self.cache[k]
                cache_hits += 1
            else:
                to_call.append((i, path))

        # 2) 并发评审未缓存者(有密钥时)
        if to_call and self.ready:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._judge_one, query, p): (i, p) for i, p in to_call}
                for fut in as_completed(futs):
                    i, p = futs[fut]
                    v = fut.result()
                    if v is not None:
                        verdicts[i] = v
                        with self._lock:
                            self.cache[self._key(query, p)] = v
            self._save_cache()

        # 3) 组装裁决
        annotated = []
        for i, c in enumerate(candidates):
            v = verdicts.get(i)
            r = dict(c)
            if v is None:
                # 无密钥 或 评审失败 → fail-open: 保留, 但标记未判定(纯 CLIP 召回)
                r.update({'vlm_match': None, 'vlm_conf': None, 'vlm_desc': '',
                          'judged': False, 'keep': True})
            else:
                keep = bool(v.get('match')) and float(v.get('conf', 0)) >= self.min_conf
                r.update({'vlm_match': bool(v.get('match')), 'vlm_conf': round(float(v.get('conf', 0)), 2),
                          'vlm_desc': v.get('desc', ''), 'judged': True, 'keep': keep})
            annotated.append(r)

        # 4) 相关者在前, 再按 VLM 置信度、余弦排序
        annotated.sort(key=lambda r: (r['keep'],
                                      r['vlm_conf'] if r['vlm_conf'] is not None else 0,
                                      r.get('score', 0)), reverse=True)
        kept = sum(1 for r in annotated if r['keep'])
        stats = {'judged': sum(1 for r in annotated if r['judged']),
                 'kept': kept, 'dropped': len(annotated) - kept,
                 'no_key': not self.ready, 'cache_hits': cache_hits}
        return annotated, stats
