"""
视觉特征编码器
- CLIP图像特征提取
- 文本特征提取（用于跨模态检索）
- CPU-friendly: 轻量模型 + 批处理
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MULTIMODAL_CONFIG

# 延迟加载
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_device = 'cpu'


def _load_clip():
    """延迟加载 CLIP 模型。

    主后端: 中文原生 Chinese-CLIP (transformers, OFA-Sys/chinese-clip-vit-base-patch16),
            对中文"以文搜图/以图搜文"对齐更优, projection_dim=512。
    回退: 多语言 open_clip (xlm-roberta-base-ViT-B-32) → openai/clip-vit-base-patch32。

    无论走哪条后端, 模块级 _clip_model.encode_image/encode_text 统一返回 torch 张量
    (形如 (1, 512)), 与 VisualEncoder 中 `features[0].cpu().detach().numpy()` 的契约一致。
    """
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return _clip_model is not False

    backend = MULTIMODAL_CONFIG.get('clip_backend', 'chinese-clip')

    # === 主后端: Chinese-CLIP (transformers) ===
    if backend == 'chinese-clip':
        try:
            import torch
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
            hf_id = MULTIMODAL_CONFIG.get('clip_hf_id', 'OFA-Sys/chinese-clip-vit-base-patch16')
            _model = ChineseCLIPModel.from_pretrained(hf_id)
            _processor = ChineseCLIPProcessor.from_pretrained(hf_id)
            _model.eval()
            _model.to(_device)

            class CNCLIPWrapper:
                """统一封装: 输入张量/分词结果 → 返回 detach 后的 CPU 张量。

                transformers v5 的 get_*_features 返回 BaseModelOutputWithPooling,
                投影后的 512 维嵌入在 .pooler_output; 旧版本直接返回张量 → 两者兼容。
                """
                @staticmethod
                def _emb(out):
                    return out.pooler_output if hasattr(out, 'pooler_output') else out
                def encode_image(self, pixel_values):
                    with torch.no_grad():
                        out = _model.get_image_features(pixel_values=pixel_values.to(_device))
                    return self._emb(out).detach().cpu()
                def encode_text(self, tok):
                    with torch.no_grad():
                        out = _model.get_text_features(
                            input_ids=tok['input_ids'].to(_device),
                            attention_mask=tok['attention_mask'].to(_device))
                    return self._emb(out).detach().cpu()

            _clip_model = CNCLIPWrapper()
            # 预处理返回单图张量 (3,H,W); VisualEncoder.encode_image 会再 .unsqueeze(0)
            _clip_preprocess = lambda img: _processor(images=img, return_tensors='pt')['pixel_values'][0]
            # 分词接收 list, 截断到 Chinese-CLIP 训练上下文长度 52
            _clip_tokenizer = lambda texts: _processor(
                text=texts, padding=True, truncation=True, max_length=52, return_tensors='pt')
            print(f'[CLIP] Chinese-CLIP 已加载: {hf_id}')
            return True
        except Exception as e:
            print(f'[CLIP] Chinese-CLIP 加载失败, 回退多语言 open_clip: {e}')

    # === 回退后端: 多语言 open_clip → openai/clip ===
    try:
        import open_clip
        model_name = MULTIMODAL_CONFIG.get('clip_model', 'ViT-B-32')
        pretrained = MULTIMODAL_CONFIG.get('clip_pretrained', 'laion2b_s34b_b79k')
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        _clip_model.eval()
        _clip_model.to(_device)
        _clip_tokenizer = open_clip.get_tokenizer(model_name)
        print(f'[CLIP] 模型已加载: {model_name} ({pretrained})')
    except ImportError:
        try:
            from transformers import CLIPProcessor, CLIPModel
            _model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
            _processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            _model.eval()
            _model.to(_device)

            class CLIPWrapper:
                def encode_image(self, image):
                    import torch
                    with torch.no_grad():
                        img_feat = _model.get_image_features(pixel_values=image)
                        return img_feat.detach().cpu()
                def encode_text(self, text):
                    import torch
                    with torch.no_grad():
                        text_feat = _model.get_text_features(input_ids=text['input_ids'],
                                                             attention_mask=text['attention_mask'])
                        return text_feat.detach().cpu()

            _clip_model = CLIPWrapper()
            _clip_preprocess = lambda img: _processor(images=img, return_tensors='pt')['pixel_values'][0]
            print('[CLIP] transformers 模型已加载')
        except ImportError:
            print('[CLIP] 警告: 无可用的CLIP库')
            return False
    return _clip_model is not False


class VisualEncoder:
    """图像特征编码器"""

    def __init__(self):
        self.ready = _load_clip()

    def encode_image(self, image_path):
        """编码单张图片为特征向量"""
        if not self.ready:
            return None
        try:
            import torch
            img = Image.open(image_path).convert('RGB')
            img_tensor = _clip_preprocess(img).unsqueeze(0).to(_device)
            with torch.no_grad():
                features = _clip_model.encode_image(img_tensor)
            vec = features[0].cpu().detach().numpy() if len(features.shape) > 1 else features.cpu().detach().numpy()
            # L2归一化
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            return vec.astype(np.float32)
        except Exception as e:
            print(f'[CLIP] 编码图片失败 {image_path}: {e}')
            return None

    def encode_text(self, text):
        """编码文本为特征向量（用于跨模态检索的查询侧）

        使用与图像塔配套的 tokenizer; 多语言权重下可直接编码中文查询。
        """
        if not self.ready:
            return None
        try:
            import torch
            text_tokens = _clip_tokenizer([text]).to(_device)
            with torch.no_grad():
                features = _clip_model.encode_text(text_tokens)
            vec = features[0].cpu().detach().numpy()
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            return vec.astype(np.float32)
        except Exception as e:
            print(f'[CLIP] 编码文本失败: {e}')
            return None

    def encode_batch(self, image_paths, batch_size=8):
        """批量编码图片"""
        if not self.ready:
            return {}
        results = {}
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            for path in batch:
                vec = self.encode_image(path)
                if vec is not None:
                    results[path] = vec
            if i + batch_size < len(image_paths):
                print(f'  [CLIP] {min(i + batch_size, len(image_paths))}/{len(image_paths)}')
        return results
