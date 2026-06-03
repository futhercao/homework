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
_device = 'cpu'


def _load_clip():
    """延迟加载CLIP模型"""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            import open_clip
            model_name = MULTIMODAL_CONFIG.get('clip_model', 'ViT-B-32')
            pretrained = 'laion2b_s34b_b79k'  # 开源预训练权重
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            _clip_model.eval()
            _clip_model.to(_device)
            print(f'[CLIP] 模型已加载: {model_name}')
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
                            return img_feat.cpu().numpy()
                    def encode_text(self, text):
                        import torch
                        with torch.no_grad():
                            text_feat = _model.get_text_features(input_ids=text['input_ids'],
                                                                  attention_mask=text['attention_mask'])
                            return text_feat.cpu().numpy()

                _clip_model = CLIPWrapper()
                _clip_preprocess = lambda img: _processor(images=img, return_tensors='pt')['pixel_values']
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
        """编码文本为特征向量（用于跨模态检索的查询侧）"""
        if not self.ready:
            return None
        try:
            import torch
            import open_clip
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            text_tokens = tokenizer([text]).to(_device)
            with torch.no_grad():
                features = _clip_model.encode_text(text_tokens)
            vec = features[0].cpu().detach().numpy()
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            return vec.astype(np.float32)
        except Exception as e:
            print(f'[CLIP] 编码文本失败: {e}')
            return None
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
