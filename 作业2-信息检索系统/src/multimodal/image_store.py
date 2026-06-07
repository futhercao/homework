"""
图片本地存储与管理
- 下载去重（MD5）
- 缩略图生成
- 图片特征提取（颜色直方图、基础统计）
"""
import os
import sys
import json
import hashlib
from io import BytesIO
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import IMAGE_DIR, MULTIMODAL_CONFIG

from PIL import Image


class ImageStore:
    """图片本地存储管理器"""

    def __init__(self):
        self.image_dir = IMAGE_DIR
        self.max_size = MULTIMODAL_CONFIG.get('max_image_size', (800, 800))
        os.makedirs(self.image_dir, exist_ok=True)

    def store(self, image_data, url='', doc_id=''):
        """存储图片，返回本地路径和元数据"""
        img_hash = hashlib.md5(image_data).hexdigest()
        ext = self._guess_extension(url, image_data)
        filename = f'{img_hash}{ext}'
        filepath = os.path.join(self.image_dir, filename)

        # 去重：已存在的不重复存储
        if not os.path.exists(filepath):
            try:
                img = Image.open(BytesIO(image_data))
                img = img.convert('RGB')
                # 缩放到最大尺寸
                img.thumbnail(self.max_size, Image.LANCZOS)
                img.save(filepath, quality=85, optimize=True)
            except Exception:
                return None

        # 元数据
        try:
            img = Image.open(filepath)
            width, height = img.size
            file_size = os.path.getsize(filepath)
        except Exception:
            width, height, file_size = 0, 0, 0

        return {
            'local_path': filepath,
            'filename': filename,
            'hash': img_hash,
            'width': width,
            'height': height,
            'file_size': file_size,
            'original_url': url,
            'doc_id': doc_id,
        }

    def _guess_extension(self, url, data):
        if url:
            parsed = urlparse(url)
            ext = os.path.splitext(parsed.path)[1].lower()
            if ext in ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'):
                return ext
        # 从数据头判断
        if data[:3] == b'\xff\xd8\xff':
            return '.jpg'
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return '.png'
        return '.jpg'

    def get_all_images(self):
        """获取所有本地图片路径"""
        images = []
        for fname in os.listdir(self.image_dir):
            if fname.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                images.append(os.path.join(self.image_dir, fname))
        return images

    def stats(self):
        """图片库统计"""
        images = self.get_all_images()
        total_size = sum(os.path.getsize(p) for p in images)
        return {
            'total_images': len(images),
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'directory': self.image_dir,
        }
