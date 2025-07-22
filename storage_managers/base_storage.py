#!/usr/bin/env python3
"""
抽象存储管理器基类
定义统一的对象存储接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, IO
import os
import time
import hashlib
from datetime import datetime
from config.storage_config import StorageConfig, STORAGE_PATHS

class BaseStorageManager(ABC):
    """抽象存储管理器"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.bucket_name = config.bucket_name
        
    def generate_object_key(self, category: str, filename: str, activity_name: str = None) -> str:
        """
        生成对象存储的key，确保同一批次上传的文件也能生成唯一key
        
        Args:
            category: 文件类别 (originals, thumbnails)
            filename: 文件名
            activity_name: 活动名称
            
        Returns:
            对象存储的key
        """
        # 获取分类路径
        base_path = STORAGE_PATHS.get(category, 'picture_system')
        
        # 生成时间戳路径（按年月分组）
        now = datetime.now()
        date_path = f"{now.year}/{now.month:02d}"
        
        name, ext = os.path.splitext(filename)
        # 增加更高精度的时间戳和随机数，确保同一秒内多次上传也不会重复
        high_precision_ts = int(time.time() * 1000)  # 毫秒级时间戳
        random_part = os.urandom(4).hex()  # 4字节随机数
        
        if activity_name:
            # 生成唯一标识
            unique_id = hashlib.md5(f"{activity_name}_{filename}_{high_precision_ts}_{random_part}".encode()).hexdigest()[:8]
            final_name = f"{activity_name}_{unique_id}{ext}"
        else:
            # 生成唯一标识
            unique_id = hashlib.md5(f"{name}_{high_precision_ts}_{random_part}".encode()).hexdigest()[:8]
            final_name = f"{name}_{unique_id}{ext}"
        
        return f"{base_path}/{date_path}/{final_name}"
    
    def get_public_url(self, object_key: str) -> str:
        """获取OSS直接访问URL"""
        return self.get_direct_url(object_key)
    
    @abstractmethod
    def upload_file(self, file_obj: IO, object_key: str, content_type: str = None) -> Dict[str, Any]:
        """
        上传文件
        
        Args:
            file_obj: 文件对象
            object_key: 对象存储key
            content_type: 文件类型
            
        Returns:
            上传结果字典 {'success': bool, 'url': str, 'key': str, 'size': int}
        """
        pass
    
    @abstractmethod
    def upload_from_path(self, file_path: str, object_key: str, content_type: str = None) -> Dict[str, Any]:
        """
        从本地路径上传文件
        
        Args:
            file_path: 本地文件路径
            object_key: 对象存储key
            content_type: 文件类型
            
        Returns:
            上传结果字典
        """
        pass
    
    @abstractmethod
    def download_file(self, object_key: str, local_path: str) -> bool:
        """
        下载文件到本地
        
        Args:
            object_key: 对象存储key
            local_path: 本地保存路径
            
        Returns:
            是否下载成功
        """
        pass
    
    @abstractmethod
    def delete_file(self, object_key: str) -> bool:
        """
        删除文件
        
        Args:
            object_key: 对象存储key
            
        Returns:
            是否删除成功
        """
        pass
    
    @abstractmethod
    def get_direct_url(self, object_key: str) -> str:
        """
        获取直接访问URL
        
        Args:
            object_key: 对象存储key
            
        Returns:
            直接访问URL
        """
        pass
    
    @abstractmethod
    def get_presigned_url(self, object_key: str, expiration: int = 3600) -> str:
        """
        获取预签名URL
        
        Args:
            object_key: 对象存储key
            expiration: 过期时间（秒）
            
        Returns:
            预签名URL
        """
        pass
    
    @abstractmethod
    def file_exists(self, object_key: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            object_key: 对象存储key
            
        Returns:
            文件是否存在
        """
        pass
    
    @abstractmethod
    def get_file_info(self, object_key: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息
        
        Args:
            object_key: 对象存储key
            
        Returns:
            文件信息字典或None
        """
        pass
    
    def upload_face_image(self, file_obj: IO, person_name: str, image_type: str = 'faces') -> Dict[str, Any]:
        """
        上传人脸图片的便捷方法
        
        Args:
            file_obj: 文件对象
            person_name: 人员姓名
            image_type: 图片类型 (faces, originals, processed)
            
        Returns:
            上传结果
        """
        # 确定文件扩展名
        filename = getattr(file_obj, 'name', 'image.jpg')
        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
            filename += '.jpg'
        
        # 生成对象key
        object_key = self.generate_object_key(image_type, filename, person_name)
        
        # 上传文件
        result = self.upload_file(file_obj, object_key, 'image/jpeg')
        
        # 添加公共访问URL
        if result.get('success'):
            result['public_url'] = self.get_public_url(object_key)
            result['category'] = image_type
            result['person_name'] = person_name
        
        return result 