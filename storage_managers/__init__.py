#!/usr/bin/env python3
"""
存储管理器工厂
根据配置创建相应的存储管理器实例
"""

from typing import Optional
from config.storage_config import StorageConfig, get_storage_config
from .base_storage import BaseStorageManager

# 延迟导入避免依赖问题
def create_storage_manager(provider: str = None, config: StorageConfig = None) -> BaseStorageManager:
    """
    创建存储管理器实例
    
    Args:
        provider: 存储提供商 (aws, aliyun, tencent, local)
        config: 存储配置对象，如果为None则使用默认配置
        
    Returns:
        存储管理器实例
        
    Raises:
        ValueError: 不支持的存储提供商
        ImportError: 缺少必要的依赖包
    """
    # 获取配置
    if config is None:
        config = get_storage_config(provider)
    
    provider = config.provider.lower()
    
    if provider == 'aliyun':
        try:
            from .aliyun_oss import OSSStorageManager
            return OSSStorageManager(config)
        except ImportError:
            raise ImportError("阿里云OSS依赖未安装: pip install oss2")
    
    elif provider == 'local':
        from .local_storage import LocalStorageManager
        return LocalStorageManager(config)
    
    else:
        raise ValueError(f"不支持的存储提供商: {provider}")

# 全局存储管理器实例（单例模式）
_storage_manager_instance: Optional[BaseStorageManager] = None

def get_storage_manager(provider: str = None, force_reload: bool = False) -> BaseStorageManager:
    """
    获取存储管理器实例（单例模式）
    
    Args:
        provider: 存储提供商
        force_reload: 是否强制重新加载
        
    Returns:
        存储管理器实例
    """
    global _storage_manager_instance
    
    if _storage_manager_instance is None or force_reload:
        _storage_manager_instance = create_storage_manager(provider)
    
    return _storage_manager_instance

def reset_storage_manager():
    """重置存储管理器实例"""
    global _storage_manager_instance
    _storage_manager_instance = None

# 便捷函数
def upload_face_image(file_obj, person_name: str, image_type: str = 'faces') -> dict:
    """上传人脸图片的便捷函数"""
    manager = get_storage_manager()
    return manager.upload_face_image(file_obj, person_name, image_type)

def get_image_url(object_key: str) -> str:
    """获取图片URL的便捷函数（直接使用OSS访问，不使用CDN）"""
    manager = get_storage_manager()
    return manager.get_public_url(object_key)

def delete_image(object_key: str) -> bool:
    """删除图片的便捷函数"""
    manager = get_storage_manager()
    return manager.delete_file(object_key) 