#!/usr/bin/env python3
"""
本地存储管理器实现
用于开发环境或不使用云存储的场景
"""

import os
import shutil
from typing import Dict, Any, Optional, IO
from .base_storage import BaseStorageManager
from config.storage_config import StorageConfig

class LocalStorageManager(BaseStorageManager):
    """本地存储管理器"""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        
        # 本地存储根路径
        self.storage_root = os.path.abspath(config.bucket_name)
        
        # 确保存储目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        from config.storage_config import STORAGE_PATHS
        
        # 创建根目录
        os.makedirs(self.storage_root, exist_ok=True)
        
        # 创建各个分类目录
        for category in STORAGE_PATHS.values():
            category_path = os.path.join(self.storage_root, category)
            os.makedirs(category_path, exist_ok=True)
    
    def _get_local_path(self, object_key: str) -> str:
        """获取完整的本地路径"""
        return os.path.join(self.storage_root, object_key)
    
    def upload_file(self, file_obj: IO, object_key: str, content_type: str = None) -> Dict[str, Any]:
        """上传文件到本地存储"""
        try:
            local_path = self._get_local_path(object_key)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 保存文件
            with open(local_path, 'wb') as f:
                if hasattr(file_obj, 'read'):
                    shutil.copyfileobj(file_obj, f)
                else:
                    f.write(file_obj)
            
            file_size = os.path.getsize(local_path)
            
            return {
                'success': True,
                'url': self.get_direct_url(object_key),
                'key': object_key,
                'size': file_size,
                'provider': 'local',
                'local_path': local_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"本地存储上传失败: {e}",
                'key': object_key
            }
    
    def upload_from_path(self, file_path: str, object_key: str, content_type: str = None) -> Dict[str, Any]:
        """从本地路径上传文件"""
        try:
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f"源文件不存在: {file_path}",
                    'key': object_key
                }
            
            local_path = self._get_local_path(object_key)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 复制文件
            shutil.copy2(file_path, local_path)
            
            return {
                'success': True,
                'url': self.get_direct_url(object_key),
                'key': object_key,
                'size': os.path.getsize(local_path),
                'provider': 'local',
                'local_path': local_path,
                'source_path': file_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"本地文件复制失败: {e}",
                'key': object_key
            }
    
    def download_file(self, object_key: str, local_path: str) -> bool:
        """下载文件（本地存储中相当于复制）"""
        try:
            source_path = self._get_local_path(object_key)
            
            if not os.path.exists(source_path):
                return False
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 复制文件
            shutil.copy2(source_path, local_path)
            return True
            
        except Exception as e:
            print(f"本地文件下载失败: {e}")
            return False
    
    def delete_file(self, object_key: str) -> bool:
        """删除本地文件"""
        try:
            local_path = self._get_local_path(object_key)
            
            if os.path.exists(local_path):
                os.remove(local_path)
                return True
            else:
                return False
                
        except Exception as e:
            print(f"删除本地文件失败: {e}")
            return False
    
    def get_direct_url(self, object_key: str) -> str:
        """获取直接访问URL"""
        if self.cdn_domain:
            return f"{self.cdn_domain.rstrip('/')}/{object_key}"
        else:
            # 返回相对路径URL
            return f"/static/uploads/{object_key}"
    
    def get_presigned_url(self, object_key: str, expiration: int = 3600) -> str:
        """获取预签名URL（本地存储返回直接URL）"""
        return self.get_direct_url(object_key)
    
    def file_exists(self, object_key: str) -> bool:
        """检查文件是否存在"""
        local_path = self._get_local_path(object_key)
        return os.path.exists(local_path)
    
    def get_file_info(self, object_key: str) -> Optional[Dict[str, Any]]:
        """获取文件信息"""
        try:
            local_path = self._get_local_path(object_key)
            
            if not os.path.exists(local_path):
                return None
            
            stat = os.stat(local_path)
            
            return {
                'key': object_key,
                'size': stat.st_size,
                'last_modified': stat.st_mtime,
                'local_path': local_path,
                'url': self.get_direct_url(object_key),
                'public_url': self.get_public_url(object_key)
            }
            
        except Exception:
            return None
    
    def list_files(self, prefix: str = "", max_files: int = 1000) -> list:
        """列出本地文件"""
        try:
            files = []
            search_path = os.path.join(self.storage_root, prefix) if prefix else self.storage_root
            
            if not os.path.exists(search_path):
                return files
            
            count = 0
            for root, dirs, filenames in os.walk(search_path):
                for filename in filenames:
                    if count >= max_files:
                        break
                    
                    file_path = os.path.join(root, filename)
                    # 计算相对于存储根目录的路径
                    relative_path = os.path.relpath(file_path, self.storage_root)
                    # 转换为统一的key格式（使用/分隔符）
                    object_key = relative_path.replace(os.sep, '/')
                    
                    stat = os.stat(file_path)
                    files.append({
                        'key': object_key,
                        'size': stat.st_size,
                        'last_modified': stat.st_mtime,
                        'local_path': file_path,
                        'url': self.get_direct_url(object_key),
                        'public_url': self.get_public_url(object_key)
                    })
                    
                    count += 1
                
                if count >= max_files:
                    break
            
            return files
            
        except Exception as e:
            print(f"列出本地文件失败: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            total_size = 0
            total_files = 0
            
            for root, dirs, files in os.walk(self.storage_root):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                        total_files += 1
                    except OSError:
                        continue
            
            return {
                'total_files': total_files,
                'total_size': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'storage_root': self.storage_root
            }
            
        except Exception as e:
            return {
                'error': f"获取存储统计失败: {e}"
            } 