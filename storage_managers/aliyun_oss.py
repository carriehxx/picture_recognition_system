#!/usr/bin/env python3
"""
阿里云OSS存储管理器
"""

import oss2
import logging
from typing import Dict, Any, IO, Optional
import os
import time
import hashlib
from datetime import datetime
from urllib.parse import urljoin

from .base_storage import BaseStorageManager
from config.storage_config import StorageConfig

logger = logging.getLogger(__name__)

class OSSStorageManager(BaseStorageManager):
    """阿里云OSS存储管理器"""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        
        # 验证配置
        if not all([config.access_key, config.secret_key, config.endpoint]):
            raise ValueError("阿里云OSS配置不完整，请检查环境变量")
        
        # 初始化OSS客户端
        try:
            print(config.access_key, config.secret_key, config.endpoint, config.bucket_name)
            self.auth = oss2.Auth(config.access_key, config.secret_key)
            self.bucket = oss2.Bucket(self.auth, config.endpoint, config.bucket_name)
            
            # 测试连接
            self._test_connection()
            logger.info(f"✅ 阿里云OSS连接成功: {config.bucket_name}")
            
        except Exception as e:
            logger.error(f"❌ 阿里云OSS初始化失败: {e}")
            raise
    
    def _test_connection(self):
        """测试OSS连接"""
        try:
            # 尝试获取bucket信息来测试连接
            self.bucket.get_bucket_info()
            logger.info("✅ Bucket访问正常")
            
        except oss2.exceptions.NoSuchBucket:
            raise Exception(f"Bucket '{self.bucket_name}' 不存在")
        except oss2.exceptions.AccessDenied:
            raise Exception("OSS访问权限不足，请检查AccessKey")
        except oss2.exceptions.InvalidArgument:
            raise Exception("OSS AccessKey无效")
        except Exception as e:
            logger.warning(f"OSS连接测试警告: {e}")
    
    def upload_file(self, file_obj: IO, object_key: str, content_type: str = None) -> Dict[str, Any]:
        """
        上传文件到OSS
        
        Args:
            file_obj: 文件对象
            object_key: OSS对象键
            content_type: 文件类型
            
        Returns:
            上传结果信息
        """
        try:
            # 重置文件指针
            file_obj.seek(0)
            
            # 读取文件内容
            file_content = file_obj.read()
            file_size = len(file_content)
            
            # 生成文件MD5
            file_md5 = hashlib.md5(file_content).hexdigest()
            
            # 设置上传参数
            headers = {}
            if content_type:
                headers['Content-Type'] = content_type
            
            # 上传文件
            start_time = time.time()
            result = self.bucket.put_object(
                object_key, 
                file_content,
                headers=headers
            )
            upload_time = time.time() - start_time
            
            # 构建返回结果
            upload_result = {
                'success': True,
                'object_key': object_key,
                'url': self.get_direct_url(object_key),
                'size': file_size,
                'md5': file_md5,
                'etag': result.etag.strip('"'),
                'upload_time': upload_time,
                'content_type': content_type,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ 文件上传成功: {object_key} ({file_size} bytes)")
            return upload_result
            
        except oss2.exceptions.OssError as e:
            error_msg = f"OSS上传失败: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'object_key': object_key
            }
        except Exception as e:
            error_msg = f"文件上传异常: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'object_key': object_key
            }
    
    def upload_from_path(self, file_path: str, object_key: str, content_type: str = None) -> Dict[str, Any]:
        """
        从本地路径上传文件到OSS
        
        Args:
            file_path: 本地文件路径
            object_key: OSS对象键
            content_type: 文件类型
            
        Returns:
            上传结果信息
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f'文件不存在: {file_path}',
                    'object_key': object_key
                }
            
            # 获取文件大小
            file_size = os.path.getsize(file_path)
            
            # 生成文件MD5
            with open(file_path, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            
            # 设置上传参数
            headers = {}
            if content_type:
                headers['Content-Type'] = content_type
            elif file_path.lower().endswith(('.jpg', '.jpeg')):
                headers['Content-Type'] = 'image/jpeg'
            elif file_path.lower().endswith('.png'):
                headers['Content-Type'] = 'image/png'
            elif file_path.lower().endswith('.webp'):
                headers['Content-Type'] = 'image/webp'
            
            # 上传文件
            start_time = time.time()
            result = self.bucket.put_object_from_file(
                object_key, 
                file_path,
                headers=headers
            )
            upload_time = time.time() - start_time
            
            # 构建返回结果
            upload_result = {
                'success': True,
                'object_key': object_key,
                'url': self.get_direct_url(object_key),
                'size': file_size,
                'md5': file_md5,
                'etag': result.etag.strip('"'),
                'upload_time': upload_time,
                'content_type': headers.get('Content-Type'),
                'timestamp': datetime.now().isoformat(),
                'local_path': file_path
            }
            
            logger.info(f"✅ 文件从路径上传成功: {file_path} -> {object_key} ({file_size} bytes)")
            return upload_result
            
        except oss2.exceptions.OssError as e:
            error_msg = f"OSS上传失败: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'object_key': object_key,
                'local_path': file_path
            }
        except Exception as e:
            error_msg = f"文件上传异常: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'object_key': object_key,
                'local_path': file_path
            }

    def download_file(self, object_key: str, local_path: str) -> bool:
        """
        从OSS下载文件到本地
        
        Args:
            object_key: OSS对象键
            local_path: 本地保存路径
            
        Returns:
            是否下载成功
        """
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 下载到本地文件
            self.bucket.get_object_to_file(object_key, local_path)
            logger.info(f"✅ 文件下载成功: {object_key} -> {local_path}")
            return True
                
        except oss2.exceptions.NoSuchKey:
            logger.error(f"文件不存在: {object_key}")
            return False
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False
    
    def download_file_to_memory(self, object_key: str) -> Dict[str, Any]:
        """
        从OSS下载文件到内存（扩展方法）
        
        Args:
            object_key: OSS对象键
            
        Returns:
            下载结果信息
        """
        try:
            # 下载到内存
            result = self.bucket.get_object(object_key)
            content = result.read()
            return {
                'success': True,
                'object_key': object_key,
                'content': content,
                'size': len(content)
            }
                
        except oss2.exceptions.NoSuchKey:
            error_msg = f"文件不存在: {object_key}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        except Exception as e:
            error_msg = f"下载失败: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def delete_file(self, object_key: str) -> bool:
        """
        删除OSS文件
        
        Args:
            object_key: OSS对象键
            
        Returns:
            是否删除成功
        """
        try:
            self.bucket.delete_object(object_key)
            logger.info(f"✅ 文件删除成功: {object_key}")
            return True
        except Exception as e:
            logger.error(f"删除失败: {e}")
            return False
    
    def delete_file_by_url(self, url: str) -> bool:
        """
        通过URL删除OSS文件
        
        Args:
            url: OSS文件URL
            
        Returns:
            是否删除成功
        """
        try:
            # 从URL中提取Object Key
            object_key = self._extract_object_key_from_url(url)
            if not object_key:
                logger.error(f"无法从URL中提取Object Key: {url}")
                return False
            
            # 调用原有的删除方法
            return self.delete_file(object_key)
            
        except Exception as e:
            logger.error(f"通过URL删除失败: {e}")
            return False
    
    def _extract_object_key_from_url(self, url: str) -> str:
        """
        从OSS URL中提取Object Key
        
        Args:
            url: OSS文件URL
            
        Returns:
            Object Key
        """
        try:
            # 移除协议和域名部分
            # 例如: https://mybucket.oss-cn-beijing.aliyuncs.com/photos/2024/01/child_123.jpg
            # 提取: photos/2024/01/child_123.jpg
            
            # 方法1：字符串分割
            if '.aliyuncs.com/' in url:
                parts = url.split('.aliyuncs.com/')
                if len(parts) > 1:
                    return parts[1]
            
            # 方法2：使用URL解析
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.path.lstrip('/')
            
        except Exception as e:
            logger.error(f"提取Object Key失败: {e}")
            return ""
    
    def delete_multiple_files(self, object_keys: list) -> Dict[str, Any]:
        """
        批量删除多个文件
        
        Args:
            object_keys: Object Key列表
            
        Returns:
            删除结果统计
        """
        try:
            if not object_keys:
                return {'success': True, 'deleted': 0, 'failed': 0}
            
            # 批量删除
            result = self.bucket.batch_delete_objects(object_keys)
            
            deleted_count = len(result.deleted_keys)
            failed_count = len(result.error_list) if hasattr(result, 'error_list') else 0
            
            logger.info(f"✅ 批量删除完成: 成功 {deleted_count} 个，失败 {failed_count} 个")
            
            return {
                'success': True,
                'deleted': deleted_count,
                'failed': failed_count,
                'deleted_keys': result.deleted_keys if hasattr(result, 'deleted_keys') else [],
                'failed_keys': result.error_list if hasattr(result, 'error_list') else []
            }
            
        except Exception as e:
            logger.error(f"批量删除失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'deleted': 0,
                'failed': len(object_keys)
            }
    
    def file_exists(self, object_key: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            object_key: OSS对象键
            
        Returns:
            文件是否存在
        """
        try:
            return self.bucket.object_exists(object_key)
        except Exception as e:
            logger.error(f"检查文件存在性失败: {e}")
            return False
    
    def get_file_info(self, object_key: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息
        
        Args:
            object_key: OSS对象键
            
        Returns:
            文件信息字典或None
        """
        try:
            meta = self.bucket.head_object(object_key)
            return {
                'object_key': object_key,
                'size': meta.content_length,
                'last_modified': meta.last_modified,
                'etag': meta.etag.strip('"'),
                'content_type': meta.content_type
            }
        except oss2.exceptions.NoSuchKey:
            logger.error(f"文件不存在: {object_key}")
            return None
        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return None
    
    def list_files(self, prefix: str = '', max_keys: int = 100) -> Dict[str, Any]:
        """
        列出文件
        
        Args:
            prefix: 前缀过滤
            max_keys: 最大数量
            
        Returns:
            文件列表
        """
        try:
            files = []
            for obj in oss2.ObjectIterator(self.bucket, prefix=prefix, max_keys=max_keys):
                files.append({
                    'object_key': obj.key,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'etag': obj.etag.strip('"'),
                    'url': self.get_direct_url(obj.key)
                })
            
            return {
                'success': True,
                'files': files,
                'count': len(files)
            }
        except Exception as e:
            error_msg = f"列出文件失败: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def get_direct_url(self, object_key: str) -> str:
        """
        获取OSS直接访问URL
        
        Args:
            object_key: OSS对象键
            
        Returns:
            直接访问URL
        """
        # 构建直接访问URL
        endpoint = self.config.endpoint
        if not endpoint.startswith('http'):
            endpoint = f"https://{endpoint}"
        
        # 移除协议部分，重新构建
        endpoint_host = endpoint.replace('https://', '').replace('http://', '')
        
        # 构建完整URL
        bucket_url = f"https://{self.bucket_name}.{endpoint_host}"
        return f"{bucket_url}/{object_key}"
    
    def get_signed_url(self, object_key: str, expires: int = 3600, method: str = 'GET') -> str:
        """
        获取签名URL（用于私有访问）
        
        Args:
            object_key: OSS对象键
            expires: 过期时间（秒）
            method: HTTP方法
            
        Returns:
            签名URL
        """
        try:
            return self.bucket.sign_url(method, object_key, expires)
        except Exception as e:
            logger.error(f"生成签名URL失败: {e}")
            return self.get_direct_url(object_key)
    
    def get_presigned_url(self, object_key: str, expiration: int = 3600) -> str:
        """
        获取预签名URL（抽象方法实现）
        
        Args:
            object_key: OSS对象键
            expiration: 过期时间（秒）
            
        Returns:
            预签名URL
        """
        try:
            return self.bucket.sign_url('GET', object_key, expiration)
        except Exception as e:
            logger.error(f"生成预签名URL失败: {e}")
            return self.get_direct_url(object_key)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            # 计算存储用量（采样前100个文件）
            total_size = 0
            file_count = 0
            
            for obj in oss2.ObjectIterator(self.bucket, max_keys=100):
                total_size += obj.size
                file_count += 1
            
            return {
                'success': True,
                'bucket_name': self.bucket_name,
                'file_count_sample': file_count,
                'total_size_sample': total_size,
                'avg_file_size': total_size / file_count if file_count > 0 else 0
            }
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {'success': False, 'error': str(e)}