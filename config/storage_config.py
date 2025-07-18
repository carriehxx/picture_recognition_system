#!/usr/bin/env python3
"""
对象存储 OSS 配置管理
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# 强制覆盖已存在的环境变量
load_dotenv(override=True)

@dataclass
class StorageConfig:
    """存储配置基类"""
    provider: str  
    bucket_name: str
    region: str
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint: Optional[str] = None

# 配置示例
STORAGE_CONFIGS = {
    'aliyun': StorageConfig(
        provider='aliyun',
        bucket_name=os.getenv('ALIYUN_BUCKET_NAME', 'zdlt-pic-sys'),
        region=os.getenv('ALIYUN_REGION', 'oss-cn-shenzhen'),
        access_key=os.getenv('ALIYUN_ACCESS_KEY_ID', 'your-access-key-id'),
        secret_key=os.getenv('ALIYUN_SECRET_ACCESS_KEY', 'your-secret-access-key'),
        endpoint=os.getenv('ALIYUN_ENDPOINT', 'https://oss-cn-shenzhen.aliyuncs.com')
    ),
    
    'local': StorageConfig(
        provider='local',
        bucket_name='uploads',
        region='local'
    )
}

def get_storage_config(provider: str = None) -> StorageConfig:
    """获取存储配置"""
    if provider is None:
        provider = os.getenv('STORAGE_PROVIDER', 'local')
    
    if provider not in STORAGE_CONFIGS:
        raise ValueError(f"不支持的存储提供商: {provider}")
    
    config = STORAGE_CONFIGS[provider]
    
    # 验证配置是否完整（对于aliyun提供商）
    if provider == 'aliyun':
        missing_configs = []
        if not config.access_key:
            missing_configs.append('ALIYUN_ACCESS_KEY_ID')
        if not config.secret_key:
            missing_configs.append('ALIYUN_SECRET_ACCESS_KEY')
        if not config.bucket_name or config.bucket_name == 'your-oss-bucket':
            missing_configs.append('ALIYUN_BUCKET_NAME')
        if not config.endpoint:
            missing_configs.append('ALIYUN_ENDPOINT')
            
        if missing_configs:
            print(f"⚠️  缺少配置: {', '.join(missing_configs)}")
        else:
            print(f"✅ aliyun配置完整: bucket={config.bucket_name}, endpoint={config.endpoint}")
    
    return config

# 存储路径配置
STORAGE_PATHS = {
    'faces': 'faces',           # 人脸图片
    'originals': 'originals',   # 原始图片
    'processed': 'processed',   # 处理后图片
    'thumbnails': 'thumbnails', # 缩略图
    'test_images': 'test_images', # 识别测试图片
    'temp': 'temp'             # 临时文件
} 