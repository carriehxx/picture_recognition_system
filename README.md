# 儿童人脸识别系统

一个专为儿童人脸识别场景优化的智能识别系统，集成了MySQL数据库、阿里云OSS对象存储和深度学习人脸识别技术。

## 🎯 系统特点

- **儿童优化**: 专门针对儿童面部特征进行优化，提高识别准确率
- **多模态存储**: 支持MySQL数据库 + 阿里云OSS对象存储
- **RESTful API**: 提供完整的HTTP API接口
- **批量处理**: 支持批量人脸识别和数据处理
- **质量评估**: 自动评估人脸图片质量，确保识别效果
- **日志管理**: 完整的日志记录和轮转机制

## 📁 项目结构

```
pic_sys/
├── start_system.py              # 系统启动脚本（主要入口）
├── face_recognition_api.py      # RESTful API服务
├── child_face_system.py         # 儿童人脸识别核心模块
├── database_integration.py      # 数据库集成模块
├── config/
│   └── storage_config.py        # 存储配置管理
├── storage_managers/            # 存储管理器
│   ├── __init__.py
│   ├── base_storage.py         # 存储基类
│   └── aliyun_oss.py           # 阿里云OSS存储实现
├── storage/                     # 本地图片存储
├── resource/                   # 资源文件
├── logs/                       # 日志文件目录
├── results/                    # 识别结果
├── processed/                  # 处理后的图片
└── temp/                       # 临时文件
```

## 功能描述

- `face_recognition_api.py` 主要包含了接口信息
- `child_face_system.py` 主要包含了人脸数据创建类 和 图像人脸识别类，以及一些相关参数
- `database_integration.py` 主要实现了数据库和OSS的连接，数据和图像存储和修改
- `start_system.py` 主要用于系统启动，包含了日志设计，环境变量检查，系统要求检查，MySQL和OSS连接，API启动

## 🚀 快速开始

### 1. 环境要求

- Python 3.7+
- MySQL 5.7+
- 阿里云OSS账户（可选）

### 2. 安装依赖

```bash
# 安装基础依赖
pip install torch facenet-pytorch mysql-connector-python flask flask-cors pillow numpy python-dotenv opencv-python

# 安装OSS依赖（阿里云OSS)
pip install oss2
```

### 3. 配置环境变量

创建 `.env` 文件：

```env
# 数据库配置
DB_HOST=远端数据库
DB_PORT=3306
DB_NAME=kindergarten_system
DB_USER=用户名
DB_PASSWORD=your_password

# API配置
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=False

# 日志配置
LOG_LEVEL=INFO
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5

# 阿里云OSS配置
STORAGE_PROVIDER=aliyun
ALIYUN_BUCKET_NAME=your-bucket-name
ALIYUN_REGION=oss-cn-shenzhen
ALIYUN_ACCESS_KEY_ID=your-access-key
ALIYUN_SECRET_ACCESS_KEY=your-secret-key
ALIYUN_ENDPOINT=https://oss-cn-shenzhen.aliyuncs.com
```

### 4. 启动系统

```bash
# 启动系统（主要方式）
python start_system.py
```

系统启动后会：
- 检查环境配置
- 测试数据库和OSS连接
- 加载人脸识别模型
- 启动API服务器

## 🔧 主要功能

### 1. 人脸识别API

#### 批量识别
**注意**: 所有数组参数的长度必须与 `images` 数组长度一致
```bash
curl -X POST http://localhost:5000/batch_recognize \
  -H 'Content-Type: application/json' \
  -d '{
    "images": ["base64图片1", "base64图片2"],
    "class_id": [1, 1],
    "activity_detail": ["户外活动", "室内游戏"],
    "is_public": [true, false],
    "uploader_id": [1, 1],
    "max_workers": 4
  }'
```

### 2. 儿童管理API

#### 添加儿童
**注意**: `images` 支持多张照片，`profile` 包含儿童档案信息
```bash
curl -X POST http://localhost:5000/database/add_child \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "张三",
    "images": ["base64图片数据1", "base64图片数据2"],
    "update_if_exists": true,
    "profile": {
      "student_id": 1001,
      "age": 5,
      "class_id": 1
    }
  }'
```

#### 获取儿童列表
```bash
curl http://localhost:5000/database/children
```

#### 获取儿童档案
```bash
curl http://localhost:5000/child/1001/profile
```

### 3. 系统管理API

#### 健康检查
```bash
curl http://localhost:5000/health
```

#### 系统统计
```bash
curl http://localhost:5000/system/statistics
```

#### 获取配置
```bash
curl http://localhost:5000/config
```

#### 删除儿童
```bash
curl -X POST http://localhost:5000/database/delete_child \
  -H 'Content-Type: application/json' \
  -d '{"student_id": 1001}'
```

#### 获取数据库信息
```bash
curl http://localhost:5000/database/info
```

## 🗄️ 数据库设计

系统使用MySQL数据库存储以下信息：

- **children**: 儿童基本信息（姓名、年龄、班级等）
- **photos**: 照片记录（路径、上传者、活动详情等）
- **photo_child**: 照片与儿童的关联记录
- **face_embeddings**: 人脸特征向量和识别阈值
- **classes**: 班级信息
- **users**: 用户信息（教师、家长）
- **system_config**: 系统配置参数

## 📊 存储架构

### 本地存储
- 图片存储在 `storage/` 目录
- 数据库文件：`face_database.pkl`
- 儿童档案：`child_profiles.json`

### 对象存储（阿里云OSS）
- 自动上传识别后的图片
- 提供公开访问URL

## 🔍 日志管理

系统日志存储在 `logs/` 目录：
- 按日期自动分割日志文件
- 支持日志轮转（最大10MB，保留5个备份）
- 同时输出到控制台和文件

### 日志管理工具
```bash
# 查看日志
python manage_logs.py list
python manage_logs.py view 20241201

# 清理日志
python manage_logs.py clean --days 30
```

## 🎛️ 配置说明

### 儿童识别配置
系统针对不同年龄段儿童优化了识别参数：

```python
CHILD_CONFIG = {
    'recognition': {
        'default_threshold': 0.55,    # 默认识别阈值
        'quality_threshold': 0.6,     # 质量阈值
        'min_face_size': 60,         # 最小人脸尺寸
        'max_age_days': 180,         # 最大年龄差异天数
    },
    'age_thresholds': {
        'infant': (0, 2, 0.65),      # 婴儿期
        'toddler': (2, 5, 0.60),     # 幼儿期
        'child': (5, 10, 0.55),      # 儿童期
        'preteen': (10, 13, 0.50)    # 青春期前
    }
}
```

### 存储配置
支持多种存储方式：
- **local**: 本地文件存储
- **aliyun**: 阿里云OSS存储

## 🔧 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查MySQL服务是否启动
   - 验证数据库配置信息
   - 确认数据库用户权限

2. **OSS连接失败**
   - 检查阿里云OSS配置
   - 验证AccessKey权限
   - 确认Bucket存在且可访问

3. **模型加载失败**
   - 检查PyTorch安装
   - 确认网络连接（首次运行需要下载模型）
   - 验证GPU驱动（如果使用GPU）

4. **API服务启动失败**
   - 检查端口是否被占用
   - 验证防火墙设置
   - 确认依赖包安装完整

### 调试模式
```bash
# 启用调试模式
export API_DEBUG=true
python start_system.py
```

## 📈 性能优化

### 系统优化建议
1. **GPU加速**: 使用CUDA加速人脸识别
2. **批量处理**: 使用批量API提高处理效率
3. **缓存机制**: 启用模型缓存减少加载时间
4. **连接池**: 配置数据库连接池

### 监控指标
- 识别准确率
- 处理延迟
- 系统资源使用率
- 存储空间使用情况

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 支持

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件 [点击这里](mailto:carriehuangxin@foxmail.com)
- 查看项目文档

---

**注意**: 本系统专门针对儿童人脸识别场景优化，请确保遵守相关法律法规和隐私保护要求。 
