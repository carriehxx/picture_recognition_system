# 儿童人脸识别系统 API 接口文档

## 概述

儿童人脸识别系统提供RESTful API接口，专门针对儿童人脸识别场景优化。系统支持批量识别、儿童管理、系统监控等功能。

**基础信息：**
- 服务地址：`http://localhost:5000`
- 内容类型：`application/json`
- 最大请求大小：500MB
- 支持格式：JPEG, PNG, JPG

---

## 1. 健康检查接口

### 接口信息
- **路径**: `GET /health`
- **描述**: 检查系统健康状态和模型加载情况

### 输入
无参数

### 主要功能
- 检查系统运行状态
- 验证模型是否已加载
- 返回系统类型信息

### 输出
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "model_loaded": true,
  "system_type": "child_face_recognition"
}
```

---

## 2. 获取系统配置

### 接口信息
- **路径**: `GET /config`
- **描述**: 获取系统配置信息和上传限制

### 输入
无参数

### 主要功能
- 返回儿童识别配置参数
- 显示上传限制信息
- 提供支持的图片格式

### 输出
```json
{
  "success": true,
  "data": {
    "child_config": {
      "recognition": {
        "default_threshold": 0.55,
        "quality_threshold": 0.6,
        "min_face_size": 60,
        "max_age_days": 180
      },
      "age_thresholds": {
        "infant": [0, 2, 0.65],
        "toddler": [2, 5, 0.60],
        "child": [5, 10, 0.55],
        "preteen": [10, 13, 0.50]
      }
    },
    "upload_limits": {
      "max_file_size_mb": 500,
      "max_images_per_request": 50,
      "supported_formats": ["JPEG", "PNG", "JPG"],
      "description": "支持批量上传最多50张图片，总大小不超过500MB"
    },
    "description": "儿童人脸识别系统配置"
  }
}
```

---

## 3. 获取数据库信息

### 接口信息
- **路径**: `GET /database/info`
- **描述**: 获取儿童数据库统计信息和存储路径

### 输入
无参数

### 主要功能
- 返回数据库统计信息
- 显示存储路径配置
- 提供数据库状态概览

### 输出
```json
{
  "success": true,
  "data": {
    "statistics": {
      "total_children": 100,
      "total_photos": 500,
      "avg_quality": 0.75
    },
    "database_path": "results/face_database.pkl",
    "storage_path": "storage"
  }
}
```

---

## 4. 获取所有儿童列表

### 接口信息
- **路径**: `GET /database/children`
- **描述**: 获取系统中所有儿童的列表和详细信息

### 输入
无参数

### 主要功能
- 返回所有儿童的基本信息
- 包含照片数量和质量评分
- 显示更新状态

### 输出
```json
{
  "success": true,
  "data": {
    "total_children": 100,
    "children": [
      {
        "student_id": 1001,
        "name": "张三",
        "age": 5,
        "class_id": 1,
        "created_at": "2024-01-01T10:00:00",
        "last_updated": "2024-01-15T14:30:00",
        "photos_count": 3,
        "avg_quality": 0.82,
        "update_due_date": "2024-02-15T14:30:00"
      }
    ]
  }
}
```

---

## 5. 添加新儿童

### 接口信息
- **路径**: `POST /database/add_child`
- **描述**: 添加新儿童到系统，支持多张照片

### 输入
```json
{
  "name": "张三",
  "images": ["base64图片数据1", "base64图片数据2"],
  "update_if_exists": true,
  "profile": {
    "student_id": 1001,
    "age": 5,
    "class_id": 1
  }
}
```

**参数说明：**
- `name` (string, 必需): 儿童姓名
- `images` (array, 必需): base64编码的图片列表
- `update_if_exists` (boolean, 可选): 是否更新已存在的儿童，默认true
- `profile` (object, 必需): 儿童档案信息
  - `student_id` (integer, 必需): 学号
  - `age` (integer, 可选): 年龄，默认0
  - `class_id` (integer, 可选): 班级ID，默认0

### 主要功能
- 处理多张儿童照片
- 提取人脸特征向量
- 保存到本地数据库和MySQL
- 自动质量评估

### 输出
```json
{
  "success": true,
  "message": "成功添加儿童: 张三(1001)",
  "data": {
    "student_id": 1001,
    "child_name": "张三",
    "photos_processed": 2,
    "avg_quality": 0.85,
    "threshold": 0.55,
    "database_integration": true
  }
}
```

---

## 6. 批量人脸识别

### 接口信息
- **路径**: `POST /batch_recognize`
- **描述**: 批量识别多张图片中的儿童人脸

### 输入
```json
{
  "images": ["base64图片1", "base64图片2"],
  "class_id": [1, 1],
  "activity_detail": ["户外活动", "室内游戏"],
  "is_public": [true, false],
  "uploader_id": [1, 1],
  "max_workers": 4
}
```

**参数说明：**
- `images` (array, 必需): base64编码的图片列表，最多50张
- `class_id` (array, 必需): 班级ID列表，长度必须与images一致
- `activity_detail` (array, 必需): 活动描述列表，长度必须与images一致
- `is_public` (array, 必需): 是否公开列表，长度必须与images一致
- `uploader_id` (array, 必需): 上传者ID列表，长度必须与images一致
- `max_workers` (integer, 可选): 最大工作线程数，默认4

### 主要功能
- 批量处理多张图片
- 人脸检测和识别
- 自动上传到OSS存储
- 保存识别记录到数据库
- 支持并行处理

### 输出
```json
{
  "success": true,
  "data": {
    "total_processed": 2,
    "successful_recognitions": 2,
    "failed_recognitions": 0,
    "results": [
      {
        "image_index": 0,
        "student_id": 1001,
        "child_name": "张三",
        "confidence": 0.92,
        "oss_url": "https://bucket.oss-cn-shenzhen.aliyuncs.com/photo1.jpg",
        "object_key": "photos/2024/01/photo1.jpg",
        "photo_id": 123,
        "recognition_time": "2024-01-01T12:00:00"
      }
    ],
    "processing_time": 3.5
  }
}
```

---

## 7. 获取儿童档案

### 接口信息
- **路径**: `GET /child/{student_id}/profile`
- **描述**: 获取指定儿童的档案信息

### 输入
- `student_id` (path parameter): 儿童学号

### 主要功能
- 查询儿童基本信息
- 返回档案详情

### 输出
```json
{
  "success": true,
  "data": {
    "profile": {
      "student_id": 1001,
      "name": "张三",
      "age": 5,
      "class_id": 1
    }
  }
}
```

---

## 8. 获取更新需求

### 接口信息
- **路径**: `GET /maintenance/update_requirements`
- **描述**: 获取需要更新照片的儿童列表

### 输入
无参数

### 主要功能
- 检查照片更新需求
- 返回需要更新的儿童信息
- 提供维护建议

### 输出
```json
{
  "success": true,
  "data": {
    "total_requiring_update": 5,
    "children_needing_update": [
      {
        "child_name": 1001,
        "last_updated": "2024-01-01T10:00:00",
        "update_due_date": "2024-02-01T10:00:00",
        "age": 5,
        "photos_count": 2
      }
    ]
  }
}
```

---

## 9. 获取系统统计

### 接口信息
- **路径**: `GET /system/statistics`
- **描述**: 获取系统统计信息和性能指标

### 输入
无参数

### 主要功能
- 返回数据库统计信息
- 显示识别性能指标
- 提供系统版本信息

### 输出
```json
{
  "success": true,
  "data": {
    "database_statistics": {
      "total_children": 100,
      "total_photos": 500,
      "avg_quality": 0.75
    },
    "recognition_statistics": {
      "total_recognitions": 1000,
      "success_rate": 0.95,
      "avg_confidence": 0.88
    },
    "system_info": {
      "version": "1.0_child_face_recognition_system",
      "config": {...},
      "timestamp": "2024-01-01T12:00:00"
    }
  }
}
```

---

## 10. 删除儿童

### 接口信息
- **路径**: `POST /database/delete_child`
- **描述**: 删除指定儿童的所有数据

### 输入
```json
{
  "student_id": 1001
}
```

**参数说明：**
- `student_id` (integer, 必需): 要删除的儿童学号

### 主要功能
- 删除儿童的人脸数据
- 清理相关照片记录
- 移除数据库记录

### 输出
```json
{
  "success": true,
  "message": "成功删除儿童: 1001"
}
```

---

## 11. 删除图片

### 接口信息
- **路径**: `POST /database/delete_image`
- **描述**: 删除OSS上的图片和数据库记录

### 输入
```json
{
  "object_keys": ["photos/2024/01/photo1.jpg", "photos/2024/01/photo2.jpg"],
  "ids": [123, 124]
}
```

**参数说明：**
- `object_keys` (array, 必需): OSS对象键列表
- `ids` (array, 必需): 数据库记录ID列表，长度必须与object_keys一致

### 主要功能
- 从OSS删除图片文件
- 删除数据库中的照片记录
- 清理关联关系

### 输出
```json
{
  "success": true,
  "message": "成功删除图片",
  "data": {
    "oss_deleted": 2,
    "oss_failed": 0,
    "db_photos_deleted": 2,
    "db_relations_deleted": 2,
    "total_processed": 2
  }
}
```

---

## 错误处理

### 通用错误响应格式
```json
{
  "success": false,
  "error": "错误描述信息"
}
```

### 常见错误码
- `400`: 请求参数错误
- `404`: 资源不存在
- `413`: 请求实体过大
- `500`: 服务器内部错误

### 特殊错误处理
- **413错误**: 上传文件过大
- **参数长度不匹配**: 批量接口中数组长度不一致
- **图片格式错误**: 不支持的图片格式
- **模型未加载**: 人脸识别模型未初始化

---

## 使用示例

### 完整的儿童管理流程
1. **添加儿童**
```bash
curl -X POST http://localhost:5000/database/add_child \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "小明",
    "images": ["base64图片数据"],
    "profile": {
      "student_id": 1002,
      "age": 4,
      "class_id": 1
    }
  }'
```

2. **批量识别**
```bash
curl -X POST http://localhost:5000/batch_recognize \
  -H 'Content-Type: application/json' \
  -d '{
    "images": ["base64图片1", "base64图片2"],
    "class_id": [1, 1],
    "activity_detail": ["户外活动", "室内游戏"],
    "is_public": [true, false],
    "uploader_id": [1, 1]
  }'
```

3. **查询结果**
```bash
curl http://localhost:5000/database/children
```

---

## 注意事项

1. **图片格式**: 支持JPEG、PNG、JPG格式，建议使用JPEG格式
2. **图片质量**: 系统会自动评估图片质量，质量过低会被拒绝
3. **批量处理**: 批量识别最多支持50张图片
4. **并发安全**: 系统支持多线程并发处理
5. **数据一致性**: 所有数组参数的长度必须一致
6. **存储限制**: 单次请求最大500MB
7. **模型加载**: 首次调用可能需要等待模型加载完成 