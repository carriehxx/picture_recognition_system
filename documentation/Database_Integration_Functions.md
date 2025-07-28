# 数据库集成模块函数功能文档

## 概述

`database_integration.py` 是儿童人脸识别系统的数据库集成模块，负责整合MySQL数据库和阿里云OSS对象存储，提供完整的数据持久化解决方案。

**主要功能：**
- MySQL数据库连接和操作
- 阿里云OSS对象存储管理
- 儿童档案管理
- 人脸特征向量存储
- 图片识别记录管理
- 数据清理和维护

---

## 类定义

### DatabaseConfig
数据库配置类，管理MySQL连接参数。

**属性：**
- `host`: 数据库主机地址
- `port`: 数据库端口
- `database`: 数据库名称
- `user`: 用户名
- `password`: 密码
- `charset`: 字符集

---

### ChildFaceDatabaseIntegration
主要的数据库集成类，提供所有数据库操作功能。

---

## 核心函数详解

### 1. 初始化相关函数

#### `__init__(use_object_storage: bool = True)`
**功能**: 初始化数据库集成实例

**输入参数:**
- `use_object_storage` (bool): 是否使用对象存储，默认True

**主要实现:**
- 初始化数据库配置
- 创建对象存储管理器
- 测试数据库连接

**输出**: 无返回值，初始化实例

**异常处理:**
- 数据库连接失败时抛出异常
- 对象存储初始化失败时降级到本地存储

---

#### `_test_connection()`
**功能**: 测试MySQL数据库连接

**输入参数**: 无

**主要实现:**
- 尝试连接MySQL数据库
- 验证连接状态
- 记录连接结果

**输出**: 无返回值

**异常处理:**
- 连接失败时抛出异常并记录错误日志

---

#### `_get_connection()`
**功能**: 获取数据库连接实例

**输入参数**: 无

**主要实现:**
- 创建新的MySQL连接
- 配置连接参数

**输出**: MySQL连接对象

---

### 2. 数据序列化函数

#### `_serialize_embedding(embedding) -> bytes`
**功能**: 序列化人脸特征向量

**输入参数:**
- `embedding`: 人脸特征向量（PyTorch tensor或numpy数组）

**主要实现:**
- 检查输入是否为None
- 转换为numpy数组（如果是PyTorch tensor）
- 使用pickle序列化

**输出**: 序列化后的字节数据

**异常处理:**
- 输入为None时返回None

---

#### `_deserialize_embedding(data: bytes)`
**功能**: 反序列化人脸特征向量

**输入参数:**
- `data` (bytes): 序列化的特征向量数据

**主要实现:**
- 使用pickle反序列化数据
- 恢复原始特征向量格式

**输出**: 原始特征向量对象

**异常处理:**
- 输入为None时返回None

---

#### `_make_json_safe(obj)`
**功能**: 确保对象可以被JSON序列化

**输入参数:**
- `obj`: 任意对象

**主要实现:**
- 递归处理字典、列表等复杂对象
- 转换datetime对象为ISO格式字符串
- 将不支持的类型转换为字符串

**输出**: JSON安全的对象

---

### 4. 图片上传和识别记录函数

#### `upload_recognized_photo(test_image_path: str, recognized_child_id: int = None, recognition_confidence: float = 0.0, session_id: str = None, class_id: int = None, activity_detail: str = None, is_public: bool = False, uploader_id: int = 1, category: str = 'originals') -> Dict[str, Any]`
**功能**: 上传识别图片到OSS并保存记录到数据库

**输入参数:**
- `test_image_path` (str): 图片文件路径
- `recognized_child_id` (int, 可选): 识别出的儿童ID
- `recognition_confidence` (float, 可选): 识别置信度
- `session_id` (str, 可选): 批量识别会话ID
- `class_id` (int, 可选): 班级ID
- `activity_detail` (str, 可选): 活动详情
- `is_public` (bool, 可选): 是否公开，默认False
- `uploader_id` (int, 可选): 上传者ID，默认1
- `category` (str, 可选): 图片类别，默认'originals'

**主要实现:**
- 生成标准化文件名
- 上传图片到OSS存储
- 保存识别记录到数据库
- 建立照片与儿童的关联关系

**输出**: 包含上传结果的字典
```json
{
  "photo_id": 123,
  "oss_url": "https://bucket.oss-cn-shenzhen.aliyuncs.com/photo1.jpg",
  "object_key": "photos/2024/01/photo1.jpg",
  "file_size": 1024000,
  "recognized_child_id": 1001,
  "recognition_confidence": 0.92,
  "session_id": "session_123",
  "uploader_id": 1
}
```

**异常处理:**
- 上传失败时抛出异常
- 数据库操作失败时回滚事务

---

#### `_save_photo_record(oss_url: str, object_key: str, file_size: int, recognized_child_id: int = None, recognition_confidence: float = 0.0, session_id: str = None, class_id: int = None, activity_detail: str = None, is_public: bool = False, uploader_id: int = 1) -> int`
**功能**: 保存图片识别记录到数据库

**输入参数:**
- `oss_url` (str): OSS图片URL
- `object_key` (str): OSS对象键
- `file_size` (int): 文件大小
- `recognized_child_id` (int, 可选): 识别的儿童ID
- `recognition_confidence` (float, 可选): 识别置信度
- `session_id` (str, 可选): 会话ID
- `class_id` (int, 可选): 班级ID
- `activity_detail` (str, 可选): 活动详情
- `is_public` (bool, 可选): 是否公开
- `uploader_id` (int, 可选): 上传者ID

**主要实现:**
- 插入照片记录到photos表
- 构建识别数据JSON
- 建立照片与儿童的关联关系

**输出**: 照片记录ID（整数）

**数据库操作:**
```sql
INSERT INTO photos (path, object_key, uploader_id, class_id, is_public, activity, recognition_data, created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)

INSERT INTO photo_child (photo_id, child_id) VALUES (%s, %s)
```

---

### 5. 人脸特征向量管理函数

#### `save_face_embedding(student_id: int, embedding, photos_count: int, avg_quality: float, adaptive_threshold: float) -> int`
**功能**: 保存人脸特征向量到数据库

**输入参数:**
- `student_id` (int): 儿童学号
- `embedding`: 人脸特征向量
- `photos_count` (int): 照片数量
- `avg_quality` (float): 平均质量分数
- `adaptive_threshold` (float): 自适应阈值

**主要实现:**
- 序列化特征向量
- 检查是否已存在记录
- 更新现有记录或创建新记录
- 计算下次更新日期

**输出**: 特征记录ID（整数）

**数据库操作:**
```sql
-- 更新现有记录
UPDATE face_embeddings 
SET embedding_data = %s, photos_count = %s, avg_quality = %s, 
    adaptive_threshold = %s, last_updated = %s, update_due_date = %s
WHERE student_id = %s

-- 创建新记录
INSERT INTO face_embeddings (student_id, embedding_data, photos_count, avg_quality, 
                           adaptive_threshold, created_at, last_updated, update_due_date)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
```

---

### 6. 数据查询函数

#### `get_child_by_student_id(student_id: int) -> Dict[str, Any]`
**功能**: 根据学号获取儿童信息

**输入参数:**
- `student_id` (int): 儿童学号

**主要实现:**
- 查询children表
- 解析features字段中的JSON数据
- 合并基本信息

**输出**: 儿童信息字典或None

**数据库操作:**
```sql
SELECT * FROM children WHERE id = %s
```

---

#### `get_all_children_with_embeddings() -> List[Dict[str, Any]]`
**功能**: 获取所有有人脸特征的儿童信息

**输入参数**: 无

**主要实现:**
- 联表查询children和face_embeddings
- 解析features字段
- 反序列化特征向量

**输出**: 儿童信息列表

**数据库操作:**
```sql
SELECT c.*, f.embedding_data, f.photos_count, f.avg_quality, 
       f.adaptive_threshold, f.last_updated, f.update_due_date
FROM children c
INNER JOIN face_embeddings f ON c.id = f.student_id
ORDER BY c.name
```

---

#### `get_children_needing_update() -> List[Dict[str, Any]]`
**功能**: 获取需要更新照片的儿童列表

**输入参数**: 无

**主要实现:**
- 查询更新日期已到的儿童
- 按更新日期排序
- 解析features字段

**输出**: 需要更新的儿童列表

**数据库操作:**
```sql
SELECT c.*, f.last_updated, f.update_due_date
FROM children c
INNER JOIN face_embeddings f ON c.id = f.student_id
WHERE f.update_due_date <= %s
ORDER BY f.update_due_date ASC
```

---

### 7. 数据删除函数

#### `delete_images(object_keys: List[str], ids: List[int]) -> Dict[str, Any]`
**功能**: 删除OSS上的图片和数据库记录

**输入参数:**
- `object_keys` (List[str]): OSS对象键列表
- `ids` (List[int]): 数据库记录ID列表

**主要实现:**
- 批量删除OSS上的图片文件
- 删除数据库中的照片记录
- 删除照片与儿童的关联关系
- 统计删除结果

**输出**: 删除结果字典
```json
{
  "success": true,
  "message": "成功删除 2 张图片",
  "oss_deleted": 2,
  "oss_failed": 0,
  "db_photos_deleted": 2,
  "db_relations_deleted": 2,
  "total_processed": 2
}
```

**数据库操作:**
```sql
DELETE FROM photo_child WHERE photo_id = %s
DELETE FROM photos WHERE id = %s
```

**异常处理:**
- OSS删除失败时记录警告
- 数据库删除失败时回滚事务

---

### 8. 工具函数

#### `get_database_integration() -> ChildFaceDatabaseIntegration`
**功能**: 获取数据库集成实例（单例模式）

**输入参数**: 无

**主要实现:**
- 检查全局实例是否存在
- 创建新实例（如果不存在）
- 返回单例实例

**输出**: 数据库集成实例

---

## 数据库表结构

### children表
存储儿童基本信息
```sql
CREATE TABLE children (
  id INT PRIMARY KEY AUTO_INCREMENT,
  student_id INT UNIQUE,
  name VARCHAR(100) NOT NULL,
  class_id INT NOT NULL,
  age INT DEFAULT NULL,
  features TEXT DEFAULT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### photos表
存储照片记录
```sql
CREATE TABLE photos (
  id INT PRIMARY KEY AUTO_INCREMENT,
  path VARCHAR(255) NOT NULL,
  object_key VARCHAR(255),
  uploader_id INT NOT NULL,
  class_id INT NOT NULL,
  is_public TINYINT(1) DEFAULT 1,
  activity TEXT DEFAULT NULL,
  recognition_data JSON DEFAULT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### photo_child表
照片与儿童的关联表
```sql
CREATE TABLE photo_child (
  photo_id INT NOT NULL,
  child_id INT NOT NULL,
  PRIMARY KEY (photo_id, child_id)
);
```

### face_embeddings表
存储人脸特征向量
```sql
CREATE TABLE face_embeddings (
  id INT PRIMARY KEY AUTO_INCREMENT,
  student_id INT UNIQUE,
  embedding_data LONGBLOB,
  photos_count INT DEFAULT 1,
  avg_quality FLOAT DEFAULT 0.0,
  adaptive_threshold FLOAT DEFAULT 0.55,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  update_due_date TIMESTAMP
);
```

---

## 错误处理机制

### 1. 数据库连接错误
- 连接失败时抛出异常并记录错误日志
- 自动重试机制（通过_get_connection实现）

### 2. 事务管理
- 使用try-finally确保连接正确关闭
- 操作失败时自动回滚事务
- 支持批量操作的原子性

### 3. 数据验证
- 检查必需参数
- 验证数据类型和格式
- 处理空值和默认值

### 4. 异常分类
- **数据库错误**: 连接、查询、事务相关
- **存储错误**: OSS上传、删除相关
- **数据错误**: 序列化、反序列化相关
- **参数错误**: 输入参数验证相关

---

## 性能优化

### 1. 连接池管理
- 每次操作创建新连接
- 操作完成后立即关闭连接
- 避免连接泄漏

### 2. 批量操作
- 支持批量删除图片
- 使用executemany提高效率
- 减少数据库往返次数

### 3. 数据序列化
- 使用pickle进行高效序列化
- 支持PyTorch tensor和numpy数组
- 优化存储空间使用

### 4. 查询优化
- 使用索引字段进行查询
- 联表查询减少数据获取次数
- 按需加载特征向量数据

---

## 使用示例

### 基本使用流程
```python
# 1. 获取数据库集成实例
db_integration = get_database_integration()

# 2. 添加儿童档案
child_profile = ChildProfile(student_id=1001, name="张三", age=5, class_id=1)
child_id = db_integration.add_child_profile(child_profile)

# 3. 保存人脸特征
embedding = torch.randn(512)  # 示例特征向量
db_integration.save_face_embedding(1001, embedding, 3, 0.85, 0.55)

# 4. 上传识别图片
result = db_integration.upload_recognized_photo(
    test_image_path="path/to/image.jpg",
    recognized_child_id=1001,
    recognition_confidence=0.92,
    class_id=1,
    activity_detail="户外活动"
)

# 5. 查询儿童信息
children = db_integration.get_all_children_with_embeddings()

# 6. 删除图片
db_integration.delete_images(
    object_keys=["photos/2024/01/photo1.jpg"],
    ids=[123]
)
```

---

## 注意事项

1. **数据一致性**: 确保student_id在children和face_embeddings表中的一致性
2. **存储空间**: 特征向量数据较大，注意存储空间管理
3. **并发安全**: 多线程环境下注意连接管理
4. **错误恢复**: 重要操作建议添加重试机制
5. **数据备份**: 定期备份数据库和OSS数据
6. **性能监控**: 监控查询性能和存储使用情况 