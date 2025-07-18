#!/usr/bin/env python3
"""
数据库集成模块
整合MySQL和OSS，支持儿童人脸识别系统
"""

import mysql.connector
from mysql.connector import Error
import pickle
import base64
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import asdict
import uuid
import shutil

from child_face_system import ChildProfile
from storage_managers import get_storage_manager
from config.storage_config import get_storage_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """数据库配置"""
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', '3306'))
        self.database = os.getenv('DB_NAME', 'kindergarten_system')
        self.user = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', '')
        self.charset = 'utf8mb4'

class ChildFaceDatabaseIntegration:
    """儿童人脸识别数据库集成类"""
    
    def __init__(self, use_object_storage: bool = True):
        """
        初始化数据库集成
        
        Args:
            use_object_storage: 是否使用对象存储
        """
        self.db_config = DatabaseConfig()
        self.use_object_storage = use_object_storage
        
        # 初始化对象存储
        self.storage_manager = None
        if self.use_object_storage:
            try:
                self.storage_manager = get_storage_manager()
                logger.info("✅ 对象存储管理器初始化成功")
            except Exception as e:
                logger.warning(f"⚠️  对象存储初始化失败，将使用本地存储: {e}")
                self.use_object_storage = False
        
        # 测试数据库连接
        self._test_connection()
    
    def _test_connection(self):
        """测试数据库连接"""
        try:
            connection = mysql.connector.connect(**self._get_db_config())
            if connection.is_connected():
                logger.info("✅ MySQL数据库连接成功")
                connection.close()
        except Error as e:
            logger.error(f"❌ MySQL连接失败: {e}")
            raise
    
    def _get_db_config(self) -> Dict:
        """获取数据库配置"""
        return {
            'host': self.db_config.host,
            'port': self.db_config.port,
            'database': self.db_config.database,
            'user': self.db_config.user,
            'password': self.db_config.password,
            'charset': self.db_config.charset,
            'autocommit': True
        }
    
    def _get_connection(self):
        """获取数据库连接"""
        return mysql.connector.connect(**self._get_db_config())
    
    def _serialize_embedding(self, embedding) -> bytes:
        """序列化人脸特征向量"""
        if embedding is None:
            return None
        return pickle.dumps(embedding.numpy() if hasattr(embedding, 'numpy') else embedding)
    
    def _deserialize_embedding(self, data: bytes):
        """反序列化人脸特征向量"""
        if data is None:
            return None
        return pickle.loads(data)
    
    def _make_json_safe(self, obj):
        """确保对象可以被JSON序列化"""
        if obj is None:
            return {}
        
        if isinstance(obj, dict):
            safe_dict = {}
            for key, value in obj.items():
                if isinstance(key, str):
                    safe_dict[key] = self._make_json_safe(value)
            return safe_dict
        
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        
        elif isinstance(obj, datetime):
            return obj.isoformat()
        
        else:
            # 对于其他类型，转换为字符串
            return str(obj)
    
    def add_child_profile(self, child_profile: ChildProfile) -> int:
        """
        添加儿童档案到数据库
        
        Args:
            child_profile: 儿童档案信息
            
        Returns:
            创建的儿童ID
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            # 检查student_id是否已存在（如果数据库有student_id字段）
            try:
                class_id = child_profile.class_id
                check_query = "SELECT id FROM children WHERE student_id = %s"
                cursor.execute(check_query, (child_profile.student_id,))
                existing = cursor.fetchone()
                
                if existing:
                    logger.warning(f"学生ID {child_profile.student_id} 已存在，返回现有ID: {existing[0]}")
                    return existing[0]
                
                # 插入儿童基本信息（包含student_id）
                insert_query = """
                INSERT INTO children (student_id, name, class_id, age, features, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                
                values = (
                    child_profile.student_id,
                    child_profile.name,
                    class_id,
                    child_profile.age,
                    json.dumps({
                        'created_by': 'face_recognition_api',
                        'profile_type': 'basic'
                    }, ensure_ascii=False),
                    datetime.now()
                )
                
            except Error:
                # 如果student_id字段不存在，使用原来的查询
                logger.warning("数据库可能没有student_id字段，使用原始查询")
                insert_query = """
                INSERT INTO children (name, class_id, age, features, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """
                
                values = (
                    child_profile.name,
                    class_id,
                    child_profile.age,
                    json.dumps({
                        'student_id': child_profile.student_id,  # 存储在features中
                        'created_by': 'face_recognition_api',
                        'profile_type': 'basic_legacy'
                    }, ensure_ascii=False),
                    datetime.now()
                )
            
            cursor.execute(insert_query, values)
            child_id = cursor.lastrowid
            
            logger.info(f"✅ 成功添加儿童档案: {child_profile.name}({child_profile.student_id}) (ID: {child_id})")
            return child_id
            
        except Error as e:
            logger.error(f"❌ 添加儿童档案失败: {e}")
            raise
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def upload_recognized_photo(self, test_image_path: str, recognized_child_id: int = None, 
                          recognition_confidence: float = 0.0, session_id: str = None,
                          class_id: int = None, activity_detail: str = None, 
                          is_public: bool = False, uploader_id: int = 1) -> Dict[str, Any]:
        """
        上传测试图片到OSS并保存识别记录到数据库
        
        Args:
            test_image_path: 测试图片文件路径
            recognized_child_id: 识别出的儿童student_id（如果识别成功）
            recognition_confidence: 识别置信度
            session_id: 批量识别会话ID
            class_id: 班级ID
            activity_detail: 活动详情
            is_public: 是否为公开图片
            
        Returns:
            上传结果和数据库记录信息
        """
        try:
            # 获取文件信息
            filename = os.path.basename(test_image_path)
            file_size = os.path.getsize(test_image_path)
            
            # 生成标准化文件名（用于测试图片）
            timestamp = int(datetime.now().timestamp())
            file_extension = os.path.splitext(filename)[1] if '.' in filename else '.jpg'
            standard_filename = f"detectImg_{session_id or timestamp}_{timestamp}{file_extension}"
            
            # 上传到OSS
            oss_url = None
            object_key = None
            
            if self.use_object_storage and self.storage_manager:
                # 生成对象键（存储在detect_images目录）
                object_key = self.storage_manager.generate_object_key(
                    'detect_images', standard_filename, 'batch_recognition'
                )
                
                # 上传文件
                upload_result = self.storage_manager.upload_from_path(
                    test_image_path, 
                    object_key,
                    content_type='image/jpeg'
                )
                
                oss_url = upload_result.get('url')
                logger.info(f"✅ 测试图片已上传到OSS: {oss_url}")
            else:
                # 本地存储备用方案
                local_path = f"processed/{filename}"
                shutil.copy2(test_image_path, local_path)
                oss_url = local_path
                object_key = local_path
            
            # 保存识别记录到数据库
            # 处理 activity_detail（现在应该是字符串类型）
            safe_activity_detail = activity_detail or ""

            photo_id = self._save_photo_record(
                oss_url=oss_url,
                object_key=object_key,
                file_size=file_size,
                recognized_child_id=recognized_child_id,
                recognition_confidence=recognition_confidence,
                session_id=session_id,
                class_id=class_id,
                activity_detail=safe_activity_detail,
                is_public=is_public,
                uploader_id=uploader_id
            )
            
            return {
                'photo_id': photo_id,
                'oss_url': oss_url,
                'object_key': object_key,
                'file_size': file_size,
                'recognized_child_id': recognized_child_id,
                'recognition_confidence': recognition_confidence,
                'session_id': session_id,
                'uploader_id': uploader_id
            }
            
        except Exception as e:
            logger.error(f"❌ 上传测试图片失败: {e}")
            raise

    def _save_photo_record(self, oss_url: str, object_key: str, file_size: int,
                          recognized_child_id: int = None, recognition_confidence: float = 0.0,
                          session_id: str = None, class_id: int = None, 
                          activity_detail: str = None, is_public: bool = False,
                          uploader_id: int = 1) -> int:
        """保存测试图片识别记录到数据库"""
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            # 插入照片记录
            insert_query = """
            INSERT INTO photos (path, object_key, uploader_id, class_id, is_public, activity, recognition_data, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # 确保 activity_detail 可以被序列化（现在是字符串类型）
            safe_activity_detail = activity_detail or ""
            
            # 构建识别数据
            recognition_data = {
                'object_key': object_key,
                'file_size': file_size,
                'recognized_child_id': recognized_child_id,
                'recognition_confidence': recognition_confidence,
                'session_id': session_id,
                'is_test_image': True,  # 标记为测试图片
                'activity_detail': safe_activity_detail,  # 使用安全的版本
                'upload_timestamp': datetime.now().isoformat()
            }
            
            values = (
                oss_url,
                object_key,
                uploader_id,
                class_id or 0,  # 如果没有班级ID，默认为0
                1 if is_public else 0,
                safe_activity_detail,  # 直接存储字符串
                json.dumps(recognition_data, ensure_ascii=False),
                datetime.now()
            )
            
            cursor.execute(insert_query, values)
            photo_id = cursor.lastrowid
            
            # recognized_child_id 就是 children.id，直接建立关联关系
            if recognized_child_id:
                link_query = "INSERT INTO photo_child (photo_id, child_id) VALUES (%s, %s)"
                cursor.execute(link_query, (photo_id, recognized_child_id))
                logger.info(f"✅ 建立照片-儿童关联: Photo ID {photo_id} -> Child ID {recognized_child_id}")
            
            connection.commit()
            logger.info(f"✅ 测试图片识别记录已保存: Photo ID {photo_id}")
            return photo_id
            
        except Error as e:
            logger.error(f"❌ 保存测试图片记录失败: {e}")
            connection.rollback()
            raise
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def save_face_embedding(self, student_id: int, embedding, photos_count: int, 
                           avg_quality: float, adaptive_threshold: float) -> int:
        """
        保存人脸特征向量到数据库
        
        Args:
            student_id: 儿童ID
            embedding: 人脸特征向量
            photos_count: 照片数量
            avg_quality: 平均质量分数
            adaptive_threshold: 自适应阈值
            
        Returns:
            特征记录ID
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            # 序列化特征向量
            embedding_data = self._serialize_embedding(embedding)
            
            # 检查是否已存在该儿童的特征记录
            check_query = "SELECT id FROM face_embeddings WHERE student_id = %s"
            cursor.execute(check_query, (student_id,))
            existing_record = cursor.fetchone()
            
            if existing_record:
                # 更新现有记录
                update_query = """
                UPDATE face_embeddings 
                SET embedding_data = %s, photos_count = %s, avg_quality = %s, 
                    adaptive_threshold = %s, last_updated = %s, update_due_date = %s
                WHERE student_id = %s
                """
                
                # 计算下次更新日期（30天后）
                update_due_date = datetime.now() + timedelta(days=30)
                
                values = (
                    embedding_data,
                    photos_count,
                    avg_quality,
                    adaptive_threshold,
                    datetime.now(),
                    update_due_date,
                    student_id
                )
                
                cursor.execute(update_query, values)
                embedding_id = existing_record[0]
                logger.info(f"✅ 更新人脸特征记录: student ID {student_id}")
            else:
                # 创建新记录
                insert_query = """
                INSERT INTO face_embeddings (student_id, embedding_data, photos_count, avg_quality, 
                                           adaptive_threshold, created_at, last_updated, update_due_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # 计算下次更新日期
                update_due_date = datetime.now() + timedelta(days=30)
                
                values = (
                    student_id,
                    embedding_data,
                    photos_count,
                    avg_quality,
                    adaptive_threshold,
                    datetime.now(),
                    datetime.now(),
                    update_due_date
                )
                
                cursor.execute(insert_query, values)
                embedding_id = cursor.lastrowid
                logger.info(f"✅ 保存人脸特征记录: student ID {student_id}")
            
            return embedding_id
            
        except Error as e:
            logger.error(f"❌ 保存人脸特征失败: {e}")
            raise
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def get_child_by_student_id(self, student_id: int) -> Dict[str, Any]:
        """根据学号获取儿童信息"""
        try:
            connection = self._get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # 直接根据id查找，因为student_id就是children.id
            query = "SELECT * FROM children WHERE id = %s"
            cursor.execute(query, (student_id,))
            result = cursor.fetchone()
            
            if result and result.get('features'):
                # 解析features字段
                features_data = json.loads(result['features'])
                result.update(features_data)
            
            return result
            
        except Error as e:
            logger.error(f"❌ 根据学号获取儿童信息失败: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def get_all_children_with_embeddings(self) -> List[Dict[str, Any]]:
        """获取所有有人脸特征的儿童信息"""
        try:
            connection = self._get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = """
            SELECT c.*, f.embedding_data, f.photos_count, f.avg_quality, 
                   f.adaptive_threshold, f.last_updated, f.update_due_date
            FROM children c
            INNER JOIN face_embeddings f ON c.id = f.student_id
            ORDER BY c.name
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            for result in results:
                if result['features']:
                    features_data = json.loads(result['features'])
                    result.update(features_data)
                
                # 反序列化特征向量
                if result['embedding_data']:
                    result['embedding'] = self._deserialize_embedding(result['embedding_data'])
            
            return results
            
        except Error as e:
            logger.error(f"❌ 获取儿童列表失败: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def save_recognition_record(self, test_image_path: str, recognized_child_id: Optional[int],
                               confidence: float, distance: float, threshold_used: float,
                               recognition_method: str, face_quality_score: float,
                               processing_time_ms: int, session_id: str = None) -> int:
        """
        保存识别记录
        
        Args:
            test_image_path: 测试图片路径
            recognized_child_id: 识别出的儿童ID
            confidence: 识别置信度
            distance: 特征距离
            threshold_used: 使用的阈值
            recognition_method: 识别方法
            face_quality_score: 人脸质量分数
            processing_time_ms: 处理时间（毫秒）
            session_id: 会话ID
            
        Returns:
            识别记录ID
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            # 确定识别状态
            if recognized_child_id is not None:
                status = 'success'
            elif confidence > 0:
                status = 'unknown'
            else:
                status = 'failed'
            
            # 生成会话ID
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            insert_query = """
            INSERT INTO recognition_records (test_image_url, recognized_child_id, confidence, 
                                           distance, threshold_used, recognition_method, 
                                           face_quality_score, processing_time_ms, recognition_status, 
                                           session_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                test_image_path,
                recognized_child_id,
                confidence,
                distance,
                threshold_used,
                recognition_method,
                face_quality_score,
                processing_time_ms,
                status,
                session_id,
                datetime.now()
            )
            
            cursor.execute(insert_query, values)
            record_id = cursor.lastrowid
            
            logger.info(f"✅ 识别记录已保存: Record ID {record_id}")
            return record_id
            
        except Error as e:
            logger.error(f"❌ 保存识别记录失败: {e}")
            raise
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def get_children_needing_update(self) -> List[Dict[str, Any]]:
        """获取需要更新照片的儿童列表"""
        try:
            connection = self._get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = """
            SELECT c.*, f.last_updated, f.update_due_date
            FROM children c
            INNER JOIN face_embeddings f ON c.id = f.student_id
            WHERE f.update_due_date <= %s
            ORDER BY f.update_due_date ASC
            """
            
            cursor.execute(query, (datetime.now(),))
            results = cursor.fetchall()
            
            for result in results:
                if result['features']:
                    features_data = json.loads(result['features'])
                    result.update(features_data)
            
            return results
            
        except Error as e:
            logger.error(f"❌ 获取更新需求失败: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def cleanup_old_records(self, days: int = 90):
        """清理旧的识别记录"""
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            delete_query = "DELETE FROM recognition_records WHERE created_at < %s"
            cursor.execute(delete_query, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            logger.info(f"✅ 清理了 {deleted_count} 条旧识别记录")
            
            return deleted_count
            
        except Error as e:
            logger.error(f"❌ 清理记录失败: {e}")
            return 0
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def delete_images(self, object_keys: List[str], ids: List[int]) -> Dict[str, Any]:
        """
        根据图片path和ID删除图片
        
        Args:
            object_keys: OSS对象键列表
            ids: 数据库记录ID列表
            
        Returns:
            删除结果信息
        """
        try:
            # 统计OSS删除结果
            oss_deleted = 0
            oss_failed = 0
            
            # 删除OSS上的图片
            for object_key in object_keys:
                if self.storage_manager.delete_file(object_key):
                    oss_deleted += 1
                    logger.info(f"✅ OSS图片删除成功: {object_key}")
                else:
                    oss_failed += 1
                    logger.warning(f"⚠️ OSS图片删除失败: {object_key}")
            
            # 删除数据库记录
            connection = self._get_connection()
            cursor = connection.cursor()
            
            try:
                # 删除相关的识别记录
                delete_query = "DELETE FROM photo_child WHERE photo_id = %s"
                cursor.executemany(delete_query, [(id,) for id in ids])
                relations_deleted = cursor.rowcount
                
                # 删除图片记录
                delete_query = "DELETE FROM photos WHERE id = %s"
                cursor.executemany(delete_query, [(id,) for id in ids])
                photos_deleted = cursor.rowcount
                
                # 提交事务
                connection.commit()
                
                logger.info(f"✅ 数据库记录删除完成: {photos_deleted} 张图片, {relations_deleted} 个关联记录")
                
                return {
                    'success': True,
                    'message': f'成功删除 {photos_deleted} 张图片',
                    'oss_deleted': oss_deleted,
                    'oss_failed': oss_failed,
                    'db_photos_deleted': photos_deleted,
                    'db_relations_deleted': relations_deleted,
                    'total_processed': len(object_keys)
                }
                
            except Error as e:
                # 回滚事务
                connection.rollback()
                logger.error(f"❌ 数据库删除失败: {e}")
                return {
                    'success': False,
                    'error': f'数据库删除失败: {e}',
                    'oss_deleted': oss_deleted,
                    'oss_failed': oss_failed
                }

        except Exception as e:
            logger.error(f"❌ 删除图片失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
       
# 全局数据库实例
db_integration = None

def get_database_integration() -> ChildFaceDatabaseIntegration:
    """获取数据库集成实例（单例模式）"""
    global db_integration
    if db_integration is None:
        db_integration = ChildFaceDatabaseIntegration()
    return db_integration 