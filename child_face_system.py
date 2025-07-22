"""
儿童人脸识别系统 - 重构版本
职责分离：
- ChildFaceDatabase: 负责构建和管理儿童人脸数据库
- ChildFaceRecognizer: 负责批量识别和匹配

优化特性：
- 使用余弦相似度进行人脸匹配（比欧几里得距离更适合人脸识别）
- 自适应相似度阈值根据儿童年龄调整
- 多模型特征融合提高识别准确性
"""

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch.nn.functional as F
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import glob
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from dataclasses import dataclass
import shutil
from datetime import datetime, timedelta
import json
import cv2
from collections import defaultdict
import logging
import functools
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

CHILD_CONFIG = {
    # MTCNN 参数
    'mtcnn': {
        'image_size': 160,
        'margin': 30,
        'min_face_size': 25,
        'thresholds': [0.4, 0.4, 0.4],  # 降低阈值，提高儿童人脸检测率
        'factor': 0.709,                  # 降低缩放因子，检测更小的人脸
        'keep_all': True,
        'post_process': True
    },
    
    # 识别参数 
    'recognition': {
        'default_threshold': 0.55,  # 余弦相似度阈值
        'quality_threshold': 0.6,
        'min_face_size': 60,
        'max_age_days': 180,
    },
    
    # 质量评估
    'quality': {
        'min_face_size': 60,
        'max_blur_variance': 120,
        'min_brightness': 40,
        'max_brightness': 220,
        'pose_threshold': 35,
        'expression_tolerance': 0.8
    },
    
    # 年龄分组阈值（余弦相似度）
    'age_thresholds': {
        'infant': (0, 2, 0.65),      # 婴儿期，相似度阈值0.65（最严格）
        'toddler': (2, 5, 0.6),      # 幼儿期，相似度阈值0.6
        'child': (5, 10, 0.55),      # 儿童期，相似度阈值0.55
        'preteen': (10, 13, 0.5)     # 青春期前，相似度阈值0.5（最宽松）
    }
}

@dataclass
class ChildProfile:
    """儿童档案信息"""
    student_id: int  # 学号，作为唯一标识符
    name: str        # 姓名，用于显示
    age: int         # 年龄
    class_id: int    # 班级ID

@dataclass
class FaceQualityResult:
    """人脸质量评估结果"""
    quality_score: float
    confidence: float
    face_size: Tuple[int, int]
    brightness: float
    blur_variance: float
    pose_angle: float
    issues: List[str]
    is_suitable_for_children: bool

class ChildFaceDatabase:
    """
    儿童人脸数据库管理类
    职责：
    1. 建立和调取数据库
    2. 读取数据库信息
    3. 添加新的儿童（通过识别图片中最高置信度的人脸）
    4. 存储新的儿童人脸数据
    """
    
    def __init__(self, storage_path: str = "storage", 
                 database_path: str = "face_database.pkl",
                 profiles_path: str = "child_profiles.json",
                 enable_db_integration: bool = True):
        """
        初始化儿童人脸数据库
        
        Args:
            storage_path: 儿童照片存储路径
            database_path: 数据库文件路径
            profiles_path: 儿童档案文件路径
            enable_db_integration: 是否启用数据库集成（MySQL）
        """
        self.storage_path = storage_path
        self.database_path = os.path.join('results', database_path)
        self.profiles_path = os.path.join('results', profiles_path)
        
        # 初始化数据库集成
        self.db_integration = None
        if enable_db_integration:
            try:
                from database_integration import ChildFaceDatabaseIntegration
                self.db_integration = ChildFaceDatabaseIntegration()
                logger.info("✅ 数据库集成模块已启用")
            except Exception as e:
                logger.warning(f"⚠️ 数据库集成模块初始化失败: {e}")
                logger.warning("将使用本地存储模式")
        
        # 初始化人脸检测和特征提取模型
        self._init_models()
        
        # 数据存储
        self.face_embeddings: Dict[int, Dict] = {}  # {student_id: {embedding, metadata}}
        self.child_profiles: Dict[int, ChildProfile] = {}  # {student_id: ChildProfile}
        
        # 确保目录存在
        self._ensure_directories_exist()
        
        # 加载数据库
        self.load_database()
        self.load_profiles()
        
        logger.info(f"儿童人脸数据库初始化完成，当前有 {len(self.face_embeddings)} 个儿童档案")
    
    def _init_models(self):
        """初始化人脸检测和特征提取模型"""
        mtcnn_config = CHILD_CONFIG['mtcnn']
        self.mtcnn = MTCNN(
            image_size=mtcnn_config['image_size'],
            margin=mtcnn_config['margin'],
            min_face_size=mtcnn_config['min_face_size'],
            thresholds=mtcnn_config['thresholds'],
            factor=mtcnn_config['factor'],
            keep_all=mtcnn_config['keep_all'],
            post_process=mtcnn_config['post_process'],
            device=device
        )
        
        # 使用多个预训练模型进行集成
        self.models = {
            'vggface2': InceptionResnetV1(pretrained='vggface2').eval().to(device),
            'casia': InceptionResnetV1(pretrained='casia-webface').eval().to(device)
        }
    
    def _ensure_directories_exist(self):
        """确保所有必要目录存在"""
        directories = [
            self.storage_path,
            os.path.join(self.storage_path, "active"),
            os.path.join(self.storage_path, "archived"),
            "results",
            "logs"
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"创建目录: {directory}")
    
    def evaluate_child_face_quality(self, img_path: str, face_tensor: torch.Tensor, 
                                  box: np.ndarray, prob: float) -> FaceQualityResult:
        """评估儿童人脸质量"""
        config = CHILD_CONFIG['quality']
        quality_score = 0.0
        issues = []
        
        # 1. 检测置信度评估
        if prob < 0.6:
            issues.append(f"检测置信度偏低: {prob:.3f}")
        else:
            quality_score += 0.25
        
        # 2. 人脸尺寸评估
        face_width = box[2] - box[0] if box is not None else 0
        face_height = box[3] - box[1] if box is not None else 0
        min_size = min(face_width, face_height)
        
        if min_size < config['min_face_size']:
            issues.append(f"人脸尺寸过小: {min_size:.1f}px")
        elif min_size > 80:
            quality_score += 0.25
        else:
            quality_score += 0.15
        
        # 3. 图像清晰度评估
        blur_variance = 0
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                blur_variance = cv2.Laplacian(img, cv2.CV_64F).var()
                if blur_variance < config['max_blur_variance']:
                    issues.append(f"图像模糊: {blur_variance:.1f}")
                else:
                    quality_score += 0.20
        except Exception as e:
            logger.warning(f"无法计算模糊度: {e}")
        
        # 4. 亮度评估
        brightness = float(torch.mean(face_tensor))
        if brightness < config['min_brightness'] or brightness > config['max_brightness']:
            issues.append(f"亮度不适宜: {brightness:.1f}")
        else:
            quality_score += 0.15
        
        # 5. 姿态评估
        pose_angle = self._estimate_pose_angle(box)
        if pose_angle > config['pose_threshold']:
            issues.append(f"姿态偏差过大: {pose_angle:.1f}°")
        else:
            quality_score += 0.15
        
        # 儿童特殊适应性检查
        is_suitable_for_children = (
            len(issues) <= 2 and
            quality_score >= 0.5 and
            min_size >= config['min_face_size']
        )
        
        return FaceQualityResult(
            quality_score=quality_score,
            confidence=prob,
            face_size=(int(face_width), int(face_height)),
            brightness=brightness,
            blur_variance=blur_variance,
            pose_angle=pose_angle,
            issues=issues,
            is_suitable_for_children=is_suitable_for_children
        )
    
    def _estimate_pose_angle(self, box: np.ndarray) -> float:
        """估算人脸姿态角度"""
        if box is None:
            return 0.0
        
        width = box[2] - box[0]
        height = box[3] - box[1]
        aspect_ratio = width / height
        
        normal_ratio = 0.75
        deviation = abs(aspect_ratio - normal_ratio)
        estimated_angle = min(deviation * 60, 45)
        
        return estimated_angle
    
    def extract_best_face_embedding(self, img_path: str) -> Tuple[Optional[torch.Tensor], FaceQualityResult]:
        """
        提取图片中最佳人脸特征（用于构建数据库）
        
        Args:
            img_path: 图片路径
            
        Returns:
            Tuple[embedding, quality_result]: 特征向量和质量评估结果
        """
        try:
            img = Image.open(img_path).convert('RGB')
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None or len(boxes) == 0:
                logger.warning(f"未检测到人脸: {img_path}")
                return None, FaceQualityResult(0, 0, (0, 0), 0, 0, 0, ["未检测到人脸"], False)
            
            # 选择置信度最高的人脸
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            prob = probs[best_idx]
            
            # 提取人脸
            face_tensor = self.mtcnn.extract(img, [box], save_path=None)
            if face_tensor is None or len(face_tensor) == 0:
                logger.warning(f"人脸提取失败: {img_path}")
                return None, FaceQualityResult(0, prob, (0, 0), 0, 0, 0, ["人脸提取失败"], False)
            
            face_tensor = face_tensor[0].unsqueeze(0).to(device)
            
            # 质量评估
            quality_result = self.evaluate_child_face_quality(img_path, face_tensor, box, prob)
            
            # 如果质量不适合儿童识别，返回None
            if not quality_result.is_suitable_for_children:
                logger.warning(f"人脸质量不适合儿童识别: {img_path}, 问题: {quality_result.issues}")
                return None, quality_result
            
            # 多模型特征提取和融合
            embeddings = []
            for model_name, model in self.models.items():
                with torch.no_grad():
                    embedding = model(face_tensor)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    embeddings.append(embedding)
            
            # 特征融合（加权平均）
            weights = [1, 0]  # vggface2权重更高，对儿童效果更好
            final_embedding = sum(w * emb for w, emb in zip(weights, embeddings))
            final_embedding = F.normalize(final_embedding, p=2, dim=1)
            
            logger.info(f"成功提取儿童人脸特征: {img_path}, 质量分数: {quality_result.quality_score:.2f}")
            return final_embedding.cpu(), quality_result
            
        except Exception as e:
            logger.error(f"提取人脸特征时出错: {img_path}, 错误: {e}")
            return None, FaceQualityResult(0, 0, (0, 0), 0, 0, 0, [f"处理错误: {str(e)}"], False)
    
    def add_child(self, img_paths: Union[str, List[str]], child_profile: ChildProfile, 
                  update_if_exists: bool = True) -> bool:
        """
        添加新儿童到数据库
        
        Args:
            img_paths: 图片路径（单张或多张）
            child_profile: 儿童档案信息
            update_if_exists: 如果已存在是否更新
            
        Returns:
            bool: 是否成功添加
        """
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        
        student_id = child_profile.student_id
        child_name = child_profile.name
        
        # 检查是否已存在
        if student_id in self.face_embeddings and not update_if_exists:
            logger.warning(f"学生 {child_name}({student_id}) 已存在，设置 update_if_exists=True 来更新")
            return False
        
        logger.info(f"添加儿童: {child_name}({student_id})，照片数量: {len(img_paths)}")
        
        embeddings = []
        quality_scores = []
        
        # 处理所有照片，提取最佳人脸
        for img_path in img_paths:
            if not os.path.exists(img_path):
                logger.warning(f"图片不存在: {img_path}")
                continue
            
            embedding, quality_result = self.extract_best_face_embedding(img_path)
            
            if embedding is not None:
                embeddings.append(embedding)
                quality_scores.append(quality_result.quality_score)
                logger.info(f"✅ 成功处理照片: {img_path}")
            else:
                logger.warning(f"❌ 照片处理失败: {img_path}")
        
        if not embeddings:
            logger.error(f"没有成功处理的照片，无法添加儿童: {child_name}({student_id})")
            return False
        
        # 特征融合（取平均）
        if len(embeddings) == 1:
            final_embedding = embeddings[0]
        else:
            # 根据质量分数加权平均
            weights = torch.tensor(quality_scores, dtype=torch.float32)
            weights = weights / weights.sum()  # 归一化
            
            weighted_embeddings = []
            for i, embedding in enumerate(embeddings):
                weighted_embeddings.append(embedding * weights[i])
            
            final_embedding = torch.stack(weighted_embeddings).sum(dim=0)
            final_embedding = F.normalize(final_embedding, p=2, dim=1)
        
        # 保存到本地数据库
        self.face_embeddings[student_id] = {
            'embedding': final_embedding,
            'profile': child_profile,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'photos_count': len(embeddings),
            'avg_quality': np.mean(quality_scores),
            'update_due_date': (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        self.child_profiles[student_id] = child_profile
        
        # 保存到MySQL数据库（如果启用）
        if self.db_integration:
            try:
                self.db_integration.save_face_embedding(
                    student_id,
                    final_embedding,
                    len(embeddings),
                    np.mean(quality_scores),
                    self.get_adaptive_threshold(student_id)
                )
                logger.info(f"✅ 儿童数据已保存到MySQL数据库: {student_id}")
            except Exception as e:
                logger.warning(f"保存到MySQL数据库失败: {e}")
        
        # 自动保存本地数据库
        self.save_database()
        self.save_profiles()
        
        logger.info(f"✅ 成功添加儿童 {child_name}({student_id})，平均质量分数: {np.mean(quality_scores):.2f}")
        return True
    
    def get_adaptive_threshold(self, student_id: int) -> float:
        """根据儿童年龄获取自适应识别阈值"""
        if student_id not in self.child_profiles:
            return CHILD_CONFIG['recognition']['default_threshold']
        
        age = self.child_profiles[student_id].age
        
        # 根据年龄组确定阈值
        for group_name, (min_age, max_age, threshold) in CHILD_CONFIG['age_thresholds'].items():
            if min_age <= age < max_age:
                return threshold
        
        return CHILD_CONFIG['recognition']['default_threshold']
    
    def get_children_statistics(self) -> Dict:
        """获取儿童数据库统计信息"""
        stats = {
            'total_children': len(self.face_embeddings),
            'age_distribution': defaultdict(int),
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'update_required': len(self.check_update_requirements()),
            'avg_photos_per_child': 0
        }
        
        if not self.face_embeddings:
            return stats
        
        total_photos = 0
        quality_scores = []
        
        for student_id, data in self.face_embeddings.items():
            # 年龄分布
            if student_id in self.child_profiles:
                age = self.child_profiles[student_id].age
                if age <= 3:
                    stats['age_distribution']['0-3岁'] += 1
                elif age <= 6:
                    stats['age_distribution']['4-6岁'] += 1
                elif age <= 10:
                    stats['age_distribution']['7-10岁'] += 1
                else:
                    stats['age_distribution']['11岁以上'] += 1
            
            # 照片统计
            total_photos += data.get('photos_count', 1)
            
            # 质量分布
            quality = data.get('avg_quality', 0)
            quality_scores.append(quality)
            if quality >= 0.8:
                stats['quality_distribution']['high'] += 1
            elif quality >= 0.6:
                stats['quality_distribution']['medium'] += 1
            else:
                stats['quality_distribution']['low'] += 1
        
        stats['avg_photos_per_child'] = total_photos / len(self.face_embeddings)
        stats['avg_quality_score'] = np.mean(quality_scores) if quality_scores else 0
        
        return stats
    
    def check_update_requirements(self) -> List[int]:
        """检查需要更新照片的儿童"""
        need_update = []
        current_time = datetime.now()
        
        for student_id, data in self.face_embeddings.items():
            update_due_date = datetime.fromisoformat(data.get('update_due_date', current_time.isoformat()))
            if current_time >= update_due_date:
                need_update.append(student_id)
        
        return need_update
    
    def save_database(self):
        """保存数据库"""
        try:
            with open(self.database_path, 'wb') as f:
                pickle.dump({
                    'face_embeddings': self.face_embeddings,
                    'version': '2.0_child_optimized',
                    'saved_at': datetime.now().isoformat()
                }, f)
            logger.info(f"数据库已保存: {self.database_path}")
        except Exception as e:
            logger.error(f"保存数据库失败: {e}")
    
    def load_database(self):
        """加载数据库"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.face_embeddings = data.get('face_embeddings', {})
                    logger.info(f"数据库已加载: {len(self.face_embeddings)} 个儿童档案")
        except Exception as e:
            logger.error(f"加载数据库失败: {e}")
            self.face_embeddings = {}
    
    def save_profiles(self):
        """保存儿童档案"""
        try:
            profiles_data = {}
            for student_id, profile in self.child_profiles.items():
                # 这里使用字符串键是因为JSON规范要求字典的key必须为字符串类型。
                # 如果用int类型，json.dump会自动转为字符串，读取时需要注意转换回int。
                # 只要在load_profiles时做类型转换（如int(student_id)），不会有问题。
                profiles_data[str(student_id)] = {
                    'student_id': int(student_id),
                    'name': profile.name,
                    'age': profile.age,
                    'class_id': profile.class_id
                }
            
            with open(self.profiles_path, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, ensure_ascii=False, indent=2)
            logger.info(f"儿童档案已保存: {self.profiles_path}")
        except Exception as e:
            logger.error(f"保存儿童档案失败: {e}")
    
    def delete_child(self, student_id: int):
        """删除儿童"""
        if student_id in self.face_embeddings:
            del self.face_embeddings[student_id]
            del self.child_profiles[student_id]
            self.save_database()
            self.save_profiles()
            # 在数据库中移除这个学生的embedding，删除关于这个学生的匹配记录
            # 这个重新考虑一下recognize data的逻辑，因为里面包含了识别的child的信息，但其实没必要
    
    def load_profiles(self):
        """加载儿童档案"""
        try:
            if os.path.exists(self.profiles_path):
                with open(self.profiles_path, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    
                for student_id, data in profiles_data.items():
                    if 'student_id' not in data:
                        data['student_id'] = int(student_id)
                    
                    # 确保 student_id 是整数类型
                    student_id_int = int(student_id) 
                    profile_data = {
                        'student_id': student_id_int,
                        'name': data.get('name', 'Unknown'),
                        'age': data.get('age', 0),
                        'class_id': data.get('class_id', 0)
                    }
                    self.child_profiles[student_id_int] = ChildProfile(**profile_data)
                logger.info(f"儿童档案已加载: {len(self.child_profiles)} 个档案")
        except Exception as e:
            logger.error(f"加载儿童档案失败: {e}")
            self.child_profiles = {}

class ChildFaceRecognizer:
    """
    儿童人脸识别器
    职责：
    1. 更新数据缓存
    2. 并行处理批量图片
    3. 检测图片中的所有人脸
    4. 识别所有人脸
    5. 识别成功的图片存到OSS
    6. 存储数据到数据库
    """
    
    def __init__(self, face_database: ChildFaceDatabase, global_threshold: float = None):
        """
        初始化儿童人脸识别器
        
        Args:
            face_database: 儿童人脸数据库
            global_threshold: 全局阈值（如果不设置则使用自适应阈值）
        """
        self.face_database = face_database
        self.global_threshold = global_threshold
        self._embedding_cache = {}   # 特征向量缓存
        self._last_cache_update = datetime.now()
        
        logger.info(f"儿童人脸识别器初始化完成，数据库包含 {len(face_database.face_embeddings)} 个儿童")
    
    def _update_embedding_cache(self):
        """更新特征向量缓存以提高识别速度"""
        current_time = datetime.now()
        should_update = (
            len(self._embedding_cache) == 0 or
            (current_time - self._last_cache_update).seconds > 300
        )
        
        if should_update:
            self._embedding_cache.clear()
            for student_id, data in self.face_database.face_embeddings.items():
                embedding = data['embedding']
                
                # 确保特征向量维度正确 (应该是 [1, 512])
                if embedding.dim() == 3:  # [1, 1, 512]
                    embedding = embedding.squeeze(0)  # 变成 [1, 512]
                elif embedding.dim() == 1:  # [512]
                    embedding = embedding.unsqueeze(0)  # 变成 [1, 512]
                
                # 验证维度
                if embedding.size(1) != 512:
                    logger.warning(f"学生 {student_id} 的特征向量维度异常: {embedding.size()}")
                    continue
                
                self._embedding_cache[student_id] = embedding
            
            self._last_cache_update = current_time
            logger.info(f"特征向量缓存已更新，包含 {len(self._embedding_cache)} 个学生")
        else:
            logger.debug(f"缓存无需更新，当前包含 {len(self._embedding_cache)} 个学生")
    
    def extract_all_faces_for_recognition(self, img_path: str) -> List[Tuple[torch.Tensor, FaceQualityResult, np.ndarray, float]]:
        """
        提取图片中所有人脸特征（用于识别）
        
        Args:
            img_path: 图片路径
            
        Returns:
            List[Tuple[embedding, quality_result, box, prob]]: 所有人脸的特征向量、质量评估、边界框和置信度
        """
        try:
            img = Image.open(img_path).convert('RGB')
            boxes, probs = self.face_database.mtcnn.detect(img)
            
            if boxes is None or len(boxes) == 0:
                logger.warning(f"未检测到人脸: {img_path}")
                return []
            
            results = []
            
            # 处理所有检测到的人脸
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                try:
                    # 提取人脸
                    face_tensor = self.face_database.mtcnn.extract(img, [box], save_path=None)
                    if face_tensor is None or len(face_tensor) == 0:
                        logger.warning(f"人脸 {i+1} 提取失败: {img_path}")
                        continue
                    
                    face_tensor = face_tensor[0].unsqueeze(0).to(device)
                    
                    # 质量评估（识别时放宽质量要求）
                    quality_result = self.face_database.evaluate_child_face_quality(img_path, face_tensor, box, prob)
                    
                    # 对于识别，我们接受质量较低的人脸，但记录质量信息
                    if quality_result.quality_score < 0.3:  # 质量太差的人脸跳过
                        logger.warning(f"人脸 {i+1} 质量过低，跳过: {img_path}, 质量分数: {quality_result.quality_score:.2f}")
                        continue
                    
                    # 多模型特征提取和融合
                    embeddings = []
                    for model_name, model in self.face_database.models.items():
                        with torch.no_grad():
                            embedding = model(face_tensor)
                            embedding = F.normalize(embedding, p=2, dim=1)
                            embeddings.append(embedding)
                    
                    # 特征融合（加权平均）
                    weights = [1, 0]  # vggface2权重更高，对儿童效果更好
                    final_embedding = sum(w * emb for w, emb in zip(weights, embeddings))
                    final_embedding = F.normalize(final_embedding, p=2, dim=1)
                    
                    results.append((final_embedding.cpu(), quality_result, box, prob))
                    logger.info(f"成功提取人脸 {i+1} 特征: {img_path}, 质量分数: {quality_result.quality_score:.2f}, 置信度: {prob:.3f}")
                    
                except Exception as e:
                    logger.error(f"处理人脸 {i+1} 时出错: {img_path}, 错误: {e}")
                    continue
            
            logger.info(f"图片 {img_path} 共检测到 {len(boxes)} 张人脸，成功提取 {len(results)} 张人脸特征")
            return results
            
        except Exception as e:
            logger.error(f"提取所有人脸特征时出错: {img_path}, 错误: {e}")
            return []

    def recognize_all_faces_in_image(self, img_path: str, use_adaptive_threshold: bool = True) -> List[Tuple[int, float, Dict]]:
        """
        识别图片中的所有脸
        
        Args:
            img_path: 待识别图片路径
            use_adaptive_threshold: 是否使用自适应阈值
            
        Returns:
            List[Tuple[int, confidence, details]]: 所有人脸的识别结果列表
        """
        if not self.face_database.face_embeddings:
            return [(-1, 0.0, {"error": "数据库为空"})]
        
        # 更新缓存
        self._update_embedding_cache()
        
        # 输入验证
        if not os.path.exists(img_path):
            return [(-1, 0.0, {"error": f"图片文件不存在: {img_path}"})]
        
        try:
            # 提取待识别图片中所有人脸特征
            face_results = self.extract_all_faces_for_recognition(img_path)
            
            if not face_results:
                return [(-1, 0.0, {
                    "error": "无法提取人脸特征",
                    "image_path": img_path
                })]
            
            # 确保缓存已填充
            if len(self._embedding_cache) == 0:
                logger.warning("缓存为空，可能数据库中没有数据")
                return [(-1, 0.0, {"error": "人脸数据库缓存为空，请检查是否已添加学生数据"})]
            
            # 使用缓存进行批量计算（性能优化）
            # 确保所有特征向量都是正确的维度 [1, 512]
            embeddings_list = []
            student_ids = []
            
            for student_id, embedding in self._embedding_cache.items():
                # 确保维度正确
                if embedding.dim() == 3:  # [1, 1, 512]
                    embedding = embedding.squeeze(0)  # 变成 [1, 512]
                elif embedding.dim() == 1:  # [512]
                    embedding = embedding.unsqueeze(0)  # 变成 [1, 512]
                
                # 验证维度
                if embedding.size(1) == 512:
                    embeddings_list.append(embedding)
                    student_ids.append(student_id)
                else:
                    logger.warning(f"跳过维度异常的特征向量: {student_id}, 维度: {embedding.size()}")
            
            if not embeddings_list:
                logger.error("没有有效的特征向量用于识别")
                return [(-1, 0.0, {"error": "数据库中没有有效的特征向量"})]
            
            embeddings_tensor = torch.cat(embeddings_list, dim=0)  # 变成 [N, 512]
            
            # 调试信息
            logger.info(f"缓存中的特征向量维度: {embeddings_tensor.size()}")
            logger.info(f"学生ID列表: {student_ids}")
            
            all_face_results = []
            
            # 对每张人脸进行识别
            for face_idx, (test_embedding, quality_result, box, prob) in enumerate(face_results):
                try:
                    # 使用余弦相似度进行人脸匹配（比欧几里得距离更适合人脸识别）
                    # 确保特征向量维度正确
                    if test_embedding.dim() == 1:
                        test_embedding = test_embedding.unsqueeze(0)
                    
                    # 检查维度匹配
                    if test_embedding.size(1) != embeddings_tensor.size(1):
                        logger.error(f"特征向量维度不匹配: 测试向量 {test_embedding.size()}, 数据库向量 {embeddings_tensor.size()}")
                        continue
                    
                    # 使用余弦相似度计算（更适于人脸识别）
                    # 确保特征向量已归一化
                    test_embedding_normalized = F.normalize(test_embedding, p=2, dim=1)
                    embeddings_normalized = F.normalize(embeddings_tensor, p=2, dim=1)
                    
                    # 计算余弦相似度
                    similarities = F.cosine_similarity(test_embedding_normalized, embeddings_normalized, dim=1)
                    
                    # 确保相似度张量维度正确
                    if similarities.size(0) != len(student_ids):
                        logger.error(f"相似度张量维度错误: {similarities.size()}, 期望: {len(student_ids)}")
                        continue
                    
                    # 找到最大相似度（余弦相似度越大越好）
                    max_similarity_idx = torch.argmax(similarities)
                    max_similarity = similarities[max_similarity_idx].item()
                    best_match_student_id = student_ids[max_similarity_idx]
                    
                    # 构建所有相似度字典
                    all_similarities = {student_ids[i]: similarities[i].item() for i in range(len(student_ids))}
                    
                    # 确定使用的阈值
                    if use_adaptive_threshold and self.global_threshold is None:
                        threshold = self.face_database.get_adaptive_threshold(best_match_student_id)
                    else:
                        threshold = self.global_threshold or CHILD_CONFIG['recognition']['default_threshold']
                    
                    # 判断是否识别成功（基于余弦相似度）
                    if max_similarity > threshold:
                        recognized_student_id = best_match_student_id
                        confidence = max_similarity  # 直接使用相似度作为置信度
                        # 获取学生姓名用于显示
                        recognized_name = self.face_database.child_profiles[recognized_student_id].name if recognized_student_id in self.face_database.child_profiles else "未知"
                    else:
                        recognized_student_id = -1
                        recognized_name = "未知"
                        confidence = 0.0
                    
                    # 构建详细结果
                    details = {
                        "student_id": recognized_student_id,
                        "name": recognized_name,
                        "similarity": max_similarity,
                        "threshold": threshold,
                        "quality_score": quality_result.quality_score,
                        "all_similarities": all_similarities,
                        "face_size": quality_result.face_size,
                        "recognition_method": "adaptive" if use_adaptive_threshold else "global",
                        "face_index": face_idx + 1,  # 人脸索引（从1开始）
                        "total_faces_detected": len(face_results),
                        "face_confidence": prob,
                        "face_box": box.tolist() if box is not None else None,
                    }
                    
                    if recognized_student_id != -1:
                        # 添加儿童档案信息
                        if recognized_student_id in self.face_database.child_profiles:
                            profile = self.face_database.child_profiles[recognized_student_id]
                            details["child_info"] = {
                                "student_id": profile.student_id,
                                "name": profile.name,
                                "age": profile.age,
                            }
                    
                    all_face_results.append((recognized_student_id, confidence, details))
                    
                    logger.info(f"人脸 {face_idx+1} 识别结果: {recognized_name}({recognized_student_id}) (相似度: {max_similarity:.3f}, 置信度: {confidence:.3f}, 阈值: {threshold:.3f})")
                    
                except Exception as e:
                    logger.error(f"处理人脸 {face_idx+1} 时出错: {e}")
                    all_face_results.append((-1, 0.0, {
                        "error": f"处理人脸 {face_idx+1} 时出错: {str(e)}",
                        "face_index": face_idx + 1,
                        "total_faces_detected": len(face_results)
                    }))
            
            logger.info(f"图片 {img_path} 共识别 {len(all_face_results)} 张人脸")
            return all_face_results
                
        except Exception as e:
            logger.error(f"识别过程中发生错误: {img_path}, 错误: {e}")
            return [(-1, 0.0, {
                "error": f"识别过程异常: {str(e)}",
                "image_path": img_path
            })]
        finally:
            # 内存清理
            if 'test_embedding' in locals():
                del test_embedding
            if 'embeddings_tensor' in locals():
                del embeddings_tensor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def recognize_all_faces_batch(self, img_paths: List[str], 
                                 max_workers: int = 4) -> List[List[Tuple[int, float, Dict]]]:
        """
        批量识别所有图片中的所有脸（并发处理）
        
        Args:
            img_paths: 待识别图片路径列表
            max_workers: 最大工作线程数
            
        Returns:
            List[List[Tuple[int, confidence, details]]]: 每张图片的所有人脸识别结果列表
        """
        if not img_paths:
            return []
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.recognize_all_faces_in_image, path): path 
                for path in img_paths
            }
            
            for future in future_to_path:
                try:
                    result = future.result(timeout=30)  # 30秒超时
                    results.append(result)
                except Exception as e:
                    path = future_to_path[future]
                    logger.error(f"批量识别失败: {path}, 错误: {e}")
                    results.append([(-1, 0.0, {"error": str(e)})])
        
        return results
    
    def process_batch_with_individual_params(self, img_paths: List[str], 
                                           class_ids: List[int], 
                                           activity_details: List[str],
                                           is_publics: List[bool],
                                           uploader_ids: List[int],
                                           max_workers: int = 4) -> Dict:
        """
        批量处理图片并存储到OSS和数据库（支持每张图片独立参数）
        
        Args:
            img_paths: 待识别图片路径列表
            class_ids: 每张图片对应的班级ID列表
            activity_details: 每张图片对应的活动详情列表
            is_publics: 每张图片对应的公开状态列表
            uploader_ids: 每张图片对应的上传者ID列表
            max_workers: 最大工作线程数
            
        Returns:
            Dict: 处理结果
        """
        if not img_paths:
            return {
                'success': False,
                'error': '没有提供图片路径'
            }
        
        # 验证参数长度
        if len(img_paths) != len(class_ids) or len(img_paths) != len(activity_details) or \
           len(img_paths) != len(is_publics) or len(img_paths) != len(uploader_ids):
            return {
                'success': False,
                'error': '参数数组长度不匹配'
            }
        
        try:
            # 1. 批量识别所有脸
            start_time = datetime.now()
            all_faces_results = self.recognize_all_faces_batch(img_paths, max_workers)
            end_time = datetime.now()
            total_processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            # 2. 处理结果并准备存储
            processed_results = []
            recognition_records = []
            recognized_count = 0
            session_id = f"batch_{int(datetime.now().timestamp())}"
            
            # 处理每张图片的所有人脸结果
            for i, face_results in enumerate(all_faces_results):
                img_path = img_paths[i]
                class_id = class_ids[i]
                activity_detail = activity_details[i]
                is_public = is_publics[i]
                uploader_id = uploader_ids[i]
                
                # 处理这张图片中的所有人脸
                for face_idx, (student_id, confidence, details) in enumerate(face_results):
                    # logger.info(f"debug检查 在process_batch_with_individual_params中 处理每张图片的所有人脸结果: {face_results}")
                    # 确定识别出的学生ID
                    recognized_child_id = None
                    if student_id != -1: # 识别出儿童
                        recognized_count += 1
                        recognized_child_id = student_id
                    
                    # 准备识别记录
                    record_data = {
                        'test_image_path': img_path,
                        'recognized_child_id': recognized_child_id,
                        'confidence': confidence,
                        'similarity': details.get('similarity', 0),
                        'threshold_used': details.get('threshold', 0),
                        'recognition_method': details.get('recognition_method', 'adaptive'),
                        'face_quality_score': details.get('quality_score', 0),
                        'processing_time_ms': total_processing_time // len(all_faces_results),
                        'session_id': session_id,
                        'face_index': details.get('face_index', face_idx + 1),
                        'class_id': class_id,
                        'activity_detail': activity_detail,
                        'is_public': is_public,
                        'uploader_id': uploader_id
                    }
                    recognition_records.append(record_data)
                    
                    processed_results.append({
                        'image_index': i,
                        'face_index': details.get('face_index', face_idx + 1),
                        'recognized_name': details.get('name', '未知'),
                        'student_id': student_id,
                        'confidence': confidence,
                        'is_known_child': student_id != -1,
                        'test_image_path': img_path,
                        'class_id': class_id,
                        'activity_detail': activity_detail,
                        'is_public': is_public,
                        'uploader_id': uploader_id,
                        'details': details
                    })
            
            # 3. 上传识别成功的图片到OSS并保存记录到数据库
            saved_records = []
            upload_results = []
            
            if hasattr(self.face_database, 'db_integration') and self.face_database.db_integration:
                try:
                    for record in recognition_records:
                        # 上传测试图片到OSS并保存识别记录
                        upload_result = self.face_database.db_integration.upload_recognized_photo(
                            test_image_path=record['test_image_path'],
                            recognized_child_id=record['recognized_child_id'],
                            recognition_confidence=record['confidence'],
                            session_id=session_id,
                            class_id=record['class_id'],
                            activity_detail=record['activity_detail'],
                            is_public=record['is_public'],
                            uploader_id=record['uploader_id']
                        )
                        upload_results.append(upload_result)
                        saved_records.append(upload_result['photo_id'])
                        
                        # 更新 processed_results 中的 test_image_url 为真实的OSS URL
                        for result in processed_results:
                            if (result['test_image_path'] == record['test_image_path'] and 
                                result['face_index'] == record['face_index']):
                                result['test_image_url'] = upload_result.get('oss_url', record['test_image_path'])
                                break
                    
                    logger.info(f"✅ 测试图片上传和识别记录保存完成: {len(saved_records)} 条记录")
                except Exception as db_error:
                    logger.warning(f"上传测试图片和保存识别记录失败: {db_error}")
                    # 继续处理，不中断整个流程
            
            # 计算总人脸数
            total_faces = sum(len(face_results) for face_results in all_faces_results)
            unknown_faces = total_faces - recognized_count
            
            return {
                'success': True,
                'data': {
                    'session_id': session_id,
                    'total_images': len(img_paths),
                    'total_faces_detected': total_faces,
                    'recognized_children': recognized_count,
                    'unknown_faces': unknown_faces,
                    'total_processing_time_ms': total_processing_time,
                    'avg_processing_time_ms': total_processing_time // len(all_faces_results),
                    'results': processed_results,
                    'database_records_saved': len(saved_records),
                    'oss_uploads_completed': len(upload_results),
                    'upload_results': upload_results,
                    'processing_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_recognition_statistics(self) -> Dict:
        """获取识别统计信息"""
        return {
            "database_size": len(self.face_database.face_embeddings),
            "cache_size": len(self._embedding_cache),
            "last_cache_update": self._last_cache_update.isoformat(),
            "memory_usage": f"{torch.cuda.memory_allocated() / 1024**2:.1f}MB" if torch.cuda.is_available() else "N/A"
        }

    def check_database_status(self) -> Dict:
        """
        检查数据库和缓存状态，用于问题诊断
        
        Returns:
            包含数据库状态信息的字典
        """
        status = {
            'database_loaded': hasattr(self.face_database, 'face_embeddings'),
            'database_size': len(self.face_database.face_embeddings) if hasattr(self.face_database, 'face_embeddings') else 0,
            'cache_size': len(self._embedding_cache),
            'students_in_database': list(self.face_database.face_embeddings.keys()) if hasattr(self.face_database, 'face_embeddings') else [],
            'students_in_cache': list(self._embedding_cache.keys()),
            'last_cache_update': self._last_cache_update.isoformat() if self._last_cache_update else None
        }
        
        print("📊 数据库状态检查:")
        print(f"   - 数据库是否加载: {status['database_loaded']}")
        print(f"   - 数据库中学生数量: {status['database_size']}")
        print(f"   - 缓存中学生数量: {status['cache_size']}")
        print(f"   - 数据库中的学生: {status['students_in_database']}")
        print(f"   - 缓存中的学生: {status['students_in_cache']}")
        print(f"   - 最后缓存更新时间: {status['last_cache_update']}")
        
        return status

