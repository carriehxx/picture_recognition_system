#!/usr/bin/env python3
"""
儿童人脸识别API服务
提供RESTful API接口供Express应用调用
专门优化用于儿童人脸识别场景
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
import tempfile
import threading
import time
from datetime import datetime, timedelta
import logging
import json

from database_integration import db_integration

from child_face_system import (
    ChildFaceDatabase, 
    ChildFaceRecognizer, 
    ChildProfile,
    CHILD_CONFIG
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


CORS(app, 
     origins=["*"],  
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     expose_headers=["Content-Range", "X-Content-Range"],
     supports_credentials=True,
     max_age=86400  
)

# 全局变量存储模型实例（避免重复加载）
child_db = None
child_recognizer = None
model_lock = threading.Lock()

def initialize_models():
    """初始化儿童人脸识别模型（线程安全）"""
    global child_db, child_recognizer
    
    with model_lock:
        if child_db is None:
            logger.info("🔄 正在初始化儿童人脸识别模型...")
            
            # 初始化儿童人脸数据库
            child_db = ChildFaceDatabase()
            
            # 初始化儿童人脸识别器
            child_recognizer = ChildFaceRecognizer(child_db)
            
            logger.info("✅ 儿童人脸识别模型初始化完成")

def base64_to_image(base64_string):
    """将base64字符串转换为PIL图像"""
    try:
        # 移除base64前缀
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # 解码
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # 记录图片信息用于调试
        logger.info(f"成功解码图片: 格式={image.format}, 模式={image.mode}, 尺寸={image.size}")
        
        return image
    except Exception as e:
        logger.error(f"Base64解码失败: {e}")
        return None

def save_temp_image(image, format='JPEG'):
    """保存临时图片文件"""
    try:
        original_mode = image.mode
        logger.info(f"保存临时图片: 原始模式={original_mode}, 目标格式={format}")
        
        # 如果是RGBA格式且要保存为JPEG，需要转换为RGB
        if image.mode == 'RGBA' and format.upper() == 'JPEG':
            logger.info("检测到RGBA格式，转换为RGB格式以保存为JPEG")
            # 创建白色背景
            background = Image.new('RGB', image.size, (255, 255, 255))
            # 将RGBA图片粘贴到白色背景上
            background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
            image = background
        elif image.mode == 'RGBA' and format.upper() == 'PNG':
            # PNG支持RGBA，直接保存
            logger.info("RGBA格式，直接保存为PNG")
            pass
        elif image.mode != 'RGB' and format.upper() == 'JPEG':
            # 其他格式转换为RGB
            logger.info(f"将{image.mode}格式转换为RGB格式")
            image = image.convert('RGB')
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format.lower()}')
        image.save(temp_file.name, format=format)
        logger.info(f"临时文件保存成功: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"保存临时文件失败: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': child_db is not None,
        'system_type': 'child_face_recognition'
    })

@app.route('/config', methods=['GET'])
def get_config():
    """获取系统配置信息"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'child_config': CHILD_CONFIG,
                'description': '儿童人脸识别系统配置'
            }
        })
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/database/info', methods=['GET'])
def get_database_info():
    """获取儿童数据库信息"""
    try:
        initialize_models()
        stats = child_db.get_children_statistics()
        
        return jsonify({
            'success': True,
            'data': {
                'statistics': stats,
                # 'update_requirements': child_db.check_update_requirements(),
                'database_path': child_db.database_path,
                'storage_path': child_db.storage_path
            }
        })
    except Exception as e:
        logger.error(f"获取数据库信息失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/database/children', methods=['GET'])
def get_all_children():
    """获取所有儿童列表"""
    try:
        initialize_models()
        
        children_data = []
        for student_id, face_data in child_db.face_embeddings.items():
            child_info = {
                'student_id': student_id,
                'created_at': face_data.get('created_at'),
                'last_updated': face_data.get('last_updated'),
                'photos_count': face_data.get('photos_count', 1),
                'avg_quality': face_data.get('avg_quality', 0),
                'update_due_date': face_data.get('update_due_date')
            }
            
            # 添加档案信息
            if student_id in child_db.child_profiles:
                profile = child_db.child_profiles[student_id]
                child_info.update({
                    'age': profile.age,
                    'class_id': profile.class_id,
                    'name': profile.name
                })
            
            children_data.append(child_info)
        
        return jsonify({
            'success': True,
            'data': {
                'total_children': len(children_data),
                'children': children_data
            }
        })
    except Exception as e:
        logger.error(f"获取儿童列表失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/database/add_child', methods=['POST'])
def add_child():
    """添加新儿童
    传入参数包含：
    name: 儿童姓名
    student_id: 儿童学号
    images: 儿童照片列表
    update_if_exists: 是否更新已存在的儿童
    profile: 儿童档案信息
    
    """
    try:
        initialize_models()
        
        data = request.json
        child_name = data.get('name')
        images_base64 = data.get('images', [])  # 支持多张照片
        update_if_exists = data.get('update_if_exists', True)
        
        # 儿童档案信息
        profile_data = data.get('profile', {})
        student_id = profile_data.get('student_id')
        child_profile = ChildProfile(
            student_id=profile_data.get('student_id'),
            name=child_name,
            age=profile_data.get('age', 0),
            class_id=profile_data.get('class_id', 0)
        )
        
        if not child_name or not student_id or not images_base64:
            return jsonify({
                'success': False,
                'error': '缺少必要参数: name, student_id, images'
            }), 400

        # 转换所有base64图片
        temp_paths = []
        upload_results = []
        
        try:
            for i, image_base64 in enumerate(images_base64):
                image = base64_to_image(image_base64)
                if image is None:
                    return jsonify({
                        'success': False,
                        'error': f'第{i+1}张图片格式错误'
                    }), 400

                temp_path = save_temp_image(image)
                if temp_path is None:
                    return jsonify({
                        'success': False,
                        'error': f'保存第{i+1}张图片失败'
                    }), 500

                temp_paths.append(temp_path)
            
            # 添加儿童到数据库（包含本地存储和MySQL集成）
            success = child_db.add_child(temp_paths, child_profile, update_if_exists)
            
            if not success:
                return jsonify({
                    'success': False,
                    'error': f'添加儿童失败: {child_name}'
                }), 400
            
            # 获取添加后的儿童信息
            child_data = child_db.face_embeddings.get(student_id, {})
            
            return jsonify({
                'success': True,
                'message': f'成功添加儿童: {child_name}({student_id})',
                'data': {
                    'student_id': student_id,
                    'child_name': child_name,
                    'photos_processed': len(temp_paths),
                    'avg_quality': child_data.get('avg_quality', 0),
                    'threshold': child_db.get_adaptive_threshold(student_id),
                    'database_integration': bool(hasattr(child_db, 'db_integration') and child_db.db_integration)
                }
            })
                
        finally:
            # 清理临时文件
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
    except Exception as e:
        logger.error(f"添加儿童失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch_recognize', methods=['POST'])
def batch_recognize():
    """批量儿童人脸识别接口"""
    try:
        # 初始化模型
        initialize_models()
        
        # 获取请求参数
        data = request.json
        images = data.get('images', [])
        class_ids = data.get('class_id', [])
        activity_details = data.get('activity_detail', [])
        is_publics = data.get('is_public', [])
        uploader_ids = data.get('uploader_id', [])
        max_workers = data.get('max_workers', 4)
        
        # 参数校验
        if not images:
            return jsonify({
                'success': False,
                'error': '缺少必要参数: images'
            }), 400
        
        # 检查数组长度是否匹配
        if len(class_ids) != len(images):
            return jsonify({
                'success': False,
                'error': f'class_ids 长度 ({len(class_ids)}) 与 images 长度 ({len(images)}) 不匹配'
            }), 400
        
        if len(activity_details) != len(images):
            return jsonify({
                'success': False,
                'error': f'activity_details 长度 ({len(activity_details)}) 与 images 长度 ({len(images)}) 不匹配'
            }), 400
        
        if len(is_publics) != len(images):
            return jsonify({
                'success': False,
                'error': f'is_publics 长度 ({len(is_publics)}) 与 images 长度 ({len(images)}) 不匹配'
            }), 400
        
        if len(uploader_ids) != len(images):
            return jsonify({
                'success': False,
                'error': f'uploader_ids 长度 ({len(uploader_ids)}) 与 images 长度 ({len(images)}) 不匹配'
            }), 400

        
        temp_paths = []
        try:
            # 处理所有图片
            for i, image_base64 in enumerate(images):
                image = base64_to_image(image_base64)
                if image is None:
                    return jsonify({
                        'success': False,
                        'error': f'第{i+1}张图片格式错误'
                    }), 400
                
                temp_path = save_temp_image(image)
                if temp_path is None:
                    return jsonify({
                        'success': False,
                        'error': f'保存第{i+1}张图片失败'
                    }), 500
                
                temp_paths.append(temp_path)
            
            # 使用新的批量处理方法，支持每张图片独立参数
            result = child_recognizer.process_batch_with_individual_params(
                img_paths=temp_paths,
                class_ids=class_ids,
                activity_details=activity_details,
                is_publics=is_publics,
                uploader_ids=uploader_ids,
                max_workers=max_workers
            )
            # logger.info(f"debug检查 在face_recognition_api.py中 批量处理结果: {result}")
            
            # 检查结果格式
            if not isinstance(result, dict):
                logger.error(f"批量处理返回了非字典类型结果: {type(result)}")
                return jsonify({
                    'success': False,
                    'error': '批量处理返回了无效的结果格式'
                }), 500
            
            logger.info(f"checkpoint 1")
            
            if not result.get('success', False):
                error_msg = result.get('error', '批量处理失败')
                logger.error(f"批量处理失败: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': error_msg
                }), 500
            
            logger.info(f"checkpoint 2")
            
            return jsonify({
                'success': True,
                'data': result.get('data', {})
            })
            logger.info(f"checkpoint 3")
            
        finally:
            # 清理所有临时文件
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"批量识别失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/child/<student_id>/profile', methods=['GET'])
def get_child_profile(student_id):
    """获取儿童档案信息"""
    try:
        initialize_models()
        student_id = int(student_id)
        if student_id not in child_db.child_profiles:
            return jsonify({
                'success': False,
                'error': f'儿童档案不存在: {student_id}'
            }), 404
        
        profile = child_db.child_profiles[student_id]
        face_data = child_db.face_embeddings.get(student_id, {})
        
        return jsonify({
            'success': True,
            'data': {
                'profile': {
                    'student_id': profile.student_id,
                    'name': profile.name,
                    'age': profile.age,
                    'class_id': profile.class_id
                }
            }
        })
        
    except Exception as e:
        logger.error(f"获取儿童档案失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/maintenance/update_requirements', methods=['GET'])
def get_update_requirements():
    """获取需要更新照片的儿童列表"""
    try:
        initialize_models()
        
        need_update = child_db.check_update_requirements()
        
        update_details = []
        for student_id in need_update:
            face_data = child_db.face_embeddings.get(student_id)
            profile = child_db.child_profiles.get(student_id)
            
            update_details.append({
                'child_name': student_id,
                'last_updated': face_data.get('last_updated'),
                'update_due_date': face_data.get('update_due_date'),
                'age': profile.age if profile else None,
                'photos_count': face_data.get('photos_count', 0)
            })
        
        return jsonify({
            'success': True,
            'data': {
                'total_requiring_update': len(need_update),
                'children_needing_update': update_details
            }
        })
        
    except Exception as e:
        logger.error(f"获取更新需求失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/system/statistics', methods=['GET'])
def get_system_statistics():
    """获取系统统计信息"""
    try:
        initialize_models()
        
        db_stats = child_db.get_children_statistics()
        recognition_stats = child_recognizer.get_recognition_statistics()
        
        return jsonify({
            'success': True,
            'data': {
                'database_statistics': db_stats,
                'recognition_statistics': recognition_stats,
                'system_info': {
                    'version': '1.0_child_face_recognition_system',
                    'config': CHILD_CONFIG,
                    'timestamp': datetime.now().isoformat()
                }
            }
        })
        
    except Exception as e:
        logger.error(f"获取系统统计失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/database/delete_child', methods=['POST'])
def delete_child():
    """删除儿童"""
    data = request.json
    student_id = data.get('student_id')
    if not student_id:
        return jsonify({
            'success': False,
            'error': '缺少必要参数: student_id'
        }), 400
    # 删除儿童的图片和数据库记录
    child_db.delete_child(student_id)
    return jsonify({
        'success': True,
        'message': f'成功删除儿童: {student_id}'
    })

@app.route('/database/delete_image', methods=['POST'])
def delete_image():
    """根据图片URL删除OSS上的图片还有数据库中的记录
    参数格式：
    {
        "object_keys": ["key1", "key2"],
        "ids": [1, 2]
    }
    """
    try:
        data = request.json
        object_keys = data.get('object_keys')
        ids = data.get('ids')
        
        # 参数验证
        if not object_keys:
            return jsonify({
                'success': False,
                'error': '缺少必要参数: object_keys'
            }), 400
        
        if not ids:
            return jsonify({
                'success': False,
                'error': '缺少必要参数: ids'
            }), 400
        
        if len(object_keys) != len(ids):
            return jsonify({
                'success': False,
                'error': f'参数长度不匹配: object_keys({len(object_keys)}) != ids({len(ids)})'
            }), 400
        
        # 删除图片
        result = db_integration.delete_images(object_keys, ids)
        
        # 根据删除结果返回响应
        if result['success']:
            return jsonify({
                'success': True,
                'message': result['message'],
                'data': {
                    'oss_deleted': result['oss_deleted'],
                    'oss_failed': result['oss_failed'],
                    'db_photos_deleted': result['db_photos_deleted'],
                    'db_relations_deleted': result['db_relations_deleted'],
                    'total_processed': result['total_processed']
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'data': {
                    'oss_deleted': result.get('oss_deleted', 0),
                    'oss_failed': result.get('oss_failed', 0)
                }
            }), 500
            
    except Exception as e:
        logger.error(f"删除图片API失败: {e}")
        return jsonify({
            'success': False,
            'error': f'删除图片失败: {str(e)}'
        }), 500

# @app.route('/database/upload_image',methods=['POST'])
# def upload_image():
#     """上传一些不需要人脸识别的图片"""
#     try:
#         data = request.json
        
            
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'API端点不存在'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': '服务器内部错误'
    }), 500

if __name__ == '__main__':
    # 启动时初始化模型
    initialize_models()
    
    print("🎯 儿童人脸识别API服务启动")
    print("📋 可用的API端点:")
    print("   GET  /health - 健康检查")
    print("   GET  /config - 获取系统配置")
    print("   GET  /database/info - 获取数据库信息")
    print("   GET  /database/children - 获取所有儿童列表")
    print("   POST /database/add_child - 添加新儿童")
    print("   POST /batch_recognize - 批量人脸识别")
    print("   GET  /child/<name>/profile - 获取儿童档案")
    print("   GET  /maintenance/update_requirements - 获取更新需求")
    print("   GET  /system/statistics - 获取系统统计")
    
    # 检查是否为开发模式
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    
    if debug_mode:
        print("🔧 开发模式已启用 - 代码修改后会自动重启")
        print("💡 要禁用开发模式，设置环境变量: FLASK_DEBUG=0")
    
    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=debug_mode,  # 根据环境变量决定
        threaded=True
    ) 