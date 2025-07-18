#!/usr/bin/env python3
"""
儿童人脸识别系统启动脚本
统一管理所有组件的启动和配置
"""

import sys
import os
import time
import logging
from logging.handlers import RotatingFileHandler
from threading import Thread
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """设置日志"""
    # 确保logs目录存在
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # 生成日志文件名（包含日期）
    from datetime import datetime
    log_filename = f"face_recognition_system_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = logs_dir / log_filename
    
    # 使用RotatingFileHandler进行日志轮转
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,  # 保留5个备份文件
        encoding='utf-8'
    )
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    print(f"📝 日志文件: {log_path}")
    print(f"📁 日志目录: {logs_dir.absolute()}")

def load_environment():
    """加载环境变量"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ 环境变量已加载")
    except ImportError:
        print("⚠️  python-dotenv 未安装，将使用系统环境变量")
    except Exception as e:
        print(f"⚠️  加载环境变量失败: {e}")

def check_system_requirements():
    """检查系统要求"""
    print("\n🔍 检查系统要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ Python版本过低，需要Python 3.7+")
        return False
    
    print(f"✅ Python版本: {sys.version}")
    
    # 检查依赖包
    required_packages = [
        'torch', 'facenet_pytorch', 'mysql.connector', 
        'flask', 'flask_cors', 'PIL', 'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    if missing:
        print(f"\n缺少依赖包: {', '.join(missing)}")
        print("请运行: python setup_environment.py")
        return False
    
    return True

def test_connections():
    """测试各项连接"""
    print("\n🔗 测试系统连接...")
    
    # 测试数据库连接
    try:
        from database_integration import get_database_integration
        db = get_database_integration()
        print("✅ MySQL数据库连接正常")
    except Exception as e:
        print(f"❌ MySQL连接失败: {e}")
        return False
    
    # 测试OSS连接
    try:
        from storage_managers import get_storage_manager
        storage = get_storage_manager()
        print("✅ OSS对象存储连接正常")
    except Exception as e:
        print(f"⚠️  OSS连接失败，将使用本地存储: {e}")
    
    # 测试人脸识别模型
    try:
        from child_face_system import ChildFaceDatabase
        face_db = ChildFaceDatabase()
        print("✅ 人脸识别模型加载正常")
    except Exception as e:
        print(f"❌ 人脸识别模型加载失败: {e}")
        return False
    
    return True

def start_api_server():
    """启动API服务器"""
    print("\n🚀 启动API服务器...")
    
    try:
        from face_recognition_api import app, initialize_models
        
        # 预加载模型
        initialize_models()
        
        # 获取配置
        host = os.getenv('API_HOST', '0.0.0.0')
        port = int(os.getenv('API_PORT', '5000'))
        debug = os.getenv('API_DEBUG', 'False').lower() == 'true'
        
        print(f"📡 API服务器启动在: http://{host}:{port}")
        print("📋 可用端点:")
        print("   GET  /health - 健康检查")
        print("   POST /recognize - 人脸识别")
        print("   POST /database/add_child - 添加儿童")
        print("   GET  /database/children - 获取儿童列表")
        
        # 启动Flask应用
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
        
    except Exception as e:
        print(f"❌ API服务器启动失败: {e}")
        sys.exit(1)

def print_startup_info():
    """打印启动信息"""
    print("=" * 60)
    print("🎯 儿童人脸识别系统")
    print("=" * 60)
    print("版本: 2.0 (儿童优化版)")
    print("功能: MySQL + OSS + 儿童人脸识别")
    print("=" * 60)

def print_usage_examples():
    """打印使用示例"""
    port = os.getenv('API_PORT', '5000')
    
    print(f"\n📖 API使用示例:")
    print(f"curl http://localhost:{port}/health")
    print(f"curl -X POST http://localhost:{port}/recognize -H 'Content-Type: application/json' -d '{{\"image\": \"base64数据\"}}'")
    print(f"curl -X POST http://localhost:{port}/database/add_child -H 'Content-Type: application/json' -d '{{\"name\": \"张三\", \"images\": [\"base64数据\"], \"profile\": {{\"age\": 5}}}}'")

def cleanup_old_data():
    """清理旧数据"""
    try:
        from database_integration import get_database_integration
        db = get_database_integration()
        
        # 清理90天前的识别记录
        deleted_count = db.cleanup_old_records(days=90)
        if deleted_count > 0:
            print(f"🧹 清理了 {deleted_count} 条旧识别记录")
    except Exception as e:
        print(f"⚠️  数据清理失败: {e}")

def main():
    """主启动流程"""
    print_startup_info()
    
    # 设置日志
    setup_logging()
    
    # 加载环境变量
    load_environment()
    
    # 检查系统要求
    # if not check_system_requirements():
    #     print("\n❌ 系统检查失败，请先运行 python setup_environment.py")
    #     sys.exit(1)
    
    # 测试连接
    if not test_connections():
        print("\n❌ 连接测试失败，请检查配置")
        sys.exit(1)
    
    # 清理旧数据
    cleanup_old_data()
    
    # 打印使用示例
    # print_usage_examples()
    
    # print("\n✅ 系统检查完成，正在启动API服务...")
    
    # 启动API服务器
    start_api_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 系统已停止")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 系统启动失败: {e}")
        sys.exit(1) 