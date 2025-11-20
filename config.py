# config.py
import os

# 模型保存配置
MODEL_DIR = "models"
MODEL_NAME_PATTERN = "bombing_plane_v{epoch:04d}.pth"

def get_model_path(epoch):
    """
    获取指定epoch的模型文件路径
    
    Args:
        epoch: epoch编号（从1开始）
    
    Returns:
        模型文件的完整路径
    """
    return os.path.join(MODEL_DIR, MODEL_NAME_PATTERN.format(epoch=epoch))

