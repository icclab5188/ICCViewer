"""
ICC Viewer 配置文件
用于管理软件的各种设置和参数
"""

import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

# 模型路径配置
GROUNDING_DINO_PATH = PROJECT_ROOT / "GroundingDINO"
YOLOV5_PATH = PROJECT_ROOT / "yolov5"

# 模型权重文件路径
GROUNDING_DINO_WEIGHTS = GROUNDING_DINO_PATH / "groundingdino_swint_ogc.pth"
YOLOV5_WEIGHTS = YOLOV5_PATH / "yolov5s.pt"

# 模型配置文件路径
GROUNDING_DINO_CONFIG = GROUNDING_DINO_PATH / "groundingdino/config/GroundingDINO_SwinT_OGC.py"

# 视频配置
VIDEO_FPS = 30
VIDEO_FRAME_DELAY = 1.0 / VIDEO_FPS
CAMERA_INDEX = 0  # USB相机索引

# 检测参数默认值
DEFAULT_GROUNDING_DINO_BOX_THRESHOLD = 0.3
DEFAULT_GROUNDING_DINO_TEXT_THRESHOLD = 0.25
DEFAULT_YOLOV5_CONF_THRESHOLD = 0.25
DEFAULT_YOLOV5_IOU_THRESHOLD = 0.45

# UI配置
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 768
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600
CONTROLS_MAX_WIDTH = 350
CONTROLS_MIN_WIDTH = 300
VIDEO_MIN_WIDTH = 640
VIDEO_MIN_HEIGHT = 480

# 视频保存配置
SAVE_FPS = 20.0
SAVE_FOURCC = 'mp4v' if os.name != 'nt' else 'XVID'
SEGMENT_DURATION_MINUTES = 20

# 日志配置
LOG_LEVEL = "DEBUG"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILENAME = "app_error.log"

# Supported video formats
SUPPORTED_VIDEO_FORMATS = "Video Files (*.mp4 *.avi *.mov *.mkv)"

# YOLOv5 weight options
YOLOV5_WEIGHT_OPTIONS = [
    "yolov5n.pt",
    "yolov5s.pt", 
    "yolov5m.pt",
    "yolov5l.pt",
    "yolov5x.pt"
]

# Detection model options
DETECTOR_OPTIONS = [
    "Grounding DINO",
    "YOLOv5"
]

# Color configuration
UI_COLORS = {
    'background': '#f0f0f0',
    'border': '#cccccc',
    'text': '#333333',
    'button': '#4a90e2',
    'button_hover': '#357abd',
    'slider': '#666666'
}

# Performance configuration
DOWNSAMPLE_FACTOR = 0.5
MAX_DETECTIONS = 1000
BATCH_SIZE = 1

# Thread configuration
THREAD_SLEEP_TIME = 0.03
MODEL_SWITCH_SLEEP_TIME = 0.1
PAUSE_SLEEP_TIME = 0.1

# Error handling configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0

# Development mode configuration
DEBUG_MODE = True
SHOW_FPS = True
SAVE_DEBUG_IMAGES = False

# Keyboard shortcuts configuration
KEYBOARD_SHORTCUTS = {
    'fullscreen': 'F11',
    'exit_fullscreen': 'Escape',
    'pause_camera': 'Space',
    'next_model': 'Tab',
    'save_video': 'Ctrl+S'
}

# Internationalization configuration
LANGUAGE = "en_US"
TRANSLATIONS = {
    "zh_CN": {
        "window_title": "ICC Viewer - 目标检测软件",
        "select_model": "选择检测模型:",
        "text_prompt": "文本提示:",
        "text_prompt_placeholder": "输入文本提示 (例如: 'crocodile')",
        "box_threshold": "框阈值",
        "text_threshold": "文本阈值",
        "yolov5_weights": "YOLOv5 权重:",
        "confidence_threshold": "置信度阈值",
        "iou_threshold": "IoU 阈值",
        "downsample": "下采样输入 (0.5x)",
        "play_video": "播放本地视频",
        "pause_camera": "暂停相机",
        "resume_camera": "恢复相机",
        "start_saving": "开始保存",
        "stop_saving": "停止保存",
        "loading_camera": "正在加载相机...",
        "error": "错误",
        "warning": "警告",
        "info": "信息",
        "success": "成功"
    },
    "en_US": {
        "window_title": "ICC Viewer - Object Detection Software",
        "select_model": "Select Detection Model:",
        "text_prompt": "Text Prompt:",
        "text_prompt_placeholder": "Enter text prompt (e.g., 'crocodile')",
        "box_threshold": "Box Threshold",
        "text_threshold": "Text Threshold",
        "yolov5_weights": "YOLOv5 Weights:",
        "confidence_threshold": "Confidence Threshold",
        "iou_threshold": "IoU Threshold",
        "downsample": "Downsample Input (0.5x)",
        "play_video": "Play Local Video",
        "pause_camera": "Pause Camera",
        "resume_camera": "Resume Camera",
        "start_saving": "Start Saving",
        "stop_saving": "Stop Saving",
        "loading_camera": "Loading camera...",
        "error": "Error",
        "warning": "Warning",
        "info": "Information",
        "success": "Success"
    }
}

def get_text(key):
    """获取当前语言的文本"""
    return TRANSLATIONS.get(LANGUAGE, TRANSLATIONS["zh_CN"]).get(key, key)

def get_model_path(model_type):
    """获取模型路径"""
    if model_type == "grounding_dino":
        return GROUNDING_DINO_PATH
    elif model_type == "yolov5":
        return YOLOV5_PATH
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_weights_path(model_type, weights_name=None):
    """获取权重文件路径"""
    if model_type == "grounding_dino":
        return GROUNDING_DINO_WEIGHTS
    elif model_type == "yolov5":
        if weights_name:
            return YOLOV5_PATH / weights_name
        return YOLOV5_WEIGHTS
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def check_model_files():
    """检查模型文件是否存在"""
    missing_files = []
    
    if not GROUNDING_DINO_WEIGHTS.exists():
        missing_files.append(f"Grounding DINO weights: {GROUNDING_DINO_WEIGHTS}")
    
    if not YOLOV5_WEIGHTS.exists():
        missing_files.append(f"YOLOv5 weights: {YOLOV5_WEIGHTS}")
    
    return missing_files

def create_directories():
    """创建必要的目录"""
    directories = [
        BASE_DIR,
        PROJECT_ROOT,
        GROUNDING_DINO_PATH,
        YOLOV5_PATH
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True) 