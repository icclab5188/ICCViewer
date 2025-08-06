#!/bin/bash

# ICC Viewer 启动脚本

echo "Starting ICC Viewer - Object Detection Software"
echo "==============================================="

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found, please install Python 3.8+ first"
    exit 1
fi

# Check if in correct directory
if [ ! -f "main.py" ]; then
    echo "Error: Please run this script in the ICCViewer directory"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import PySide6, cv2, torch, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Missing required dependencies"
    echo "Please run: pip install -r requirements.txt"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA
echo "Checking CUDA environment..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check model weights
echo "Checking model weights..."
if [ ! -f "../GroundingDINO/groundingdino_swint_ogc.pth" ]; then
    echo "Warning: Grounding DINO weights not found"
    echo "Please download: wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    echo "and place it in the ../GroundingDINO/ directory"
fi

if [ ! -f "../yolov5/yolov5s.pt" ]; then
    echo "Warning: YOLOv5 weights not found"
    echo "YOLOv5 weights will be automatically downloaded on first run"
fi

# Start application
echo "Starting ICC Viewer..."
python3 main.py

echo "ICC Viewer has exited" 