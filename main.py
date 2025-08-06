import os
import sys
import cv2
import torch
import platform
import threading
import queue
import time
import datetime
import traceback
import logging
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox, QSlider, QLineEdit, QHBoxLayout, QCheckBox, QComboBox, QSizePolicy
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

from detector import DetectorManager

# Setup logging
logging.basicConfig(
    filename="app_error.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


class VideoThread(threading.Thread):
    def __init__(self, frame_queue, stop_event, paused_event, model_switching_event):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.paused_event = paused_event
        self.model_switching_event = model_switching_event
        self.playing_local_video_event = threading.Event()

    def run(self):
        logging.info("VideoThread started")
        try:
            while not self.stop_event.is_set():
                if self.paused_event.is_set():
                    time.sleep(0.1)
                    continue
                
                # 检查是否正在切换模型
                if self.model_switching_event.is_set():
                    time.sleep(0.1)
                    continue
                
                # 检查是否正在播放本地视频
                if self.playing_local_video_event.is_set():
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if ret:
                    # 清空队列中的旧帧，只保留最新的
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.frame_queue.put(frame)
                else:
                    logging.warning("Camera frame read failed")
                time.sleep(0.03)
        except Exception:
            logging.error("Error in VideoThread:\n" + traceback.format_exc())
        finally:
            logging.info("VideoThread exiting")

    def release(self):
        try:
            self.cap.release()
            logging.info("Camera released")
        except Exception:
            logging.error("Error releasing camera:\n" + traceback.format_exc())


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ICC Viewer - Object Detection Software")
        
        # 设置窗口大小
        self.resize(1200, 768)
        self.setMinimumSize(800, 600)

        # 创建主布局 - 使用水平布局
        main_layout = QHBoxLayout()
        
        # 左侧控件区域
        controls_layout = QVBoxLayout()
        
        # 添加一些间距
        controls_layout.addSpacing(10)
        
        # 模型选择
        controls_layout.addWidget(QLabel("Select Detection Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Grounding DINO", "CrocoDINO", "LGTrack"])
        self.model_combo.currentTextChanged.connect(self.change_detector)
        controls_layout.addWidget(self.model_combo)
        
        controls_layout.addSpacing(10)
        
        # Grounding DINO 控件
        self.gdino_controls = QWidget()
        gdino_layout = QVBoxLayout()
        
        # Grounding DINO 权重选择
        self.gdino_weights_combo = QComboBox()
        self.gdino_weights_combo.addItems([
            "groundingdino_swint_ogc.pth",
            "groundingdino_finetuned_misc.pth",
            "groundingdino_finetuned_moderate.pth"
        ])
        self.gdino_weights_combo.setCurrentText("groundingdino_swint_ogc.pth")
        self.gdino_weights_combo.currentTextChanged.connect(self.change_gdino_weights)
        
        gdino_layout.addWidget(QLabel("Grounding DINO Weights:"))
        gdino_layout.addWidget(self.gdino_weights_combo)
        
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter text prompt (e.g., 'crocodile')")
        gdino_layout.addWidget(QLabel("Text Prompt:"))
        gdino_layout.addWidget(self.prompt_input)
        
        self.box_thresh_slider = QSlider(Qt.Horizontal)
        self.box_thresh_slider.setRange(0, 100)
        self.box_thresh_slider.setValue(30)
        gdino_layout.addWidget(QLabel("Box Threshold"))
        gdino_layout.addWidget(self.box_thresh_slider)
        
        self.text_thresh_slider = QSlider(Qt.Horizontal)
        self.text_thresh_slider.setRange(0, 100)
        self.text_thresh_slider.setValue(25)
        gdino_layout.addWidget(QLabel("Text Threshold"))
        gdino_layout.addWidget(self.text_thresh_slider)
        self.gdino_controls.setLayout(gdino_layout)

        # YOLOv5 控件
        self.yolov5_controls = QWidget()
        yolov5_layout = QVBoxLayout()
        
        # YOLOv5 权重选择
        self.weights_combo = QComboBox()
        self.weights_combo.addItems([
            "lgtrack_init.pt", "lgtrack_croco.pt"
        ])
        self.weights_combo.setCurrentText("lgtrack_init.pt")
        self.weights_combo.currentTextChanged.connect(self.change_yolov5_weights)
        
        self.conf_thresh_slider = QSlider(Qt.Horizontal)
        self.conf_thresh_slider.setRange(0, 100)
        self.conf_thresh_slider.setValue(25)
        self.iou_thresh_slider = QSlider(Qt.Horizontal)
        self.iou_thresh_slider.setRange(0, 100)
        self.iou_thresh_slider.setValue(45)
        
        yolov5_layout.addWidget(QLabel("LGTrack Weights:"))
        yolov5_layout.addWidget(self.weights_combo)
        yolov5_layout.addWidget(QLabel("Confidence Threshold"))
        yolov5_layout.addWidget(self.conf_thresh_slider)
        yolov5_layout.addWidget(QLabel("IoU Threshold"))
        yolov5_layout.addWidget(self.iou_thresh_slider)
        self.yolov5_controls.setLayout(yolov5_layout)
        self.yolov5_controls.hide()  # 初始隐藏

        # CrocoDINO 控件
        self.crocodino_controls = QWidget()
        crocodino_layout = QVBoxLayout()
        
        # CrocoDINO 权重选择
        self.crocodino_weights_combo = QComboBox()
        self.crocodino_weights_combo.addItems([
            "crocodino_moderate.pth",
            "crocodino_low.pth",
            "crocodino_high.pth"
        ])
        self.crocodino_weights_combo.setCurrentText("crocodino_moderate.pth")
        self.crocodino_weights_combo.currentTextChanged.connect(self.change_crocodino_weights)
        
        self.crocodino_conf_thresh_slider = QSlider(Qt.Horizontal)
        self.crocodino_conf_thresh_slider.setRange(0, 100)
        self.crocodino_conf_thresh_slider.setValue(30)
        
        crocodino_layout.addWidget(QLabel("CrocoDINO Weights:"))
        crocodino_layout.addWidget(self.crocodino_weights_combo)
        crocodino_layout.addWidget(QLabel("Confidence Threshold"))
        crocodino_layout.addWidget(self.crocodino_conf_thresh_slider)
        self.crocodino_controls.setLayout(crocodino_layout)
        self.crocodino_controls.hide()  # 初始隐藏

        # 添加控件到左侧布局
        controls_layout.addWidget(self.gdino_controls)
        controls_layout.addWidget(self.crocodino_controls)
        controls_layout.addWidget(self.yolov5_controls)
        
        controls_layout.addSpacing(10)
        
        # 通用控件
        self.downsample_checkbox = QCheckBox("Downsample Input (0.5x)")
        self.downsample_checkbox.setChecked(False)
        controls_layout.addWidget(self.downsample_checkbox)
        
        self.resize_large_checkbox = QCheckBox("Resize Large Images (1080x720)")
        self.resize_large_checkbox.setChecked(True)
        controls_layout.addWidget(self.resize_large_checkbox)
        
        controls_layout.addSpacing(10)
        
        # 按钮区域
        self.play_video_button = QPushButton("Play Local Video")
        self.toggle_camera_button = QPushButton("Pause Camera")
        self.start_saving_button = QPushButton("Start Saving")
        self.stop_saving_button = QPushButton("Stop Saving")
        self.stop_saving_button.setEnabled(False)
        
        controls_layout.addWidget(self.play_video_button)
        controls_layout.addWidget(self.toggle_camera_button)
        controls_layout.addWidget(self.start_saving_button)
        controls_layout.addWidget(self.stop_saving_button)
        
        # 添加弹性空间
        controls_layout.addStretch()
        
        # 设置左侧控件区域的最大宽度
        self.controls_widget = QWidget()
        self.controls_widget.setLayout(controls_layout)
        self.controls_widget.setMaximumWidth(350)
        self.controls_widget.setMinimumWidth(300)
        
        # 右侧视频显示区域
        video_layout = QVBoxLayout()
        self.label = QLabel("Loading camera...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.label.mouseDoubleClickEvent = self.toggle_fullscreen
        # 设置视频显示大小（非固定，允许调整）
        self.label.setMinimumSize(640, 480)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.label)
        

        
        # 组装主布局
        main_layout.addWidget(self.controls_widget)
        main_layout.addLayout(video_layout)
        
        self.setLayout(main_layout)

        # 初始化检测器管理器
        self.detector_manager = DetectorManager()
        
        # 初始化视频相关变量
        self.frame_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.paused_event = threading.Event()
        self.model_switching_event = threading.Event()
        self.video_thread = VideoThread(self.frame_queue, self.stop_event, self.paused_event, self.model_switching_event)
        self.video_thread.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.writer = None
        self.segment_start_time = None
        self.playing_local_video = False
        self.camera_paused = False
        self.downsample_enabled = False
        self.resize_large_enabled = True
        


        # 连接信号
        self.start_saving_button.clicked.connect(self.start_saving)
        self.stop_saving_button.clicked.connect(self.stop_saving)
        self.play_video_button.clicked.connect(self.browse_and_play_video)
        self.toggle_camera_button.clicked.connect(self.toggle_camera)
        self.downsample_checkbox.stateChanged.connect(self.toggle_downsampling)
        self.resize_large_checkbox.stateChanged.connect(self.toggle_resize_large_images)

        # 初始化检测器
        self.change_detector("Grounding DINO")
        
        # 全屏状态
        self.is_fullscreen = False
        self.previous_geometry = None

        logging.info("MainWindow initialized")

    def toggle_downsampling(self, state):
        self.downsample_enabled = bool(state)
        logging.info(f"Downsampling {'enabled' if self.downsample_enabled else 'disabled'}")

    def toggle_resize_large_images(self, state):
        """切换大图像尺寸限制模式"""
        self.resize_large_enabled = bool(state)
        logging.info(f"Large image resizing {'enabled' if self.resize_large_enabled else 'disabled'}")

    def change_detector(self, model_name):
        """切换检测器模型"""
        try:
            # 设置模型切换标志，暂停视频线程
            self.model_switching_event.set()
            
            # 清空帧队列
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            # 切换模型
            if model_name == "Grounding DINO":
                weights_path = self.gdino_weights_combo.currentText()
                success = self.detector_manager.load_detector("grounding_dino", weights_path=weights_path)
                if success:
                    self.gdino_controls.show()
                    self.yolov5_controls.hide()
                    logging.info(f"Switched to Grounding DINO detector with weights: {weights_path}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to load Grounding DINO model")
            elif model_name == "CrocoDINO":
                weights_path = self.crocodino_weights_combo.currentText()
                success = self.detector_manager.load_detector("crocodino", weights_path=weights_path)
                if success:
                    self.gdino_controls.hide()
                    self.crocodino_controls.show()
                    self.yolov5_controls.hide()
                    logging.info(f"Switched to CrocoDINO detector with weights: {weights_path}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to load CrocoDINO model")
            elif model_name == "LGTrack":
                weights_path = self.weights_combo.currentText()
                success = self.detector_manager.load_detector("yolov5", weights_path=weights_path)
                if success:
                    self.gdino_controls.hide()
                    self.yolov5_controls.show()
                    logging.info(f"Switched to LGTrack detector with weights: {weights_path}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to load LGTrack model")
            
            # 清除模型切换标志，恢复视频线程
            self.model_switching_event.clear()
            
        except Exception as e:
            logging.error(f"Error changing detector to {model_name}: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load {model_name} detector: {str(e)}")
            # 确保清除标志
            self.model_switching_event.clear()

    def change_yolov5_weights(self, weights_name):
        """切换LGTrack权重"""
        try:
            # 设置模型切换标志，暂停视频线程
            self.model_switching_event.set()
            
            # 清空帧队列
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            success = self.detector_manager.load_detector("yolov5", weights_path=weights_name)
            if success:
                logging.info(f"Changed LGTrack weights to: {weights_name}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to load LGTrack weights {weights_name}")
            
            # 清除模型切换标志，恢复视频线程
            self.model_switching_event.clear()
        except Exception as e:
            logging.error(f"Error changing LGTrack weights to {weights_name}: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load LGTrack weights {weights_name}: {str(e)}")
            # 确保清除标志
            self.model_switching_event.clear()

    def change_gdino_weights(self, weights_name):
        """切换Grounding DINO权重"""
        try:
            # 设置模型切换标志，暂停视频线程
            self.model_switching_event.set()
            
            # 清空帧队列
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            success = self.detector_manager.load_detector("grounding_dino", weights_path=weights_name)
            if success:
                logging.info(f"Changed Grounding DINO weights to: {weights_name}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to load Grounding DINO weights {weights_name}")
            
            # 清除模型切换标志，恢复视频线程
            self.model_switching_event.clear()
        except Exception as e:
            logging.error(f"Error changing Grounding DINO weights to {weights_name}: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load Grounding DINO weights {weights_name}: {str(e)}")
            # 确保清除标志
            self.model_switching_event.clear()

    def change_crocodino_weights(self, weights_name):
        """切换CrocoDINO权重"""
        try:
            # 设置模型切换标志，暂停视频线程
            self.model_switching_event.set()
            
            # 清空帧队列
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            success = self.detector_manager.load_detector("crocodino", weights_path=weights_name)
            if success:
                logging.info(f"Changed CrocoDINO weights to: {weights_name}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to load CrocoDINO weights {weights_name}")
            
            # 清除模型切换标志，恢复视频线程
            self.model_switching_event.clear()
        except Exception as e:
            logging.error(f"Error changing CrocoDINO weights to {weights_name}: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load CrocoDINO weights {weights_name}: {str(e)}")
            # 确保清除标志
            self.model_switching_event.clear()



    def toggle_fullscreen(self, event=None):
        """切换全屏模式"""
        try:
            if self.is_fullscreen:
                # 退出全屏
                self.showNormal()
                if self.previous_geometry:
                    self.setGeometry(self.previous_geometry)
                # 恢复控件显示
                self.controls_widget.show()
                self.is_fullscreen = False
                logging.info("Exited fullscreen mode")
            else:
                # 进入全屏
                self.previous_geometry = self.geometry()
                # 隐藏左侧控件区域，只显示视频
                self.controls_widget.hide()
                self.showFullScreen()
                self.is_fullscreen = True
                logging.info("Entered fullscreen mode")
        except Exception as e:
            logging.error(f"Error toggling fullscreen: {e}")

    def keyPressEvent(self, event):
        """处理按键事件"""
        try:
            if event.key() == Qt.Key_Escape and self.is_fullscreen:
                # 按ESC键退出全屏
                self.toggle_fullscreen()
            elif event.key() == Qt.Key_F11:
                # 按F11键切换全屏
                self.toggle_fullscreen()
            else:
                super().keyPressEvent(event)
        except Exception as e:
            logging.error(f"Error in keyPressEvent: {e}")

    def update_frame(self):
        try:
            # 检查是否正在切换模型
            if self.model_switching_event.is_set():
                return
            
            if not self.frame_queue.empty() and not self.playing_local_video:
                frame = self.frame_queue.get()
                
                # 获取当前模型和参数
                current_model = self.model_combo.currentText()
                
                if current_model == "Grounding DINO":
                    prompt = self.prompt_input.text()
                    box_thresh = self.box_thresh_slider.value() / 100
                    text_thresh = self.text_thresh_slider.value() / 100
                    
                    if prompt:
                        if self.downsample_enabled:
                            frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                            frame_small = self.detector_manager.detect(
                                frame_small, 
                                text_prompt=prompt, 
                                box_threshold=box_thresh, 
                                text_threshold=text_thresh,
                                resize_large_images=self.resize_large_enabled
                            )
                            frame = cv2.resize(frame_small, (frame.shape[1], frame.shape[0]))
                        else:
                            frame = self.detector_manager.detect(
                                frame, 
                                text_prompt=prompt, 
                                box_threshold=box_thresh, 
                                text_threshold=text_thresh,
                                resize_large_images=self.resize_large_enabled
                            )
                
                elif current_model == "CrocoDINO":
                    conf_thresh = self.crocodino_conf_thresh_slider.value() / 100
                    
                    if self.downsample_enabled:
                        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        frame_small = self.detector_manager.detect(
                            frame_small, 
                            conf_threshold=conf_thresh,
                            resize_large_images=self.resize_large_enabled
                        )
                        frame = cv2.resize(frame_small, (frame.shape[1], frame.shape[0]))
                    else:
                        frame = self.detector_manager.detect(
                            frame, 
                            conf_threshold=conf_thresh,
                            resize_large_images=self.resize_large_enabled
                        )
                
                elif current_model == "LGTrack":
                    conf_thresh = self.conf_thresh_slider.value() / 100
                    iou_thresh = self.iou_thresh_slider.value() / 100
                    
                    if self.downsample_enabled:
                        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        frame_small = self.detector_manager.detect(
                            frame_small, 
                            conf_thres=conf_thresh, 
                            iou_thres=iou_thresh,
                            resize_large_images=self.resize_large_enabled
                        )
                        frame = cv2.resize(frame_small, (frame.shape[1], frame.shape[0]))
                    else:
                        frame = self.detector_manager.detect(
                            frame, 
                            conf_thres=conf_thresh, 
                            iou_thres=iou_thresh,
                            resize_large_images=self.resize_large_enabled
                        )

                if self.writer:
                    now = datetime.datetime.now()
                    if (now - self.segment_start_time).total_seconds() >= 20 * 60:
                        self.writer.release()
                        logging.info("Closing video segment due to time limit")
                        self.start_new_segment()
                    self.writer.write(frame)

                self.show_frame(frame)
        except Exception:
            logging.error("Error in update_frame:\n" + traceback.format_exc())

    def show_frame(self, frame):
        try:
            # 获取当前标签大小（支持全屏自适应）
            label_size = self.label.size()
            if label_size.width() <= 0 or label_size.height() <= 0:
                return
            
            # 计算缩放因子以适应标签大小，同时保持宽高比
            frame_h, frame_w = frame.shape[:2]
            scale_w = label_size.width() / frame_w
            scale_h = label_size.height() / frame_h
            scale = min(scale_w, scale_h)
            
            # 调整帧大小
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h))
            
            # 转换为RGB用于Qt
            rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 创建pixmap并缩放到标签大小
            pixmap = QPixmap.fromImage(qt_img)
            
            # 在标签中居中显示pixmap，支持自适应大小
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)
            
        except Exception:
            logging.error("Error in show_frame:\n" + traceback.format_exc())

    def browse_and_play_video(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
            if file_path:
                # 暂停视频线程，防止继续读取摄像头
                self.video_thread.playing_local_video_event.set()
                
                # 清空帧队列
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                self.timer.stop()
                self.playing_local_video = True
                logging.info(f"Playing local video: {file_path}")
                self.play_video_file(file_path)
                self.playing_local_video = False
                
                # 恢复视频线程
                self.video_thread.playing_local_video_event.clear()
                self.timer.start(30)
                logging.info("Finished playing local video")
        except Exception:
            logging.error("Error in browse_and_play_video:\n" + traceback.format_exc())
            # 确保恢复视频线程
            self.video_thread.playing_local_video_event.clear()

    def play_video_file(self, path):
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logging.error(f"Failed to open video file: {path}")
                QMessageBox.warning(self, "Error", f"Failed to open video: {path}")
                return
            current_model = self.model_combo.currentText()
            
            if current_model == "Grounding DINO":
                prompt = self.prompt_input.text()
                box_thresh = self.box_thresh_slider.value() / 100
                text_thresh = self.text_thresh_slider.value() / 100
            elif current_model == "CrocoDINO":
                conf_thresh = self.crocodino_conf_thresh_slider.value() / 100
            elif current_model == "LGTrack":
                conf_thresh = self.conf_thresh_slider.value() / 100
                iou_thresh = self.iou_thresh_slider.value() / 100
            
            while cap.isOpened() and self.playing_local_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if current_model == "Grounding DINO" and prompt:
                    if self.downsample_enabled:
                        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        frame_small = self.detector_manager.detect(
                            frame_small, 
                            text_prompt=prompt, 
                            box_threshold=box_thresh, 
                            text_threshold=text_thresh
                        )
                        frame = cv2.resize(frame_small, (frame.shape[1], frame.shape[0]))
                    else:
                        frame = self.detector_manager.detect(
                            frame, 
                            text_prompt=prompt, 
                            box_threshold=box_thresh, 
                            text_threshold=text_thresh
                        )
                elif current_model == "CrocoDINO":
                    if self.downsample_enabled:
                        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        frame_small = self.detector_manager.detect(
                            frame_small, 
                            conf_threshold=conf_thresh
                        )
                        frame = cv2.resize(frame_small, (frame.shape[1], frame.shape[0]))
                    else:
                        frame = self.detector_manager.detect(
                            frame, 
                            conf_threshold=conf_thresh
                        )
                elif current_model == "LGTrack":
                    if self.downsample_enabled:
                        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        frame_small = self.detector_manager.detect(
                            frame_small, 
                            conf_thres=conf_thresh, 
                            iou_thres=iou_thresh
                        )
                        frame = cv2.resize(frame_small, (frame.shape[1], frame.shape[0]))
                    else:
                        frame = self.detector_manager.detect(
                            frame, 
                            conf_thres=conf_thresh, 
                            iou_thres=iou_thresh
                        )
                
                self.show_frame(frame)
                QApplication.processEvents()
                time.sleep(1 / 30)
            cap.release()
        except Exception:
            logging.error("Error in play_video_file:\n" + traceback.format_exc())

    def toggle_camera(self):
        try:
            if self.playing_local_video:
                logging.info("Stopping local video playback due to camera toggle")
                self.playing_local_video = False
                # 恢复视频线程
                self.video_thread.playing_local_video_event.clear()
                self.timer.start(30)

            self.camera_paused = not self.camera_paused
            if self.camera_paused:
                self.paused_event.set()
                self.toggle_camera_button.setText("Resume Camera")
                logging.info("Camera paused")
            else:
                self.paused_event.clear()
                self.toggle_camera_button.setText("Pause Camera")
                logging.info("Camera resumed")
        except Exception:
            logging.error("Error in toggle_camera:\n" + traceback.format_exc())

    def start_saving(self):
        try:
            self.start_new_segment()
            self.start_saving_button.setEnabled(False)
            self.stop_saving_button.setEnabled(True)
            logging.info("Started saving video")
        except Exception:
            logging.error("Error in start_saving:\n" + traceback.format_exc())

    def start_new_segment(self):
        try:
            now = datetime.datetime.now()
            self.segment_start_time = now
            end_time = now + datetime.timedelta(minutes=20)
            save_folder = now.strftime('%Y%m%d')
            os.makedirs(save_folder, exist_ok=True)
            filename = save_folder + f"/{now.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%H%M%S')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*('XVID' if platform.system() == 'Windows' else 'mp4v'))
            self.writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            logging.info(f"Started new video segment: {filename}")
        except Exception:
            logging.error("Error in start_new_segment:\n" + traceback.format_exc())

    def stop_saving(self):
        try:
            if self.writer:
                self.writer.release()
                self.writer = None
                logging.info("Stopped saving video")
            self.start_saving_button.setEnabled(True)
            self.stop_saving_button.setEnabled(False)
        except Exception:
            logging.error("Error in stop_saving:\n" + traceback.format_exc())

    def closeEvent(self, event):
        logging.info("Application closing")
        try:
            self.stop_event.set()
            self.video_thread.release()
            self.video_thread.join()
            if self.writer:
                self.writer.release()
            logging.info("Resources released successfully")
        except Exception:
            logging.error("Error during application close:\n" + traceback.format_exc())
        event.accept()


if __name__ == "__main__":
    logging.info("Application started")
    
    # 如果需要无头模式，设置环境变量
    if platform.system() != "Windows" and not os.environ.get('DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        logging.info("Running in headless mode")

    
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"Failed to start GUI application: {e}")
        print(f"GUI Error: {e}")
        print("If you're running on a server without display, try:")
        print("export DISPLAY=:0")
        print("or")
        print("xhost +local:")
