import os
import sys
import cv2
import torch
import numpy as np
import logging
import traceback
from abc import ABC, abstractmethod

def set_exclusive_module_path(subdir: str, other_paths: list, clear_modules: list = []):
    """Add `subdir` to sys.path and remove all `other_paths`. Optionally clear certain sys.modules."""
    base = os.path.dirname(__file__)
    sub_path = os.path.join(base, subdir)

    # Add selected subdir
    if sub_path not in sys.path:
        sys.path.insert(0, sub_path)

    # Remove others
    for other in other_paths:
        other_path = os.path.join(base, other)
        if other_path in sys.path:
            sys.path.remove(other_path)

    # Remove conflicting modules from sys.modules cache
    for mod in clear_modules:
        if mod in sys.modules:
            del sys.modules[mod]


class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self, **kwargs):
        """Load the detection model"""
        pass
    
    @abstractmethod
    def detect(self, frame, **kwargs):
        """Run detection on a frame"""
        pass
    
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.is_loaded


class GroundingDINODetector(BaseDetector):
    """Grounding DINO detector implementation"""
    
    def __init__(self):
        super().__init__()
        self.detector_type = "grounding_dino"
        self.weights_path = None
        
        # Set exclusive module path
        set_exclusive_module_path("groundingdino", other_paths=["crocodino", "yolov5"], clear_modules=[
            "util", "util.inference", "util.utils", "models", "models.build_model"
        ])
        
        try:
            from GroundingDINO.groundingdino.util.inference import load_model, transform_image, predict, annotate
            from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
            from GroundingDINO.groundingdino.util.slconfig import SLConfig
            from GroundingDINO.groundingdino.models import build_model
            
            # Store imported functions as instance variables
            self.load_model_func = load_model
            self.load_image_func = transform_image
            self.predict_func = predict
            self.annotate_func = annotate
            self.clean_state_dict_func = clean_state_dict
            self.get_phrases_from_posmap_func = get_phrases_from_posmap
            self.SLConfig = SLConfig
            self.build_model_func = build_model
            
        except ImportError as e:
            logging.error(f"Failed to import Grounding DINO modules: {e}")
            self.load_model_func = None
            self.load_image_func = None
            self.predict_func = None
            self.annotate_func = None
            self.clean_state_dict_func = None
            self.get_phrases_from_posmap_func = None
            self.SLConfig = None
            self.build_model_func = None
    
    def load_model(self, weights_path="groundingdino_swint_ogc.pth", **kwargs):
        """Load Grounding DINO model with specified weights"""
        try:
            if self.build_model_func is None:
                logging.error("Grounding DINO modules not imported")
                return False
            
            # Set weights path
            grounding_dino_path = os.path.join(os.path.dirname(__file__), "GroundingDINO", "groundingdino")
            self.weights_path = os.path.join(grounding_dino_path, weights_path)
            
            # Check if weights file exists
            if not os.path.exists(self.weights_path):
                logging.error(f"Grounding DINO weights file not found: {self.weights_path}")
                return False
            
            # Load model configuration
            model_config_path = os.path.join(grounding_dino_path, "config/GroundingDINO_SwinT_OGC.py")
            args = self.SLConfig.fromfile(model_config_path)
            args.device = self.device
            
            # Build model
            self.model = self.build_model_func(args)
            
            # Load pre-trained weights
            checkpoint = torch.load(self.weights_path, map_location='cpu')
            self.model.load_state_dict(self.clean_state_dict_func(checkpoint['model']), strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)
            self.is_loaded = True
            logging.info(f"Grounding DINO model loaded successfully with weights: {weights_path}")
            return True
                
        except Exception as e:
            logging.error(f"Error loading Grounding DINO: {e}")
            return False
    
    def detect(self, frame, text_prompt="", box_threshold=0.3, text_threshold=0.25, **kwargs):
        """Run Grounding DINO detection"""
        try:
            if not self.is_loaded or self.model is None:
                return frame
            
            if not text_prompt:
                return frame
            
            if self.load_image_func is None or self.predict_func is None or self.annotate_func is None:
                logging.error("Grounding DINO modules not imported")
                return frame
            
            # Store original dimensions
            original_height, original_width = frame.shape[:2]
            
            # Resize if image is larger than 1080x720 (only if resize_large_images is True)
            max_width, max_height = 1080, 720
            resized_frame = frame
            scale_x, scale_y = 1.0, 1.0
            
            # Check if resize_large_images option is enabled (default True)
            resize_large_images = kwargs.get('resize_large_images', True)
            
            if resize_large_images and (original_width > max_width or original_height > max_height):
                # Calculate scaling factors to fit within 1080x720 while maintaining aspect ratio
                scale_x = max_width / original_width
                scale_y = max_height / original_height
                scale = min(scale_x, scale_y)
                
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logging.debug(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
            
            # Convert image format
            image_source = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Load and process image
            image = self.load_image_func(image_source)
            
            # Run prediction
            boxes, logits, phrases = self.predict_func(
                model=self.model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
            
            # Draw results
            if len(boxes) > 0:
                annotated_frame = self.annotate_func(image_source, boxes, logits, phrases)
                result_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Resize back to original size if we resized the input
                if original_width > max_width or original_height > max_height:
                    result_frame = cv2.resize(result_frame, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
                
                return result_frame
            
            # If no detections, resize back to original size if we resized the input
            if original_width > max_width or original_height > max_height:
                resized_frame = cv2.resize(resized_frame, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
                return resized_frame
            
            return frame
            
        except Exception as e:
            logging.error(f"Error in Grounding DINO detection: {e}")
            return frame


class CrocoDINODetector(BaseDetector):
    """CrocoDINO detector implementation based on DINO model"""
    
    def __init__(self):
        super().__init__()
        self.detector_type = "crocodino"
        self.model = None
        self.postprocessors = None
        self.id2name = None
        
        # 🧼 Set exclusive path to crocodino, clear others (like yolov5 or groundingdino)
        set_exclusive_module_path("crocodino", other_paths=["groundingdino", "yolov5"], clear_modules=[
            "util", "models.experimental", "utils", "utils.general", "utils.torch_utils"
        ])
        
        # Import DINO modules
        try:
            from crocodino.main import build_model_main
            from crocodino.util.slconfig import SLConfig
            from crocodino.util.visualizer import COCOVisualizer, draw_coco_annotations
            from crocodino.util import box_ops
            import json
            
            self.build_model_main = build_model_main
            self.SLConfig = SLConfig
            self.COCOVisualizer = COCOVisualizer
            self.draw_coco_annotations = draw_coco_annotations
            self.box_ops = box_ops
            
            # Load COCO class names
            coco_names_path = os.path.join(os.path.dirname(__file__), "crocodino", "util/coco_id2name.json")
            if os.path.exists(coco_names_path):
                with open(coco_names_path, 'r') as f:
                    self.id2name = json.load(f)
                    self.id2name = {int(k): v for k, v in self.id2name.items()}
            
            logging.info("CrocoDINODetector initialized successfully")
            
        except ImportError as e:
            logging.error(f"Failed to import DINO modules: {e}")
            self.build_model_main = None
            self.SLConfig = None
            self.COCOVisualizer = None
            self.draw_coco_annotations = None
            self.box_ops = None
            self.id2name = None
    
    def load_model(self, weights_path="crocodino_moderate.pth", config_path="config/DINO/DINO_4scale_croco.py", **kwargs):
        """Load CrocoDINO model with specified weights and config"""
        try:
            if self.build_model_main is None:
                logging.error("DINO modules not imported")
                return False
            
            # Set paths
            crocodino_path = os.path.join(os.path.dirname(__file__), "crocodino")
            self.weights_path = os.path.join(crocodino_path, weights_path)
            self.config_path = os.path.join(crocodino_path, config_path)
            
            # Check if files exist
            if not os.path.exists(self.weights_path):
                logging.error(f"CrocoDINO weights file not found: {self.weights_path}")
                return False
            
            if not os.path.exists(self.config_path):
                logging.error(f"CrocoDINO config file not found: {self.config_path}")
                return False
            
            # Load model configuration
            args = self.SLConfig.fromfile(self.config_path)
            args.device = self.device
            
            # Build model
            self.model, criterion, self.postprocessors = self.build_model_main(args)
            
            # Load weights
            checkpoint = torch.load(self.weights_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()
            self.model = self.model.to(self.device)
            
            self.is_loaded = True
            logging.info(f"CrocoDINO model loaded successfully with weights: {weights_path}")
            return True
                
        except Exception as e:
            logging.error(f"Error loading CrocoDINO: {e}")
            return False
    
    def detect(self, frame, conf_threshold=0.3, **kwargs):
        """Run CrocoDINO detection"""
        try:
            if not self.is_loaded or self.model is None:
                return frame
            
            if self.postprocessors is None or self.id2name is None:
                logging.error("CrocoDINO modules not properly loaded")
                return frame
            
            # Store original dimensions
            original_height, original_width = frame.shape[:2]
            
            # Resize if image is larger than 1080x720 (only if resize_large_images is True)
            max_width, max_height = 1080, 720
            resized_frame = frame
            scale_x, scale_y = 1.0, 1.0
            
            # Check if resize_large_images option is enabled (default True)
            resize_large_images = kwargs.get('resize_large_images', True)
            
            if resize_large_images and (original_width > max_width or original_height > max_height):
                # Calculate scaling factors to fit within 1080x720 while maintaining aspect ratio
                scale_x = max_width / original_width
                scale_y = max_height / original_height
                scale = min(scale_x, scale_y)
                
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logging.debug(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
            
            # Convert BGR to RGB and normalize
            image_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to CHW and add batch dimension
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(image_tensor)
                output = self.postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).to(self.device))[0]
                      
            scores = output['scores']
            boxes = self.box_ops.box_xyxy_to_cxcywh(output['boxes'])
            select_mask = scores > conf_threshold

            pred_dict = {
                'boxes': boxes[select_mask],
                'size': torch.Tensor([image_tensor.squeeze().shape[1], image_tensor.squeeze().shape[2]]),
                'scores': scores[select_mask],
            }

            return self.draw_coco_annotations(image_tensor.cpu().squeeze(), pred_dict, dpi=50)
            
            
        except Exception as e:
            logging.error(f"Error in CrocoDINO detection: {e}")
            return frame


class YOLOv5Detector(BaseDetector):
    """YOLOv5 detector implementation using detect.py script"""
    
    def __init__(self):
        super().__init__()
        self.detector_type = "yolov5"
        self.model = None
        self.stride = None
        self.names = None
        self.pt = None
        
        # 🧼 Set exclusive path for yolov5, remove others like crocodino or groundingdino
        set_exclusive_module_path("yolov5", other_paths=["crocodino", "groundingdino"], clear_modules=[
            "models.experimental", "models.common", "utils", "utils.general", "utils.torch_utils"
        ])
        
        # Import YOLOv5 modules
        try:
            from yolov5.models.common import DetectMultiBackend
            from yolov5.utils.general import non_max_suppression, scale_boxes
            from yolov5.utils.torch_utils import select_device
            from ultralytics.utils.plotting import Annotator, colors
            
            self.DetectMultiBackend = DetectMultiBackend
            self.non_max_suppression = non_max_suppression
            self.scale_boxes = scale_boxes
            self.select_device = select_device
            self.Annotator = Annotator
            self.colors = colors
            
            logging.info("YOLOv5Detector initialized successfully")
            
        except ImportError as e:
            logging.error(f"Failed to import YOLOv5 modules: {e}")
            self.DetectMultiBackend = None
            self.non_max_suppression = None
            self.scale_boxes = None
            self.select_device = None
            self.Annotator = None
            self.colors = None
    
    def load_model(self, weights_path="yolov5s.pt", **kwargs):
        """Load YOLOv5 model"""
        try:
            if self.DetectMultiBackend is None:
                logging.error("YOLOv5 modules not imported")
                return False
            
            # Set device
            self.device = self.select_device('')
            
            # Load model
            yolov5_path = os.path.join(os.path.dirname(__file__), "yolov5")
            full_weights_path = os.path.join(yolov5_path, weights_path)
            
            if not os.path.exists(full_weights_path):
                logging.error(f"YOLOv5 weights file not found: {full_weights_path}")
                return False
            
            self.model = self.DetectMultiBackend(full_weights_path, device=self.device)
            self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
            
            self.is_loaded = True
            logging.info(f"YOLOv5 model loaded successfully with weights: {weights_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading YOLOv5: {e}")
            return False

    def detect(self, frame, conf_thres=0.25, iou_thres=0.45, **kwargs):
        def scale_boxes(img_shape, boxes, frame_shape):
            """
            Scale bounding boxes from model input size (img_shape) to original frame size (frame_shape).
            Matches cv2.resize(frame, (640, 640)) preprocessing without aspect ratio preservation.
            
            Args:
                img_shape (tuple): Model input size (height, width), typically (640, 640)
                boxes (torch.Tensor): Bounding boxes in format [x1, y1, x2, y2]
                frame_shape (tuple): Original frame size (height, width)
            
            Returns:
                torch.Tensor: Scaled bounding boxes
            """
            # Calculate scaling factors for x and y separately
            scale_x = img_shape[1] / frame_shape[1]  # 640 / original_width
            scale_y = img_shape[0] / frame_shape[0]  # 640 / original_height
            
            # Scale boxes
            boxes[:, [0, 2]] /= scale_x  # Scale x1, x2
            boxes[:, [1, 3]] /= scale_y  # Scale y1, y2
            
            # Clip boxes to frame boundaries
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_shape[0])
            
            return boxes.round()
        
        """Run YOLOv5 detection using OpenCV for box drawing"""
        if not self.is_loaded or self.model is None:
            return frame
        
        if (self.non_max_suppression is None or self.scale_boxes is None or 
            self.colors is None or self.names is None):
            logging.error("YOLOv5 modules not imported")
            return frame
        
        # Preprocess image for YOLOv5 (resize to 640x640)
        im = cv2.resize(frame, (640, 640))
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred = self.model(im, augment=False, visualize=False)
            pred = self.non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)
                    
        for det in pred:
            if len(det):
                # Scale boxes from 640x640 back to original image size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    # label = f'{self.names[c]} {conf:.2f}'
                    label = f'{conf:.2f}'
                    
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Draw rectangle
                    color = self.colors(c, True)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
                    
                    # Draw text
                    cv2.putText(frame, label, (x1, y1 - 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

'''
    def detect(self, frame, conf_thres=0.25, iou_thres=0.45, **kwargs):
        """Run YOLOv5 detection using detect.py approach"""
        try:
            if not self.is_loaded or self.model is None:
                return frame
            
            if (self.non_max_suppression is None or self.scale_boxes is None or 
                self.Annotator is None or self.colors is None):
                logging.error("YOLOv5 modules not imported")
                return frame
            
            # Preprocess image for YOLOv5 (resize to 640x640)
            im = cv2.resize(frame, (640, 640))
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(self.device)
            im = im.float() / 255.0
            if im.ndimension() == 3:
                im = im.unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                pred = self.model(im, augment=False, visualize=False)
                pred = self.non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)
                       
            for det in pred:
                if len(det):
                    annotator = self.Annotator(frame, line_width=3, example=str(self.names))
                    # Scale boxes from 640x640 back to original image size
                    # This is the same logic as in detect.py
                    det[:, :4] = self.scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                    
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = f'{conf:.2f}'
                        annotator.box_label(xyxy, label, color=self.colors(c, True))
                    
                    frame = annotator.result()
            
            return frame
            
        except Exception as e:
            logging.error(f"Error in YOLOv5 detection: {e}")
            return frame
'''


class DetectorManager:
    """Manager class for handling different detectors"""
    
    def __init__(self):
        self.current_detector = None
        self.detector_type = None
        self._detector_instances = {}
    
    def _get_detector_instance(self, detector_type):
        """Get or create detector instance"""
        if detector_type not in self._detector_instances:
            if detector_type == "grounding_dino":
                self._detector_instances[detector_type] = GroundingDINODetector()
            elif detector_type == "crocodino":
                self._detector_instances[detector_type] = CrocoDINODetector()
            elif detector_type == "yolov5":
                self._detector_instances[detector_type] = YOLOv5Detector()
            else:
                logging.error(f"Unknown detector type: {detector_type}")
                return None
        return self._detector_instances[detector_type]
    
    def load_detector(self, detector_type, **kwargs):
        """Load a specific detector"""
        try:
            # Clear current detector first
            self.clear_detector()
            
            # Get detector instance
            detector = self._get_detector_instance(detector_type)
            if detector is None:
                return False
            
            # Load model with isolated import environment
            success = detector.load_model(**kwargs)
            if success:
                self.current_detector = detector
                self.detector_type = detector_type
                logging.info(f"Successfully loaded {detector_type} detector")
            
            return success
            
        except Exception as e:
            logging.error(f"Error loading detector {detector_type}: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def detect(self, frame, **kwargs):
        """Run detection using current detector"""
        if self.current_detector is None or not self.current_detector.is_model_loaded():
            return frame
        
        return self.current_detector.detect(frame, **kwargs)
    
    def get_detector_type(self):
        """Get current detector type"""
        return self.detector_type
    
    def is_detector_loaded(self):
        """Check if detector is loaded"""
        return self.current_detector is not None and self.current_detector.is_model_loaded()
    
    def clear_detector(self):
        """Clear current detector to free memory"""
        if self.current_detector is not None:
            # Clear model from memory
            if hasattr(self.current_detector, 'model') and self.current_detector.model is not None:
                del self.current_detector.model
                self.current_detector.model = None
            
            self.current_detector.is_loaded = False
            self.current_detector = None
            self.detector_type = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logging.info("Detector cleared from memory")
