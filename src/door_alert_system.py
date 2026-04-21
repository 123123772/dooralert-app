import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class DoorAlertSystem:
    def __init__(self, model_path: str, config_path: str = None):
        self.model = YOLO(model_path)
        # 默认参数
        self.FOCAL_LENGTH = 721.5
        self.REAL_HEIGHTS = {'car': 1.5, 'cyclist': 1.2, 'person': 1.7}
        self.WARNING_RULES = {'car': (10, 5), 'cyclist': (12, 6), 'person': (15, 8)}
        self.CONF_THRESH = {'car': 0.5, 'cyclist': 0.5, 'person': 0.3}
        self.CLASS_NAMES = {0: 'car', 1: 'cyclist', 2: 'person'}

        # 如果提供了配置文件，则覆盖默认参数
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                self.FOCAL_LENGTH = cfg.get('camera', {}).get('fy', 721.5)
                obj_h = cfg.get('object_heights', {})
                if obj_h:
                    self.REAL_HEIGHTS = obj_h
                warn_rules = cfg.get('warning_thresholds', {})
                if warn_rules:
                    self.WARNING_RULES = warn_rules
                conf_th = cfg.get('conf_thresholds', {})
                if conf_th:
                    self.CONF_THRESH = conf_th

    def estimate_distance(self, class_name: str, bbox_height_px: int) -> float:
        H = self.REAL_HEIGHTS.get(class_name, 1.5)
        if bbox_height_px <= 5:
            return 999.0
        return (self.FOCAL_LENGTH * H) / bbox_height_px

    def determine_warning_level(self, class_name: str, distance: float) -> int:
        warn_th, emerg_th = self.WARNING_RULES.get(class_name, (10, 5))
        if distance <= emerg_th:
            return 2
        elif distance <= warn_th:
            return 1
        else:
            return 0

    def process_frame(self, image: np.ndarray, conf_thresholds_override: dict = None):
        results = self.model(image, verbose=False)[0]

        # 获取置信度阈值（使用字符串键）
        if conf_thresholds_override:
            conf_car = conf_thresholds_override.get('car', self.CONF_THRESH.get('car', 0.5))
            conf_cyclist = conf_thresholds_override.get('cyclist', self.CONF_THRESH.get('cyclist', 0.5))
            conf_person = conf_thresholds_override.get('person', self.CONF_THRESH.get('person', 0.3))
        else:
            conf_car = self.CONF_THRESH.get('car', 0.5)
            conf_cyclist = self.CONF_THRESH.get('cyclist', 0.5)
            conf_person = self.CONF_THRESH.get('person', 0.3)

        detections = []
        img_h, img_w = image.shape[:2]

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.CLASS_NAMES.get(class_id, 'unknown')

            # 根据类别过滤置信度
            if class_name == 'car' and confidence < conf_car:
                continue
            if class_name == 'cyclist' and confidence < conf_cyclist:
                continue
            if class_name == 'person' and confidence < conf_person:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bbox_height = y2 - y1
            distance = self.estimate_distance(class_name, bbox_height)
            if distance > 30:
                continue

            warning_level = self.determine_warning_level(class_name, distance)
            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'distance': round(distance, 1),
                'warning_level': warning_level
            })

        annotated_img = self.draw_annotations(image.copy(), detections)
        return annotated_img, detections

    def draw_annotations(self, img: np.ndarray, detections: list) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            level = det['warning_level']
            if level == 2:
                color = (0, 0, 255)
            elif level == 1:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['distance']}m"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img